from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoImageProcessor,
    AutoModel,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)

from .config import (
    DETECT_MODEL_NAME,
    DETECTION_LABELS,
    DEVICE,
    TEXT_MODEL_NAME,
    VISION_MODEL_NAME,
)
from .image_utils import center_crop_square, expand_box, extract_visual_attrs
from .text_utils import detect_family_from_label, group_from_family


_text_model = None
_detector_processor = None
_detector_model = None
_vision_processor = None
_vision_model = None


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm <= 1e-9:
        return vector

    return vector / norm


def load_all_models():
    get_text_model()
    get_detector()
    get_vision_model()


def get_text_model():
    global _text_model

    if _text_model is None:
        print(f"Loading text model: {TEXT_MODEL_NAME}")
        _text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)

    return _text_model


def get_detector():
    global _detector_processor, _detector_model

    if _detector_processor is None or _detector_model is None:
        print(f"Loading object detector: {DETECT_MODEL_NAME}")
        _detector_processor = OwlViTProcessor.from_pretrained(DETECT_MODEL_NAME)
        _detector_model = OwlViTForObjectDetection.from_pretrained(DETECT_MODEL_NAME)
        _detector_model.to(DEVICE)
        _detector_model.eval()

    return _detector_processor, _detector_model


def get_vision_model():
    global _vision_processor, _vision_model

    if _vision_processor is None or _vision_model is None:
        print(f"Loading vision model: {VISION_MODEL_NAME}")
        _vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
        _vision_model = AutoModel.from_pretrained(VISION_MODEL_NAME)
        _vision_model.to(DEVICE)
        _vision_model.eval()

    return _vision_processor, _vision_model


def encode_texts_sync(texts: List[str]) -> np.ndarray:
    return get_text_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")


def encode_image_sync(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")

    processor, model = get_vision_model()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {
        key: value.to(DEVICE) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        vector = outputs.pooler_output[0]
    elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        vector = outputs.image_embeds[0]
    else:
        vector = outputs.last_hidden_state[:, 0, :][0]

    vector = vector.detach().cpu().numpy().astype("float32")
    return l2_normalize(vector)


def detect_query_family_by_owl(image: Image.Image) -> Tuple[str, str, str, float]:
    image = image.convert("RGB")

    try:
        processor, model = get_detector()

        inputs = processor(
            text=[DETECTION_LABELS],
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)

        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.025
        )[0]

        scores = results.get("scores", [])
        labels = results.get("labels", [])

        family_scores: Dict[str, Tuple[float, str]] = {}

        for score_tensor, label_tensor in zip(scores, labels):
            score = float(score_tensor.detach().cpu().item())
            label_id = int(label_tensor.detach().cpu().item())

            if label_id < 0 or label_id >= len(DETECTION_LABELS):
                continue

            label = DETECTION_LABELS[label_id]
            family = detect_family_from_label(label)

            if not family:
                continue

            old = family_scores.get(family)

            if old is None or score > old[0]:
                family_scores[family] = (score, label)

        if not family_scores:
            return "", "", "", 0.0

        best_family, (best_score, best_label) = max(
            family_scores.items(),
            key=lambda item: item[1][0]
        )

        best_group = group_from_family(best_family)

        print(
            "QUERY FAMILY DETECT:",
            "family =", best_family,
            "group =", best_group,
            "label =", best_label,
            "score =", round(best_score, 4),
            "allFamilies =", {
                family: {
                    "score": round(value[0], 4),
                    "label": value[1],
                }
                for family, value in family_scores.items()
            }
        )

        if best_score < 0.025:
            return "", "", best_label, best_score

        return best_group, best_family, best_label, best_score

    except Exception as e:
        print("QUERY FAMILY DETECT ERROR:", repr(e))
        return "", "", "", 0.0


def crop_main_product(image: Image.Image) -> Tuple[Image.Image, str, float]:
    image = image.convert("RGB")

    try:
        processor, model = get_detector()

        inputs = processor(
            text=[DETECTION_LABELS],
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)

        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.08
        )[0]

        scores = results.get("scores", [])
        boxes = results.get("boxes", [])
        labels = results.get("labels", [])

        if len(scores) == 0:
            return center_crop_square(image), "center-crop", 0.0

        best_index = int(torch.argmax(scores).item())
        best_score = float(scores[best_index].detach().cpu().item())
        best_box = boxes[best_index].detach().cpu().tolist()
        best_label_id = int(labels[best_index].detach().cpu().item())

        label = DETECTION_LABELS[best_label_id] if 0 <= best_label_id < len(DETECTION_LABELS) else "product"

        x1, y1, x2, y2 = expand_box(
            best_box,
            image_width=image.size[0],
            image_height=image.size[1],
            margin_ratio=0.12
        )

        if x2 <= x1 or y2 <= y1:
            return center_crop_square(image), "center-crop", 0.0

        cropped = image.crop((x1, y1, x2, y2))

        print(
            "DETECT:",
            "label =", label,
            "score =", round(best_score, 4),
            "box =", [x1, y1, x2, y2]
        )

        return cropped, label, best_score

    except Exception as e:
        print("DETECT ERROR:", repr(e))
        return center_crop_square(image), "center-crop", 0.0


def make_visual_views(image: Image.Image) -> List[Tuple[Image.Image, str, Dict[str, object]]]:
    image = image.convert("RGB")

    views: List[Tuple[Image.Image, str, Dict[str, object]]] = []

    views.append((image, "full", extract_visual_attrs(image)))

    center = center_crop_square(image)
    views.append((center, "center", extract_visual_attrs(center)))

    try:
        cropped, label, score = crop_main_product(image)

        if cropped is not None:
            views.append((
                cropped,
                f"detected-{label}-{round(score, 3)}",
                extract_visual_attrs(cropped)
            ))
    except Exception as e:
        print("MAKE VISUAL VIEWS ERROR:", repr(e))

    result = []
    seen = set()

    for view, name, attrs in views:
        w, h = view.size

        if w < 40 or h < 40:
            continue

        key = (w // 10, h // 10, name.split("-")[0])

        if key in seen:
            continue

        seen.add(key)
        result.append((view.convert("RGB"), name, attrs))

    return result


def build_image_vector_items_sync(image: Image.Image) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []

    for view_image, view_name, attrs in make_visual_views(image):
        vector = encode_image_sync(view_image)

        items.append({
            "vector": l2_normalize(vector.astype("float32")),
            "view": view_name,
            "attrs": attrs,
        })

    return items