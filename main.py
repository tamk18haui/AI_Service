import asyncio
import base64
import hashlib
import json
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import torch
from cachetools import TTLCache
from fastapi import FastAPI, File, Form, Request, UploadFile
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoImageProcessor,
    AutoModel,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)


# =========================
# CONFIG
# =========================

TEXT_MODEL_NAME = os.getenv(
    "TEXT_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "facebook/dinov2-base")
DETECT_MODEL_NAME = os.getenv("DETECT_MODEL_NAME", "google/owlvit-base-patch32")

# Ngưỡng cũ 0.42 quá cao với ảnh chụp màn hình / ảnh crop từ app.
# Log của bạn có bestScore khoảng 0.27 - 0.28 nên để 0.24 hợp lý hơn.
IMAGE_MATCH_THRESHOLD = float(os.getenv("IMAGE_MATCH_THRESHOLD", "0.24"))
IMAGE_LOW_THRESHOLD = float(os.getenv("IMAGE_LOW_THRESHOLD", "0.18"))
MAX_IMAGE_RESULTS = int(os.getenv("MAX_IMAGE_RESULTS", "12"))
VISION_ALLOW_WEAK_FALLBACK = os.getenv("VISION_ALLOW_WEAK_FALLBACK", "true").lower() == "true"

DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT", "12"))
DOWNLOAD_CONCURRENCY = int(os.getenv("DOWNLOAD_CONCURRENCY", "8"))
MAX_IMAGES_PER_PRODUCT = int(os.getenv("MAX_IMAGES_PER_PRODUCT", "6"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lưu embedding ảnh ra ổ đĩa để lần sau không phải download/encode lại.
EMBEDDING_DATA_DIR = Path(os.getenv("EMBEDDING_DATA_DIR", "./smartcart_ai_data"))
IMAGE_VECTOR_DIR = EMBEDDING_DATA_DIR / "image_vectors"
IMAGE_INDEX_META_PATH = EMBEDDING_DATA_DIR / "image_index_meta.json"
IMAGE_INDEX_MATRIX_PATH = EMBEDDING_DATA_DIR / "image_index_matrix.npy"

EMBEDDING_DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_VECTOR_DIR.mkdir(parents=True, exist_ok=True)

DETECTION_LABELS = [
    "a product",
    "clothing",
    "shirt",
    "t-shirt",
    "pants",
    "dress",
    "shoes",
    "sneakers",
    "bag",
    "backpack",
    "handbag",
    "phone",
    "watch",
    "laptop",
    "headphones",
    "cosmetic",
    "bottle",
    "toy",
    "book",
    "food package",
    "electronic device",
]


# =========================
# GLOBAL MODELS + INDEX
# =========================

_text_model = None
_detector_processor = None
_detector_model = None
_vision_processor = None
_vision_model = None

_index_lock = asyncio.Lock()

_product_meta: Dict[int, "ProductCandidate"] = {}
_product_ids: List[int] = []
_product_text_embeddings: Optional[np.ndarray] = None

# Mỗi productId có thể xuất hiện nhiều lần vì 1 sản phẩm có nhiều ảnh/vector.
_image_product_ids: List[int] = []
_product_image_embeddings: Optional[np.ndarray] = None

# RAM cache: URL ảnh -> List vector ảnh.
_image_embedding_cache: TTLCache = TTLCache(maxsize=3000, ttl=60 * 60 * 12)


# =========================
# DTO
# =========================

class ProductCandidate(BaseModel):
    productId: int
    text: str = ""
    imageUrl: Optional[str] = None
    imageUrls: List[str] = []
    soldCount: int = 0
    rating: float = 0.0
    reviewCount: int = 0


class RecommendItem(BaseModel):
    productId: int
    score: float
    reason: str


class RecommendResponse(BaseModel):
    items: List[RecommendItem]
    page: int
    size: int
    totalElements: int
    totalPages: int
    hasMore: bool


# =========================
# MODEL LOADERS
# =========================

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI Service starting...")
    await asyncio.to_thread(load_all_models)

    # Load lại index ảnh đã lưu trước đó.
    # Nếu file tồn tại thì search ảnh có thể chạy ngay, không cần download lại toàn bộ ảnh.
    await asyncio.to_thread(load_image_index_state)

    print("AI Service ready.")
    yield
    print("AI Service stopped.")


app = FastAPI(
    title="SmartCart AI Service",
    lifespan=lifespan
)


# =========================
# BASIC UTILS
# =========================

def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def normalize_number(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return min(max(value / max_value, 0.0), 1.0)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm <= 1e-9:
        return vector

    return vector / norm


def cosine_similarity_matrix(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    if matrix is None or len(matrix) == 0:
        return np.array([], dtype=np.float32)

    query = l2_normalize(query.astype("float32"))
    return np.dot(matrix, query)


def empty_response(page: int, size: int) -> RecommendResponse:
    page = max(page, 0)
    size = max(min(size, 50), 1)

    return RecommendResponse(
        items=[],
        page=page,
        size=size,
        totalElements=0,
        totalPages=0,
        hasMore=False
    )


def paginate(items: List[RecommendItem], page: int, size: int) -> RecommendResponse:
    page = max(page, 0)
    size = max(min(size, 50), 1)

    total = len(items)
    total_pages = (total + size - 1) // size if total > 0 else 0

    start = page * size
    end = start + size

    return RecommendResponse(
        items=items[start:end],
        page=page,
        size=size,
        totalElements=total,
        totalPages=total_pages,
        hasMore=end < total
    )


def clean_url(url: str) -> str:
    return str(url or "").strip().replace("\\", "")


def split_image_urls(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        raw = str(value or "").strip()

        if not raw:
            return []

        if raw.startswith("[") and raw.endswith("]"):
            try:
                raw_items = json.loads(raw)
            except Exception:
                raw_items = raw.replace("[", "").replace("]", "").replace('"', "").split(",")
        else:
            raw_items = raw.split(",")

    urls: List[str] = []

    for item in raw_items:
        url = clean_url(str(item or "").strip().replace('"', ""))

        if not url:
            continue

        if url.startswith("http://") or url.startswith("https://"):
            urls.append(url)

    result: List[str] = []
    seen = set()

    for url in urls:
        if url in seen:
            continue

        seen.add(url)
        result.append(url)

    return result


def get_candidate_image_urls(candidate: ProductCandidate) -> List[str]:
    urls: List[str] = []

    urls.extend(split_image_urls(candidate.imageUrl))
    urls.extend(split_image_urls(candidate.imageUrls))

    result: List[str] = []
    seen = set()

    for url in urls:
        if url in seen:
            continue

        seen.add(url)
        result.append(url)

        if len(result) >= MAX_IMAGES_PER_PRODUCT:
            break

    return result


# =========================
# PARSE CANDIDATES
# =========================

def parse_candidate(raw: Dict[str, Any]) -> Optional[ProductCandidate]:
    product_id = raw.get("productId") or raw.get("id")

    if product_id is None:
        return None

    image_urls: List[str] = []
    image_urls.extend(split_image_urls(raw.get("imageUrl")))
    image_urls.extend(split_image_urls(raw.get("imageUrls")))

    return ProductCandidate(
        productId=safe_int(product_id),
        text=str(raw.get("text") or ""),
        imageUrl=raw.get("imageUrl"),
        imageUrls=image_urls,
        soldCount=safe_int(raw.get("soldCount"), 0),
        rating=safe_float(raw.get("rating"), 0.0),
        reviewCount=safe_int(raw.get("reviewCount"), 0)
    )


def parse_candidates(raw_candidates: Any) -> List[ProductCandidate]:
    if raw_candidates is None:
        return []

    if isinstance(raw_candidates, str):
        try:
            raw_candidates = json.loads(raw_candidates)
        except Exception:
            return []

    if not isinstance(raw_candidates, list):
        return []

    result: List[ProductCandidate] = []

    for item in raw_candidates:
        if not isinstance(item, dict):
            continue

        candidate = parse_candidate(item)

        if candidate is not None and candidate.productId > 0:
            result.append(candidate)

    return result


# =========================
# IMAGE PROCESSING
# =========================

def center_crop_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)

    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side

    return image.crop((left, top, right, bottom))


def expand_box(
        box: List[float],
        image_width: int,
        image_height: int,
        margin_ratio: float = 0.12
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    margin_x = box_w * margin_ratio
    margin_y = box_h * margin_ratio

    x1 = max(0, int(x1 - margin_x))
    y1 = max(0, int(y1 - margin_y))
    x2 = min(image_width, int(x2 + margin_x))
    y2 = min(image_height, int(y2 + margin_y))

    return x1, y1, x2, y2


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

        label_name = DETECTION_LABELS[best_label_id] if 0 <= best_label_id < len(DETECTION_LABELS) else "product"

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
            "label =", label_name,
            "score =", round(best_score, 4),
            "box =", [x1, y1, x2, y2]
        )

        return cropped, label_name, best_score

    except Exception as e:
        print("DETECT ERROR:", repr(e))
        return center_crop_square(image), "center-crop", 0.0


def make_visual_views(image: Image.Image) -> List[Tuple[Image.Image, str]]:
    image = image.convert("RGB")

    views: List[Tuple[Image.Image, str]] = []

    views.append((image, "full"))
    views.append((center_crop_square(image), "center"))

    try:
        cropped, label, score = crop_main_product(image)

        if cropped is not None:
            views.append((cropped, f"detected-{label}-{round(score, 3)}"))

    except Exception as e:
        print("MAKE VISUAL VIEWS DETECT ERROR:", repr(e))

    result: List[Tuple[Image.Image, str]] = []
    seen = set()

    for view, name in views:
        if view is None:
            continue

        w, h = view.size

        if w < 40 or h < 40:
            continue

        key = (w // 10, h // 10, name.split("-")[0])

        if key in seen:
            continue

        seen.add(key)
        result.append((view.convert("RGB"), name))

    return result


# =========================
# ENCODERS
# =========================

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
    else:
        vector = outputs.last_hidden_state[:, 0, :][0]

    vector = vector.detach().cpu().numpy().astype("float32")
    return l2_normalize(vector)


def build_image_vectors_sync(image: Image.Image) -> List[np.ndarray]:
    vectors: List[np.ndarray] = []

    image = image.convert("RGB")

    try:
        views = make_visual_views(image)

        for view_image, view_name in views:
            vector = encode_image_sync(view_image)
            vectors.append(l2_normalize(vector.astype("float32")))

    except Exception as e:
        print("BUILD IMAGE VECTORS ERROR:", repr(e))

    return vectors


# =========================
# PERSISTENT IMAGE CACHE
# =========================

def image_cache_key(url: str) -> str:
    raw = f"{VISION_MODEL_NAME}|{clean_url(url)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def image_vector_cache_path(url: str) -> Path:
    return IMAGE_VECTOR_DIR / f"{image_cache_key(url)}.npz"


def load_image_vectors_from_disk(url: str) -> Optional[List[np.ndarray]]:
    path = image_vector_cache_path(url)

    if not path.exists():
        return None

    try:
        data = np.load(path)
        vectors: List[np.ndarray] = []

        for key in sorted(data.files):
            vector = data[key].astype("float32")
            vector = l2_normalize(vector)
            vectors.append(vector)

        if not vectors:
            return None

        return vectors

    except Exception as e:
        print("LOAD IMAGE VECTOR CACHE ERROR:", url, repr(e))
        return None


def save_image_vectors_to_disk(url: str, vectors: List[np.ndarray]) -> None:
    if not vectors:
        return

    path = image_vector_cache_path(url)
    tmp_path = path.with_suffix(".tmp.npz")

    try:
        arrays = {
            f"v{i}": l2_normalize(vector.astype("float32"))
            for i, vector in enumerate(vectors)
        }

        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)

    except Exception as e:
        print("SAVE IMAGE VECTOR CACHE ERROR:", url, repr(e))

        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def candidate_to_dict(candidate: ProductCandidate) -> Dict[str, Any]:
    return {
        "productId": candidate.productId,
        "text": candidate.text,
        "imageUrl": candidate.imageUrl,
        "imageUrls": candidate.imageUrls,
        "soldCount": candidate.soldCount,
        "rating": candidate.rating,
        "reviewCount": candidate.reviewCount,
    }


def save_image_index_state() -> None:
    global _product_image_embeddings
    global _image_product_ids
    global _product_meta

    try:
        if _product_image_embeddings is None or not _image_product_ids:
            return

        np.save(IMAGE_INDEX_MATRIX_PATH, _product_image_embeddings.astype("float32"))

        meta = {
            "visionModel": VISION_MODEL_NAME,
            "imageProductIds": _image_product_ids,
            "productMeta": {
                str(product_id): candidate_to_dict(candidate)
                for product_id, candidate in _product_meta.items()
            }
        }

        with open(IMAGE_INDEX_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        print(
            "SAVE IMAGE INDEX STATE:",
            "vectors =", len(_image_product_ids),
            "products =", len(set(_image_product_ids))
        )

    except Exception as e:
        print("SAVE IMAGE INDEX STATE ERROR:", repr(e))


def load_image_index_state() -> None:
    global _product_meta
    global _image_product_ids
    global _product_image_embeddings

    try:
        if not IMAGE_INDEX_META_PATH.exists():
            return

        if not IMAGE_INDEX_MATRIX_PATH.exists():
            return

        with open(IMAGE_INDEX_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("visionModel") != VISION_MODEL_NAME:
            print("SKIP IMAGE INDEX STATE: vision model changed")
            return

        matrix = np.load(IMAGE_INDEX_MATRIX_PATH).astype("float32")

        image_product_ids = [
            safe_int(x)
            for x in meta.get("imageProductIds", [])
            if safe_int(x) > 0
        ]

        if len(image_product_ids) != len(matrix):
            print("SKIP IMAGE INDEX STATE: ids/matrix size mismatch")
            return

        product_meta: Dict[int, ProductCandidate] = {}

        raw_product_meta = meta.get("productMeta", {})

        if isinstance(raw_product_meta, dict):
            for _, raw in raw_product_meta.items():
                if not isinstance(raw, dict):
                    continue

                candidate = parse_candidate(raw)

                if candidate is not None and candidate.productId > 0:
                    product_meta[candidate.productId] = candidate

        _product_meta = product_meta
        _image_product_ids = image_product_ids
        _product_image_embeddings = matrix

        print(
            "LOAD IMAGE INDEX STATE:",
            "vectors =", len(_image_product_ids),
            "products =", len(set(_image_product_ids))
        )

    except Exception as e:
        print("LOAD IMAGE INDEX STATE ERROR:", repr(e))


def clear_persistent_image_cache() -> Dict[str, Any]:
    removed_files = 0

    try:
        if IMAGE_VECTOR_DIR.exists():
            for file in IMAGE_VECTOR_DIR.glob("*.npz"):
                try:
                    file.unlink()
                    removed_files += 1
                except Exception:
                    pass

        if IMAGE_INDEX_META_PATH.exists():
            IMAGE_INDEX_META_PATH.unlink()

        if IMAGE_INDEX_MATRIX_PATH.exists():
            IMAGE_INDEX_MATRIX_PATH.unlink()

    except Exception as e:
        print("CLEAR PERSISTENT IMAGE CACHE ERROR:", repr(e))

    return {
        "removedVectorFiles": removed_files,
        "metaDeleted": not IMAGE_INDEX_META_PATH.exists(),
        "matrixDeleted": not IMAGE_INDEX_MATRIX_PATH.exists()
    }


# =========================
# DOWNLOAD IMAGE
# =========================

async def download_image_async(
        client: httpx.AsyncClient,
        url: str,
        semaphore: asyncio.Semaphore
) -> Optional[Image.Image]:
    url = clean_url(url)

    if not url:
        return None

    if not (url.startswith("http://") or url.startswith("https://")):
        print("DOWNLOAD INVALID URL:", url)
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
    }

    async with semaphore:
        try:
            response = await client.get(
                url,
                timeout=DOWNLOAD_TIMEOUT,
                headers=headers,
                follow_redirects=True
            )

            print("DOWNLOAD IMAGE:", url, "status =", response.status_code)

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "image" not in content_type.lower():
                print("DOWNLOAD NOT IMAGE:", url, "content-type =", content_type)
                return None

            return Image.open(BytesIO(response.content)).convert("RGB")

        except Exception as e:
            print("DOWNLOAD IMAGE ERROR:", url, repr(e))
            return None


# =========================
# REASON BUILDERS
# =========================

def build_text_reason(semantic: float, sold_score: float, rating_score: float) -> str:
    if semantic >= 0.72:
        return "Rất phù hợp với nhu cầu của bạn"

    if semantic >= 0.55:
        return "Có nội dung tương tự sản phẩm bạn quan tâm"

    if rating_score >= 0.8:
        return "Sản phẩm có đánh giá tốt"

    if sold_score >= 0.7:
        return "Sản phẩm đang được mua nhiều"

    return "AI gợi ý dựa trên thông tin sản phẩm"


def build_image_reason(score: float) -> str:
    if score >= 0.50:
        return "Tìm thấy sản phẩm có hình ảnh rất giống ảnh bạn tải lên"

    if score >= IMAGE_MATCH_THRESHOLD:
        return "Tìm thấy sản phẩm có hình ảnh giống ảnh bạn tải lên"

    return "Sản phẩm có hình ảnh gần giống ảnh bạn tải lên"


# =========================
# INDEX BUILDING
# =========================

async def rebuild_index(candidates: List[ProductCandidate]) -> Dict[str, Any]:
    global _product_meta
    global _product_ids
    global _product_text_embeddings
    global _image_product_ids
    global _product_image_embeddings

    async with _index_lock:
        candidates = [
            c for c in candidates
            if c.productId > 0
        ]

        print("INDEX REBUILD START: candidates =", len(candidates))

        _product_meta = {c.productId: c for c in candidates}
        _product_ids = [c.productId for c in candidates]

        # ---------- TEXT INDEX ----------
        text_candidates = [
            c for c in candidates
            if c.text is not None and c.text.strip()
        ]

        if text_candidates:
            texts = [c.text.strip() for c in text_candidates]
            text_embeddings = await asyncio.to_thread(encode_texts_sync, texts)
            _product_ids = [c.productId for c in text_candidates]
            _product_text_embeddings = text_embeddings
        else:
            _product_ids = []
            _product_text_embeddings = None

        # ---------- IMAGE INDEX ----------
        image_jobs: List[Tuple[ProductCandidate, str]] = []

        for candidate in candidates:
            for url in get_candidate_image_urls(candidate):
                image_jobs.append((candidate, url))

        image_ids: List[int] = []
        image_vectors: List[np.ndarray] = []

        need_download: List[Tuple[ProductCandidate, str]] = []

        cache_hit = 0
        cache_miss = 0

        for candidate, url in image_jobs:
            # 1. RAM cache
            ram_cached = _image_embedding_cache.get(url)

            if ram_cached is not None:
                vectors = ram_cached if isinstance(ram_cached, list) else [ram_cached]

                for vector in vectors:
                    image_ids.append(candidate.productId)
                    image_vectors.append(l2_normalize(vector.astype("float32")))

                cache_hit += 1
                continue

            # 2. Disk cache
            disk_cached_vectors = load_image_vectors_from_disk(url)

            if disk_cached_vectors is not None:
                _image_embedding_cache[url] = disk_cached_vectors

                for vector in disk_cached_vectors:
                    image_ids.append(candidate.productId)
                    image_vectors.append(l2_normalize(vector.astype("float32")))

                cache_hit += 1
                continue

            # 3. Download nếu chưa có cache
            need_download.append((candidate, url))
            cache_miss += 1

        semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [
                download_image_async(client, url, semaphore)
                for _, url in need_download
            ]

            downloaded_images = await asyncio.gather(*tasks)

        download_ok = 0
        download_fail = 0

        for (candidate, url), image in zip(need_download, downloaded_images):
            if image is None:
                download_fail += 1
                continue

            try:
                vectors = await asyncio.to_thread(build_image_vectors_sync, image)

                if not vectors:
                    download_fail += 1
                    continue

                _image_embedding_cache[url] = vectors
                save_image_vectors_to_disk(url, vectors)

                for vector in vectors:
                    image_ids.append(candidate.productId)
                    image_vectors.append(l2_normalize(vector.astype("float32")))

                download_ok += 1

            except Exception as e:
                download_fail += 1
                print("INDEX IMAGE ENCODE ERROR:", candidate.productId, url, repr(e))

        _image_product_ids = image_ids

        if image_vectors:
            _product_image_embeddings = np.vstack(image_vectors).astype("float32")
        else:
            _product_image_embeddings = None

        save_image_index_state()

        print(
            "IMAGE EMBEDDING CACHE:",
            "jobs =", len(image_jobs),
            "hit =", cache_hit,
            "miss =", cache_miss,
            "downloadOk =", download_ok,
            "downloadFail =", download_fail,
            "diskDir =", str(IMAGE_VECTOR_DIR)
        )

        print(
            "INDEX REBUILD DONE:",
            "textProducts =", len(_product_ids),
            "imageVectors =", len(_image_product_ids),
            "imageProducts =", len(set(_image_product_ids))
        )

        return {
            "status": "ok",
            "totalProducts": len(candidates),
            "textProducts": len(_product_ids),
            "imageVectors": len(_image_product_ids),
            "imageProducts": len(set(_image_product_ids)),
            "cacheHit": cache_hit,
            "cacheMiss": cache_miss,
            "downloadOk": download_ok,
            "downloadFail": download_fail,
        }


# =========================
# ROUTES
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "smartcart-ai-persistent-image-embedding",
        "device": DEVICE,
        "textModel": TEXT_MODEL_NAME,
        "detectorModel": DETECT_MODEL_NAME,
        "visionModel": VISION_MODEL_NAME,
        "imageMatchThreshold": IMAGE_MATCH_THRESHOLD,
        "imageLowThreshold": IMAGE_LOW_THRESHOLD,
        "weakFallback": VISION_ALLOW_WEAK_FALLBACK,
        "indexedTextProducts": len(_product_ids),
        "indexedImageVectors": len(_image_product_ids),
        "indexedImageProducts": len(set(_image_product_ids)),
        "imageRamCacheSize": len(_image_embedding_cache),
        "imageEmbeddingDir": str(IMAGE_VECTOR_DIR),
        "imageIndexMetaExists": IMAGE_INDEX_META_PATH.exists(),
        "imageIndexMatrixExists": IMAGE_INDEX_MATRIX_PATH.exists(),
    }


@app.post("/index/products")
async def index_products(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    raw_candidates = []

    if isinstance(body, dict):
        raw_candidates = body.get("candidates") or body.get("products") or []
    elif isinstance(body, list):
        raw_candidates = body

    candidates = parse_candidates(raw_candidates)

    return await rebuild_index(candidates)


@app.post("/recommend/text", response_model=RecommendResponse)
async def recommend_text(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        body = {}

    seed_text = str(
        body.get("seedText")
        or body.get("keyword")
        or body.get("query")
        or "sản phẩm phổ biến bán chạy chất lượng tốt giá hợp lý"
    ).strip()

    page = safe_int(body.get("page"), 0)
    size = safe_int(body.get("size"), 10)

    raw_exclude = body.get("excludeProductIds") or body.get("excludeIds") or []
    exclude_ids = set()

    if isinstance(raw_exclude, list):
        exclude_ids = {safe_int(x) for x in raw_exclude if safe_int(x) > 0}

    raw_allowed = body.get("allowedProductIds") or body.get("allowedIds") or []
    allowed_ids = set()

    if isinstance(raw_allowed, list):
        allowed_ids = {safe_int(x) for x in raw_allowed if safe_int(x) > 0}

    if _product_text_embeddings is None or not _product_ids:
        print("RECOMMEND TEXT: empty index")
        return empty_response(page, size)

    query_embedding = await asyncio.to_thread(encode_texts_sync, [seed_text])
    query_embedding = query_embedding[0]

    semantic_scores = cosine_similarity_matrix(_product_text_embeddings, query_embedding)

    max_sold = max([c.soldCount for c in _product_meta.values()], default=0)
    max_review = max([c.reviewCount for c in _product_meta.values()], default=0)

    items: List[RecommendItem] = []

    for index, product_id in enumerate(_product_ids):
        if product_id in exclude_ids:
            continue

        if allowed_ids and product_id not in allowed_ids:
            continue

        candidate = _product_meta.get(product_id)

        if candidate is None:
            continue

        semantic = max(float(semantic_scores[index]), 0.0)
        sold_score = normalize_number(float(candidate.soldCount), float(max_sold))
        rating_score = normalize_number(float(candidate.rating), 5.0)
        review_score = normalize_number(float(candidate.reviewCount), float(max_review))

        final_score = (
            semantic * 0.72
            + rating_score * 0.12
            + sold_score * 0.10
            + review_score * 0.06
        )

        items.append(
            RecommendItem(
                productId=product_id,
                score=round(final_score * 100, 2),
                reason=build_text_reason(semantic, sold_score, rating_score)
            )
        )

    items.sort(key=lambda x: x.score, reverse=True)

    print("RECOMMEND TEXT RESULT:", len(items))

    return paginate(items, page, size)


@app.post("/search/image-base64", response_model=RecommendResponse)
async def search_image_base64(request: Request):
    page = 0
    size = 10

    try:
        raw_body = await request.body()
        print("REQUEST /search/image-base64 raw body length =", len(raw_body))

        if not raw_body:
            return empty_response(page, size)

        body = json.loads(raw_body.decode("utf-8"))

        if not isinstance(body, dict):
            return empty_response(page, size)

        page = safe_int(body.get("page"), 0)
        size = safe_int(body.get("size"), 10)

        image_base64 = body.get("imageBase64") or body.get("image_base64") or ""

        if not image_base64:
            print("SEARCH IMAGE: missing imageBase64")
            return empty_response(page, size)

        image_bytes = base64.b64decode(image_base64)
        query_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        return await process_visual_search(
            query_image=query_image,
            page=page,
            size=size
        )

    except Exception as e:
        print("FATAL /search/image-base64:", repr(e))
        return empty_response(page, size)


@app.post("/search/image", response_model=RecommendResponse)
async def search_image(
        file: Optional[UploadFile] = File(None),
        page: int = Form(0),
        size: int = Form(10)
):
    if file is None:
        print("ERROR /search/image: missing multipart field file")
        return empty_response(page, size)

    try:
        image_bytes = await file.read()
        query_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        return await process_visual_search(
            query_image=query_image,
            page=page,
            size=size
        )

    except Exception as e:
        print("FATAL /search/image:", repr(e))
        return empty_response(page, size)


async def process_visual_search(
        query_image: Image.Image,
        page: int,
        size: int
) -> RecommendResponse:
    if _product_image_embeddings is None or not _image_product_ids:
        print("VISUAL SEARCH: empty image index")
        return empty_response(page, size)

    try:
        query_views = await asyncio.to_thread(make_visual_views, query_image)

        query_vectors: List[np.ndarray] = []

        for view_image, view_name in query_views:
            vector = await asyncio.to_thread(encode_image_sync, view_image)
            query_vectors.append(l2_normalize(vector.astype("float32")))

        if not query_vectors:
            print("VISUAL SEARCH: no query vectors")
            return empty_response(page, size)

    except Exception as e:
        print("VISUAL SEARCH PREPARE ERROR:", repr(e))
        return empty_response(page, size)

    max_sold = max([c.soldCount for c in _product_meta.values()], default=0)

    best_by_product: Dict[int, Tuple[float, float]] = {}

    for query_vector in query_vectors:
        scores = cosine_similarity_matrix(_product_image_embeddings, query_vector)

        for index, product_id in enumerate(_image_product_ids):
            image_score = max(float(scores[index]), 0.0)

            candidate = _product_meta.get(product_id)

            if candidate is None:
                continue

            sold_score = normalize_number(
                float(candidate.soldCount),
                float(max_sold)
            )

            rating_score = normalize_number(float(candidate.rating), 5.0)

            final_score = (
                image_score * 0.92
                + rating_score * 0.05
                + sold_score * 0.03
            )

            old = best_by_product.get(product_id)

            if old is None or final_score > old[1]:
                best_by_product[product_id] = (image_score, final_score)

    scored_items = [
        (product_id, image_score, final_score)
        for product_id, (image_score, final_score) in best_by_product.items()
    ]

    scored_items.sort(key=lambda x: x[2], reverse=True)

    if not scored_items:
        print("VISUAL SEARCH: no scored items")
        return empty_response(page, size)

    best_score = scored_items[0][1]

    print(
        "VISUAL SEARCH TOP:",
        [
            {
                "productId": item[0],
                "imageScore": round(item[1], 4),
                "finalScore": round(item[2], 4),
            }
            for item in scored_items[:8]
        ]
    )

    matched = [
        item for item in scored_items
        if item[1] >= IMAGE_MATCH_THRESHOLD
    ]

    if not matched and VISION_ALLOW_WEAK_FALLBACK:
        if best_score >= IMAGE_LOW_THRESHOLD:
            matched = scored_items[:MAX_IMAGE_RESULTS]

    if not matched:
        print(
            "VISUAL SEARCH: no confident match",
            "bestScore =", round(best_score, 4),
            "threshold =", IMAGE_MATCH_THRESHOLD,
            "lowThreshold =", IMAGE_LOW_THRESHOLD,
            "indexedVectors =", len(_image_product_ids),
            "indexedProducts =", len(set(_image_product_ids))
        )
        return empty_response(page, size)

    results: List[RecommendItem] = []

    for product_id, image_score, final_score in matched[:MAX_IMAGE_RESULTS]:
        results.append(
            RecommendItem(
                productId=product_id,
                score=round(final_score * 100, 2),
                reason=build_image_reason(image_score)
            )
        )

    print(
        "VISUAL SEARCH RESULT:",
        "indexedVectors =", len(_image_product_ids),
        "indexedProducts =", len(set(_image_product_ids)),
        "matched =", len(results),
        "bestScore =", round(best_score, 4)
    )

    return paginate(results, page, size)


@app.get("/debug/image-index")
def debug_image_index():
    indexed_products = []

    for product_id in sorted(set(_image_product_ids)):
        meta = _product_meta.get(product_id)

        indexed_products.append({
            "productId": product_id,
            "text": meta.text[:150] if meta and meta.text else "",
            "imageUrl": meta.imageUrl if meta else "",
            "imageUrls": meta.imageUrls if meta else [],
            "vectorCount": _image_product_ids.count(product_id)
        })

    return {
        "indexedVectors": len(_image_product_ids),
        "indexedProducts": len(set(_image_product_ids)),
        "totalProducts": len(_product_meta),
        "products": indexed_products
    }


@app.get("/debug/image-cache")
def debug_image_cache():
    vector_files = list(IMAGE_VECTOR_DIR.glob("*.npz"))

    return {
        "ramCacheSize": len(_image_embedding_cache),
        "diskVectorFiles": len(vector_files),
        "indexedImageVectors": len(_image_product_ids),
        "indexedImageProducts": len(set(_image_product_ids)),
        "imageEmbeddingDir": str(IMAGE_VECTOR_DIR),
        "imageIndexMetaExists": IMAGE_INDEX_META_PATH.exists(),
        "imageIndexMatrixExists": IMAGE_INDEX_MATRIX_PATH.exists(),
    }


@app.post("/cache/clear")
def clear_cache():
    _image_embedding_cache.clear()

    persistent = clear_persistent_image_cache()

    return {
        "status": "ok",
        "message": "image cache cleared",
        "persistent": persistent
    }