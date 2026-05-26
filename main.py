import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from io import BytesIO
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

IMAGE_MATCH_THRESHOLD = float(os.getenv("IMAGE_MATCH_THRESHOLD", "0.42"))
IMAGE_LOW_THRESHOLD = float(os.getenv("IMAGE_LOW_THRESHOLD", "0.30"))
MAX_IMAGE_RESULTS = int(os.getenv("MAX_IMAGE_RESULTS", "12"))
VISION_ALLOW_WEAK_FALLBACK = os.getenv("VISION_ALLOW_WEAK_FALLBACK", "false").lower() == "true"

DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT", "12"))
DOWNLOAD_CONCURRENCY = int(os.getenv("DOWNLOAD_CONCURRENCY", "8"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

_image_product_ids: List[int] = []
_product_image_embeddings: Optional[np.ndarray] = None

# URL ảnh -> vector ảnh, tự hết hạn sau 6 tiếng, tối đa 500 ảnh
_image_embedding_cache: TTLCache = TTLCache(maxsize=500, ttl=60 * 60 * 6)


# =========================
# DTO
# =========================

class ProductCandidate(BaseModel):
    productId: int
    text: str = ""
    imageUrl: Optional[str] = None
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
    print("AI Service ready.")
    yield
    print("AI Service stopped.")


app = FastAPI(
    title="SmartCart AI Service",
    lifespan=lifespan
)


# =========================
# UTILS
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


def parse_candidate(raw: Dict[str, Any]) -> Optional[ProductCandidate]:
    product_id = raw.get("productId") or raw.get("id")

    if product_id is None:
        return None

    return ProductCandidate(
        productId=safe_int(product_id),
        text=str(raw.get("text") or ""),
        imageUrl=raw.get("imageUrl"),
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


def clean_url(url: str) -> str:
    return str(url or "").strip().replace("\\", "")


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

    async with semaphore:
        try:
            response = await client.get(
                url,
                timeout=DOWNLOAD_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0 SmartCartAI/1.0"}
            )

            print("DOWNLOAD IMAGE:", url, "status =", response.status_code)

            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        except Exception as e:
            print("DOWNLOAD IMAGE ERROR:", url, repr(e))
            return None


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


def build_image_reason(score: float, detected_label: str) -> str:
    if detected_label and detected_label != "center-crop":
        if score >= 0.55:
            return f"AI nhận diện vùng {detected_label} và tìm thấy sản phẩm rất giống"
        return f"AI nhận diện vùng {detected_label} và tìm thấy sản phẩm gần giống"

    if score >= 0.55:
        return "Sản phẩm có hình ảnh rất giống ảnh bạn tải lên"

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

        image_candidates = [
            c for c in candidates
            if c.imageUrl is not None and clean_url(c.imageUrl)
        ]

        image_ids: List[int] = []
        image_vectors: List[np.ndarray] = []

        need_download: List[ProductCandidate] = []

        for candidate in image_candidates:
            url = clean_url(candidate.imageUrl or "")

            cached = _image_embedding_cache.get(url)

            if cached is not None:
                image_ids.append(candidate.productId)
                image_vectors.append(cached)
            else:
                need_download.append(candidate)

        semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [
                download_image_async(client, clean_url(c.imageUrl or ""), semaphore)
                for c in need_download
            ]

            downloaded_images = await asyncio.gather(*tasks)

        for candidate, image in zip(need_download, downloaded_images):
            if image is None:
                continue

            try:
                product_image, _, _ = await asyncio.to_thread(crop_main_product, image)
                vector = await asyncio.to_thread(encode_image_sync, product_image)

                url = clean_url(candidate.imageUrl or "")
                _image_embedding_cache[url] = vector

                image_ids.append(candidate.productId)
                image_vectors.append(vector)

            except Exception as e:
                print("INDEX IMAGE ENCODE ERROR:", candidate.productId, repr(e))

        _image_product_ids = image_ids

        if image_vectors:
            _product_image_embeddings = np.vstack(image_vectors).astype("float32")
        else:
            _product_image_embeddings = None

        print(
            "INDEX REBUILD DONE:",
            "textProducts =", len(_product_ids),
            "imageProducts =", len(_image_product_ids)
        )

        return {
            "status": "ok",
            "totalProducts": len(candidates),
            "textProducts": len(_product_ids),
            "imageProducts": len(_image_product_ids)
        }


# =========================
# ROUTES
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "smartcart-ai-async-cache-vector-index",
        "device": DEVICE,
        "textModel": TEXT_MODEL_NAME,
        "detectorModel": DETECT_MODEL_NAME,
        "visionModel": VISION_MODEL_NAME,
        "indexedTextProducts": len(_product_ids),
        "indexedImageProducts": len(_image_product_ids),
        "imageCacheSize": len(_image_embedding_cache),
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
        cropped_query, detected_label, detect_score = await asyncio.to_thread(
            crop_main_product,
            query_image
        )

        query_embedding = await asyncio.to_thread(
            encode_image_sync,
            cropped_query
        )

    except Exception as e:
        print("VISUAL SEARCH PREPARE ERROR:", repr(e))
        return empty_response(page, size)

    scores = cosine_similarity_matrix(_product_image_embeddings, query_embedding)

    scored_items = []

    for index, product_id in enumerate(_image_product_ids):
        image_score = max(float(scores[index]), 0.0)

        candidate = _product_meta.get(product_id)

        if candidate is None:
            continue

        sold_score = normalize_number(
            float(candidate.soldCount),
            max([c.soldCount for c in _product_meta.values()], default=0)
        )

        rating_score = normalize_number(float(candidate.rating), 5.0)

        final_score = (
            image_score * 0.90
            + rating_score * 0.07
            + sold_score * 0.03
        )

        scored_items.append((product_id, image_score, final_score))

    scored_items.sort(key=lambda x: x[2], reverse=True)

    matched = [
        item for item in scored_items
        if item[1] >= IMAGE_MATCH_THRESHOLD
    ]

    if not matched and VISION_ALLOW_WEAK_FALLBACK:
        matched = [
            item for item in scored_items
            if item[1] >= IMAGE_LOW_THRESHOLD
        ][:5]

    if not matched:
        print(
            "VISUAL SEARCH: no confident match",
            "bestScore =", round(scored_items[0][1], 4) if scored_items else 0,
            "detected =", detected_label,
            "detectScore =", round(detect_score, 4)
        )
        return empty_response(page, size)

    results: List[RecommendItem] = []

    for product_id, image_score, final_score in matched[:MAX_IMAGE_RESULTS]:
        results.append(
            RecommendItem(
                productId=product_id,
                score=round(final_score * 100, 2),
                reason=build_image_reason(image_score, detected_label)
            )
        )

    print(
        "VISUAL SEARCH RESULT:",
        "indexed =", len(_image_product_ids),
        "matched =", len(results),
        "detected =", detected_label,
        "detectScore =", round(detect_score, 4)
    )

    if not results:
        return empty_response(page, size)

    return paginate(results, page, size)


@app.post("/cache/clear")
def clear_cache():
    _image_embedding_cache.clear()

    return {
        "status": "ok",
        "message": "image cache cleared"
    }