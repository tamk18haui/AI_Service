import base64
import json
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from PIL import Image

from smartcart_ai import vector_store as store
from smartcart_ai.config import (
    DEVICE,
    IMAGE_LOW_THRESHOLD,
    IMAGE_MATCH_THRESHOLD,
    TEXT_MODEL_NAME,
    VISION_ALLOW_WEAK_FALLBACK,
    VISION_MODEL_NAME,
    DETECT_MODEL_NAME,
)
from smartcart_ai.model_service import load_all_models
from smartcart_ai.schemas import RecommendResponse
from smartcart_ai.search_service import (
    empty_response,
    process_visual_search,
    rebuild_index,
    recommend_text,
)
from smartcart_ai.text_utils import parse_candidates


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI Service starting...")
    print("RUNNING AI FILE =", os.path.abspath(__file__))
    print("VISION_MODEL_NAME =", VISION_MODEL_NAME)
    print("IMAGE_MATCH_THRESHOLD =", IMAGE_MATCH_THRESHOLD)
    print("IMAGE_LOW_THRESHOLD =", IMAGE_LOW_THRESHOLD)
    print("VISION_ALLOW_WEAK_FALLBACK =", VISION_ALLOW_WEAK_FALLBACK)

    load_all_models()
    store.load_image_index_state()

    print("AI Service ready.")
    yield
    print("AI Service stopped.")


app = FastAPI(
    title="SmartCart AI Service",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "smartcart-ai-dinov2-family-style-v3",
        "device": DEVICE,
        "textModel": TEXT_MODEL_NAME,
        "detectorModel": DETECT_MODEL_NAME,
        "visionModel": VISION_MODEL_NAME,
        "imageMatchThreshold": IMAGE_MATCH_THRESHOLD,
        "imageLowThreshold": IMAGE_LOW_THRESHOLD,
        "weakFallback": VISION_ALLOW_WEAK_FALLBACK,
        "indexedTextProducts": len(store.product_ids),
        "indexedImageVectors": len(store.image_product_ids),
        "indexedImageProducts": len(set(store.image_product_ids)),
        "imageRamCacheSize": len(store.image_embedding_cache),
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
async def recommend_text_route(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        body = {}

    return await recommend_text(body)


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

        page = int(body.get("page") or 0)
        size = int(body.get("size") or 10)

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


@app.get("/debug/image-index")
def debug_image_index():
    products = []

    for product_id in sorted(set(store.image_product_ids)):
        meta = store.product_meta.get(product_id)

        if meta is None:
            continue

        from smartcart_ai.text_utils import detect_family_from_candidate

        products.append({
            "productId": product_id,
            "text": meta.text[:180],
            "imageUrl": meta.imageUrl,
            "imageUrls": meta.imageUrls,
            "categoryId": meta.categoryId,
            "categoryName": meta.categoryName,
            "brand": meta.brand,
            "productName": meta.productName,
            "family": detect_family_from_candidate(meta),
            "vectorCount": store.image_product_ids.count(product_id),
        })

    return {
        "indexedVectors": len(store.image_product_ids),
        "indexedProducts": len(set(store.image_product_ids)),
        "totalProducts": len(store.product_meta),
        "products": products,
    }


@app.get("/debug/image-cache")
def debug_image_cache():
    return {
        "ramCacheSize": len(store.image_embedding_cache),
        "indexedImageVectors": len(store.image_product_ids),
        "indexedImageProducts": len(set(store.image_product_ids)),
        "imageIndexMetaExists": store.IMAGE_INDEX_META_PATH.exists() if hasattr(store, "IMAGE_INDEX_META_PATH") else None,
    }


@app.post("/cache/clear")
def clear_cache():
    store.image_embedding_cache.clear()
    persistent = store.clear_persistent_image_cache()

    store.product_meta = {}
    store.product_ids = []
    store.product_text_embeddings = None
    store.image_product_ids = []
    store.product_image_embeddings = None
    store.image_vector_infos = []

    return {
        "status": "ok",
        "message": "image cache cleared",
        "persistent": persistent
    }