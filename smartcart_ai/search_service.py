import asyncio
from io import BytesIO
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np
from PIL import Image

from . import vector_store as store
from .config import (
    DOWNLOAD_CONCURRENCY,
    DOWNLOAD_TIMEOUT,
    FAMILY_THRESHOLDS,
    IMAGE_LOW_THRESHOLD,
    IMAGE_MATCH_THRESHOLD,
    MAX_IMAGE_RESULTS,
    MAX_IMAGES_PER_PRODUCT,
)
from .image_utils import visual_attr_score
from .model_service import (
    build_image_vector_items_sync,
    detect_query_family_by_owl,
    encode_image_sync,
    encode_texts_sync,
    l2_normalize,
    make_visual_views,
)
from .schemas import ProductCandidate, RecommendItem, RecommendResponse
from .text_utils import (
    clean_url,
    detect_family_from_candidate,
    detect_family_from_label,
    family_match_score,
    get_candidate_image_urls,
    group_from_family,
    normalize_number,
)


_index_lock = asyncio.Lock()


# =========================
# RESPONSE HELPERS
# =========================

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


def cosine_similarity_matrix(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    if matrix is None or len(matrix) == 0:
        return np.array([], dtype=np.float32)

    query = l2_normalize(query.astype("float32"))
    return np.dot(matrix, query)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


# =========================
# IMAGE DOWNLOAD
# =========================

async def download_image_async(
        client: httpx.AsyncClient,
        url: str,
        semaphore: asyncio.Semaphore
):
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
# INDEX BUILDING
# =========================

async def rebuild_index(candidates: List[ProductCandidate]) -> Dict[str, Any]:
    async with _index_lock:
        candidates = [
            c for c in candidates
            if c is not None and c.productId > 0
        ]

        print("INDEX REBUILD START: candidates =", len(candidates))

        store.product_meta = {c.productId: c for c in candidates}

        # ---------- TEXT INDEX ----------
        text_candidates = [
            c for c in candidates
            if c.text is not None and c.text.strip()
        ]

        if text_candidates:
            texts = [c.text.strip() for c in text_candidates]
            store.product_text_embeddings = await asyncio.to_thread(encode_texts_sync, texts)
            store.product_ids = [c.productId for c in text_candidates]
        else:
            store.product_ids = []
            store.product_text_embeddings = None

        # ---------- IMAGE INDEX ----------
        image_jobs: List[Tuple[ProductCandidate, str]] = []

        for candidate in candidates:
            for url in get_candidate_image_urls(candidate, MAX_IMAGES_PER_PRODUCT):
                image_jobs.append((candidate, url))

        image_ids: List[int] = []
        image_vectors: List[np.ndarray] = []
        image_infos: List[Dict[str, Any]] = []

        need_download: List[Tuple[ProductCandidate, str]] = []

        cache_hit = 0
        cache_miss = 0

        for candidate, url in image_jobs:
            ram_cached = store.image_embedding_cache.get(url)

            if ram_cached is not None:
                items = ram_cached
                cache_hit += 1
            else:
                items = store.load_image_vector_items_from_disk(url)

                if items is not None:
                    store.image_embedding_cache[url] = items
                    cache_hit += 1
                else:
                    need_download.append((candidate, url))
                    cache_miss += 1
                    continue

            for item in items:
                vector = item.get("vector")

                if vector is None:
                    continue

                image_ids.append(candidate.productId)
                image_vectors.append(l2_normalize(vector.astype("float32")))
                image_infos.append({
                    "productId": candidate.productId,
                    "url": url,
                    "view": item.get("view", ""),
                    "attrs": item.get("attrs", {}),
                })

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
                items = await asyncio.to_thread(build_image_vector_items_sync, image)

                if not items:
                    download_fail += 1
                    continue

                store.image_embedding_cache[url] = items
                store.save_image_vector_items_to_disk(url, items)

                for item in items:
                    vector = item.get("vector")

                    if vector is None:
                        continue

                    image_ids.append(candidate.productId)
                    image_vectors.append(l2_normalize(vector.astype("float32")))
                    image_infos.append({
                        "productId": candidate.productId,
                        "url": url,
                        "view": item.get("view", ""),
                        "attrs": item.get("attrs", {}),
                    })

                download_ok += 1

            except Exception as e:
                download_fail += 1
                print("INDEX IMAGE ENCODE ERROR:", candidate.productId, url, repr(e))

        store.image_product_ids = image_ids
        store.image_vector_infos = image_infos

        if image_vectors:
            store.product_image_embeddings = np.vstack(image_vectors).astype("float32")
        else:
            store.product_image_embeddings = None

        store.save_image_index_state()

        family_count: Dict[str, int] = {}

        for candidate in candidates:
            family = detect_family_from_candidate(candidate) or "unknown"
            family_count[family] = family_count.get(family, 0) + 1

        print("INDEX FAMILY COUNT:", family_count)

        print(
            "IMAGE EMBEDDING CACHE:",
            "jobs =", len(image_jobs),
            "hit =", cache_hit,
            "miss =", cache_miss,
            "downloadOk =", download_ok,
            "downloadFail =", download_fail,
        )

        print(
            "INDEX REBUILD DONE:",
            "textProducts =", len(store.product_ids),
            "imageVectors =", len(store.image_product_ids),
            "imageProducts =", len(set(store.image_product_ids))
        )

        return {
            "status": "ok",
            "totalProducts": len(candidates),
            "textProducts": len(store.product_ids),
            "imageVectors": len(store.image_product_ids),
            "imageProducts": len(set(store.image_product_ids)),
            "cacheHit": cache_hit,
            "cacheMiss": cache_miss,
            "downloadOk": download_ok,
            "downloadFail": download_fail,
            "familyCount": family_count,
        }


# =========================
# TEXT RECOMMEND
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


async def recommend_text(body: Dict[str, Any]) -> RecommendResponse:
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
        exclude_ids = {
            safe_int(x)
            for x in raw_exclude
            if safe_int(x) > 0
        }

    raw_allowed = body.get("allowedProductIds") or body.get("allowedIds") or []
    allowed_ids = set()

    if isinstance(raw_allowed, list):
        allowed_ids = {
            safe_int(x)
            for x in raw_allowed
            if safe_int(x) > 0
        }

    if store.product_text_embeddings is None or not store.product_ids:
        print("RECOMMEND TEXT: empty index")
        return empty_response(page, size)

    query_embedding = await asyncio.to_thread(encode_texts_sync, [seed_text])
    query_embedding = query_embedding[0]

    semantic_scores = cosine_similarity_matrix(
        store.product_text_embeddings,
        query_embedding
    )

    max_sold = max([c.soldCount for c in store.product_meta.values()], default=0)
    max_review = max([c.reviewCount for c in store.product_meta.values()], default=0)

    items: List[RecommendItem] = []

    for index, product_id in enumerate(store.product_ids):
        if product_id in exclude_ids:
            continue

        if allowed_ids and product_id not in allowed_ids:
            continue

        candidate = store.product_meta.get(product_id)

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


# =========================
# IMAGE SEARCH
# =========================

def build_image_reason(score: float, query_family: str) -> str:
    if query_family == "dress":
        if score >= 0.45:
            return "Tìm thấy váy có kiểu dáng và chi tiết rất giống ảnh bạn tải lên"
        if score >= IMAGE_MATCH_THRESHOLD:
            return "Tìm thấy váy có kiểu dáng giống ảnh bạn tải lên"
        return "Tìm thấy váy cùng loại, có màu/dáng/chi tiết gần giống ảnh bạn tải lên"

    if query_family:
        if score >= 0.45:
            return "Tìm thấy sản phẩm đúng loại và có hình ảnh rất giống ảnh bạn tải lên"
        if score >= IMAGE_MATCH_THRESHOLD:
            return "Tìm thấy sản phẩm đúng loại và có hình ảnh giống ảnh bạn tải lên"
        return "Tìm thấy sản phẩm đúng loại, có hình ảnh gần giống ảnh bạn tải lên"

    return "Sản phẩm có hình ảnh gần giống ảnh bạn tải lên"


async def process_visual_search(
        query_image: Image.Image,
        page: int,
        size: int
) -> RecommendResponse:
    if store.product_image_embeddings is None or not store.image_product_ids:
        print("VISUAL SEARCH: empty image index")
        return empty_response(page, size)

    try:
        query_group, query_family, family_label, family_score = await asyncio.to_thread(
            detect_query_family_by_owl,
            query_image
        )

        query_views = await asyncio.to_thread(make_visual_views, query_image)

        query_items = []

        for view_image, view_name, attrs in query_views:
            vector = await asyncio.to_thread(encode_image_sync, view_image)

            query_items.append({
                "vector": l2_normalize(vector.astype("float32")),
                "view": view_name,
                "attrs": attrs,
            })

        if not query_family:
            query_family = detect_family_from_label(family_label)

        if not query_group:
            query_group = group_from_family(query_family)

        if not query_items:
            print("VISUAL SEARCH: no query vectors")
            return empty_response(page, size)

    except Exception as e:
        print("VISUAL SEARCH PREPARE ERROR:", repr(e))
        return empty_response(page, size)

    best_by_product: Dict[int, Tuple[float, float, str, Dict[str, Any]]] = {}

    for query_item in query_items:
        query_vector = query_item["vector"]
        query_attrs = query_item.get("attrs", {})

        scores = cosine_similarity_matrix(
            store.product_image_embeddings,
            query_vector
        )

        for index, product_id in enumerate(store.image_product_ids):
            image_score = max(float(scores[index]), 0.0)

            candidate = store.product_meta.get(product_id)

            if candidate is None:
                continue

            candidate_family = detect_family_from_candidate(candidate)
            candidate_group = group_from_family(candidate_family)

            vector_info = store.image_vector_infos[index] if index < len(store.image_vector_infos) else {}
            product_attrs = vector_info.get("attrs", {})

            family_bonus = family_match_score(query_family, candidate_family)
            attr_bonus = visual_attr_score(query_attrs, product_attrs, query_family)

            final_score = image_score + family_bonus + attr_bonus

            old = best_by_product.get(product_id)

            if old is None or final_score > old[1]:
                best_by_product[product_id] = (
                    image_score,
                    final_score,
                    candidate_family,
                    {
                        "candidateGroup": candidate_group,
                        "queryAttrs": query_attrs,
                        "productAttrs": product_attrs,
                    }
                )

    scored_items = [
        (product_id, image_score, final_score, candidate_family, detail)
        for product_id, (image_score, final_score, candidate_family, detail)
        in best_by_product.items()
    ]

    scored_items.sort(key=lambda x: x[2], reverse=True)

    if not scored_items:
        print("VISUAL SEARCH: no scored items")
        return empty_response(page, size)

    print(
        "VISUAL SEARCH TOP RAW:",
        [
            {
                "productId": item[0],
                "imageScore": round(item[1], 4),
                "finalScore": round(item[2], 4),
                "candidateFamily": item[3],
                "name": store.product_meta[item[0]].productName if item[0] in store.product_meta else "",
                "category": store.product_meta[item[0]].categoryName if item[0] in store.product_meta else "",
                "productAttrs": item[4].get("productAttrs", {}),
            }
            for item in scored_items[:15]
        ]
    )

    # Lọc cứng theo family.
    # Ví dụ:
    # - gửi váy => chỉ trả dress
    # - gửi smartwatch => chỉ trả smartwatch
    # - gửi webcam => chỉ trả camera
    if query_family:
        same_family_items = [
            item for item in scored_items
            if item[3] == query_family
        ]

        if same_family_items:
            scored_items = same_family_items
        else:
            print(
                "VISUAL SEARCH STRICT FAMILY EMPTY:",
                "queryFamily =", query_family,
                "label =", family_label,
                "score =", round(family_score, 4),
            )
            return empty_response(page, size)

    best_image_score = scored_items[0][1]

    threshold = FAMILY_THRESHOLDS.get(
        query_family,
        IMAGE_MATCH_THRESHOLD
    )

    matched = [
        item for item in scored_items
        if item[1] >= threshold
    ]

    if not matched and best_image_score >= IMAGE_LOW_THRESHOLD:
        matched = scored_items[:MAX_IMAGE_RESULTS]

    if not matched:
        print(
            "VISUAL SEARCH: no confident same-family match",
            "queryGroup =", query_group,
            "queryFamily =", query_family,
            "label =", family_label,
            "familyScore =", round(family_score, 4),
            "bestImageScore =", round(best_image_score, 4),
            "threshold =", threshold,
            "indexedVectors =", len(store.image_product_ids),
            "indexedProducts =", len(set(store.image_product_ids)),
        )
        return empty_response(page, size)

    results: List[RecommendItem] = []

    for product_id, image_score, final_score, candidate_family, detail in matched[:MAX_IMAGE_RESULTS]:
        results.append(
            RecommendItem(
                productId=product_id,
                score=round(max(final_score, 0.0) * 100, 2),
                reason=build_image_reason(image_score, query_family)
            )
        )

    print(
        "VISUAL SEARCH RESULT:",
        "queryGroup =", query_group,
        "queryFamily =", query_family,
        "label =", family_label,
        "familyScore =", round(family_score, 4),
        "matched =", len(results),
        "bestImageScore =", round(best_image_score, 4),
    )

    return paginate(results, page, size)