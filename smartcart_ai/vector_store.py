import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cachetools import TTLCache

from .config import (
    IMAGE_INDEX_MATRIX_PATH,
    IMAGE_INDEX_META_PATH,
    IMAGE_VECTOR_DIR,
    VISION_MODEL_NAME,
)
from .model_service import l2_normalize
from .schemas import ProductCandidate
from .text_utils import candidate_to_dict, clean_url, parse_candidate


# =========================
# GLOBAL VECTOR STORE
# =========================

product_meta: Dict[int, ProductCandidate] = {}

product_ids: List[int] = []
product_text_embeddings: Optional[np.ndarray] = None

image_product_ids: List[int] = []
product_image_embeddings: Optional[np.ndarray] = None

# Mỗi vector ảnh có info tương ứng:
# {
#   "productId": ...,
#   "url": ...,
#   "view": "full/center/detected-...",
#   "attrs": {...}
# }
image_vector_infos: List[Dict[str, Any]] = []

# RAM cache: imageUrl -> List[{"vector": np.ndarray, "view": str, "attrs": dict}]
image_embedding_cache: TTLCache = TTLCache(maxsize=3000, ttl=60 * 60 * 12)


# =========================
# CACHE PATH
# =========================

def image_cache_key(url: str) -> str:
    raw = f"{VISION_MODEL_NAME}|{clean_url(url)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def image_vector_cache_path(url: str) -> Tuple[Any, Any]:
    key = image_cache_key(url)

    npz_path = IMAGE_VECTOR_DIR / f"{key}.npz"
    json_path = IMAGE_VECTOR_DIR / f"{key}.json"

    return npz_path, json_path


# =========================
# PER-IMAGE VECTOR CACHE
# =========================

def load_image_vector_items_from_disk(url: str) -> Optional[List[Dict[str, Any]]]:
    npz_path, json_path = image_vector_cache_path(url)

    if not npz_path.exists():
        return None

    if not json_path.exists():
        return None

    try:
        data = np.load(npz_path)

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta_items = meta.get("items", [])

        if not isinstance(meta_items, list):
            return None

        items: List[Dict[str, Any]] = []

        for i, item_meta in enumerate(meta_items):
            key = f"v{i}"

            if key not in data:
                continue

            vector = data[key].astype("float32")
            vector = l2_normalize(vector)

            if not isinstance(item_meta, dict):
                item_meta = {}

            items.append({
                "vector": vector,
                "view": item_meta.get("view", ""),
                "attrs": item_meta.get("attrs", {}),
            })

        if not items:
            return None

        return items

    except Exception as e:
        print("LOAD IMAGE VECTOR CACHE ERROR:", url, repr(e))
        return None


def save_image_vector_items_to_disk(url: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        return

    npz_path, json_path = image_vector_cache_path(url)

    tmp_npz_path = npz_path.with_suffix(".tmp.npz")
    tmp_json_path = json_path.with_suffix(".tmp.json")

    try:
        arrays: Dict[str, np.ndarray] = {}
        meta_items: List[Dict[str, Any]] = []

        for i, item in enumerate(items):
            vector = item.get("vector")

            if vector is None:
                continue

            arrays[f"v{i}"] = l2_normalize(vector.astype("float32"))

            meta_items.append({
                "view": item.get("view", ""),
                "attrs": item.get("attrs", {}),
            })

        if not arrays:
            return

        np.savez_compressed(tmp_npz_path, **arrays)

        with open(tmp_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "visionModel": VISION_MODEL_NAME,
                    "url": url,
                    "items": meta_items,
                },
                f,
                ensure_ascii=False
            )

        os.replace(tmp_npz_path, npz_path)
        os.replace(tmp_json_path, json_path)

    except Exception as e:
        print("SAVE IMAGE VECTOR CACHE ERROR:", url, repr(e))

        for path in [tmp_npz_path, tmp_json_path]:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass


# =========================
# WHOLE IMAGE INDEX STATE
# =========================

def save_image_index_state() -> None:
    global product_image_embeddings
    global image_product_ids
    global image_vector_infos
    global product_meta

    try:
        if product_image_embeddings is None:
            print("SAVE IMAGE INDEX STATE SKIP: empty matrix")
            return

        if not image_product_ids:
            print("SAVE IMAGE INDEX STATE SKIP: empty ids")
            return

        if len(image_product_ids) != len(product_image_embeddings):
            print(
                "SAVE IMAGE INDEX STATE SKIP: ids/matrix mismatch",
                "ids =", len(image_product_ids),
                "matrix =", len(product_image_embeddings)
            )
            return

        if len(image_vector_infos) != len(image_product_ids):
            print(
                "SAVE IMAGE INDEX STATE WARNING: infos/ids mismatch",
                "infos =", len(image_vector_infos),
                "ids =", len(image_product_ids)
            )

            image_vector_infos = [
                image_vector_infos[i] if i < len(image_vector_infos) else {}
                for i in range(len(image_product_ids))
            ]

        np.save(
            IMAGE_INDEX_MATRIX_PATH,
            product_image_embeddings.astype("float32")
        )

        meta = {
            "visionModel": VISION_MODEL_NAME,
            "imageProductIds": image_product_ids,
            "imageVectorInfos": image_vector_infos,
            "productMeta": {
                str(product_id): candidate_to_dict(candidate)
                for product_id, candidate in product_meta.items()
            }
        }

        with open(IMAGE_INDEX_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        print(
            "SAVE IMAGE INDEX STATE:",
            "vectors =", len(image_product_ids),
            "products =", len(set(image_product_ids)),
            "matrixPath =", str(IMAGE_INDEX_MATRIX_PATH),
            "metaPath =", str(IMAGE_INDEX_META_PATH)
        )

    except Exception as e:
        print("SAVE IMAGE INDEX STATE ERROR:", repr(e))


def load_image_index_state() -> None:
    global product_meta
    global image_product_ids
    global product_image_embeddings
    global image_vector_infos

    try:
        if not IMAGE_INDEX_META_PATH.exists():
            print("LOAD IMAGE INDEX STATE SKIP: meta not exists")
            return

        if not IMAGE_INDEX_MATRIX_PATH.exists():
            print("LOAD IMAGE INDEX STATE SKIP: matrix not exists")
            return

        with open(IMAGE_INDEX_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("visionModel") != VISION_MODEL_NAME:
            print(
                "SKIP IMAGE INDEX STATE: vision model changed",
                "old =", meta.get("visionModel"),
                "current =", VISION_MODEL_NAME
            )
            return

        matrix = np.load(IMAGE_INDEX_MATRIX_PATH).astype("float32")

        ids: List[int] = []

        raw_ids = meta.get("imageProductIds", [])

        if isinstance(raw_ids, list):
            for value in raw_ids:
                try:
                    product_id = int(value)

                    if product_id > 0:
                        ids.append(product_id)
                except Exception:
                    pass

        if len(ids) != len(matrix):
            print(
                "SKIP IMAGE INDEX STATE: ids/matrix size mismatch",
                "ids =", len(ids),
                "matrix =", len(matrix)
            )
            return

        raw_infos = meta.get("imageVectorInfos", [])

        if not isinstance(raw_infos, list):
            raw_infos = []

        if len(raw_infos) != len(ids):
            print(
                "LOAD IMAGE INDEX STATE WARNING: infos/ids mismatch",
                "infos =", len(raw_infos),
                "ids =", len(ids)
            )

            raw_infos = [
                raw_infos[i] if i < len(raw_infos) and isinstance(raw_infos[i], dict) else {}
                for i in range(len(ids))
            ]

        restored_meta: Dict[int, ProductCandidate] = {}

        raw_product_meta = meta.get("productMeta", {})

        if isinstance(raw_product_meta, dict):
            for _, raw_candidate in raw_product_meta.items():
                if not isinstance(raw_candidate, dict):
                    continue

                candidate = parse_candidate(raw_candidate)

                if candidate is not None and candidate.productId > 0:
                    restored_meta[candidate.productId] = candidate

        product_meta = restored_meta
        image_product_ids = ids
        product_image_embeddings = matrix
        image_vector_infos = raw_infos

        print(
            "LOAD IMAGE INDEX STATE:",
            "vectors =", len(image_product_ids),
            "products =", len(set(image_product_ids)),
            "productMeta =", len(product_meta)
        )

    except Exception as e:
        print("LOAD IMAGE INDEX STATE ERROR:", repr(e))


# =========================
# CLEAR CACHE
# =========================

def clear_persistent_image_cache() -> Dict[str, Any]:
    removed_files = 0

    try:
        if IMAGE_VECTOR_DIR.exists():
            for pattern in ["*.npz", "*.json"]:
                for file in IMAGE_VECTOR_DIR.glob(pattern):
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
        "removedFiles": removed_files,
        "metaDeleted": not IMAGE_INDEX_META_PATH.exists(),
        "matrixDeleted": not IMAGE_INDEX_MATRIX_PATH.exists()
    }