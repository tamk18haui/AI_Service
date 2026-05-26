import os
from pathlib import Path

import torch


# =========================
# DEVICE
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# MODEL CONFIG
# =========================

TEXT_MODEL_NAME = os.getenv(
    "TEXT_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Giữ DINOv2 cho image-to-image
VISION_MODEL_NAME = os.getenv(
    "VISION_MODEL_NAME",
    "facebook/dinov2-base"
)

DETECT_MODEL_NAME = os.getenv(
    "DETECT_MODEL_NAME",
    "google/owlvit-base-patch32"
)


# =========================
# IMAGE SEARCH CONFIG
# =========================

IMAGE_MATCH_THRESHOLD = float(
    os.getenv("IMAGE_MATCH_THRESHOLD", "0.22")
)

IMAGE_LOW_THRESHOLD = float(
    os.getenv("IMAGE_LOW_THRESHOLD", "0.14")
)

MAX_IMAGE_RESULTS = int(
    os.getenv("MAX_IMAGE_RESULTS", "12")
)

VISION_ALLOW_WEAK_FALLBACK = os.getenv(
    "VISION_ALLOW_WEAK_FALLBACK",
    "true"
).lower() == "true"


# =========================
# DOWNLOAD CONFIG
# =========================

DOWNLOAD_TIMEOUT = float(
    os.getenv("DOWNLOAD_TIMEOUT", "12")
)

DOWNLOAD_CONCURRENCY = int(
    os.getenv("DOWNLOAD_CONCURRENCY", "8")
)

MAX_IMAGES_PER_PRODUCT = int(
    os.getenv("MAX_IMAGES_PER_PRODUCT", "8")
)


# =========================
# PERSISTENT VECTOR CACHE
# =========================

EMBEDDING_DATA_DIR = Path(
    os.getenv("EMBEDDING_DATA_DIR", "./smartcart_ai_data")
)

IMAGE_VECTOR_DIR = EMBEDDING_DATA_DIR / "image_vectors"
IMAGE_INDEX_META_PATH = EMBEDDING_DATA_DIR / "image_index_meta.json"
IMAGE_INDEX_MATRIX_PATH = EMBEDDING_DATA_DIR / "image_index_matrix.npy"

EMBEDDING_DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_VECTOR_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# OBJECT DETECTION LABELS
# =========================

DETECTION_LABELS = [
    # Watch / wearable
    "smartwatch",
    "smart watch",
    "wristwatch",
    "watch",
    "fitness tracker",

    # Phone
    "smartphone",
    "mobile phone",
    "cell phone",
    "phone",

    # Camera / webcam
    "webcam",
    "web camera",
    "camera",
    "security camera",

    # Laptop / audio
    "laptop",
    "headphones",
    "earphones",
    "electronic device",

    # Fashion - ưu tiên váy/đầm trước
    "dress",
    "skirt",
    "gown",
    "women dress",
    "shirt",
    "t-shirt",
    "blouse",
    "jacket",
    "hoodie",
    "pants",
    "jeans",
    "clothing",

    # Other
    "shoes",
    "sneakers",
    "bag",
    "backpack",
    "handbag",
    "cosmetic",
    "bottle",
    "toy",
    "book",
    "food package",
    "a product",
]


# =========================
# FAMILY-SPECIFIC THRESHOLD
# =========================

FAMILY_THRESHOLDS = {
    "dress": 0.16,
    "top_clothing": 0.17,
    "bottom_clothing": 0.17,
    "shoes": 0.18,
    "bag": 0.18,

    "smartwatch": 0.13,
    "phone": 0.15,
    "camera": 0.16,
    "laptop": 0.16,
    "headphones": 0.16,

    "beauty": 0.16,
    "book": 0.16,
}