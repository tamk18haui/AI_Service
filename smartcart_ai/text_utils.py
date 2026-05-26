import json
import unicodedata
from typing import Any, Dict, List, Optional

from .schemas import ProductCandidate


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


def normalize_vi_text(value: Any) -> str:
    value = str(value or "").lower().strip()

    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")

    chars = []

    for ch in value:
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")

    return " ".join("".join(chars).split())


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

        categoryId=safe_int(raw.get("categoryId"), 0) or None,
        categoryName=str(raw.get("categoryName") or ""),
        brand=str(raw.get("brand") or ""),
        productName=str(raw.get("productName") or raw.get("name") or ""),

        soldCount=safe_int(raw.get("soldCount"), 0),
        rating=safe_float(raw.get("rating"), 0.0),
        reviewCount=safe_int(raw.get("reviewCount"), 0),
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


def candidate_text(candidate: ProductCandidate) -> str:
    if candidate is None:
        return ""

    return normalize_vi_text(
        " ".join([
            candidate.productName or "",
            candidate.categoryName or "",
            candidate.brand or "",
            candidate.text or "",
        ])
    )


def detect_family_from_label(label: str) -> str:
    label = normalize_vi_text(label)

    # Đồng hồ / wearable
    if any(x in label for x in [
        "smartwatch",
        "smart watch",
        "wristwatch",
        "fitness tracker",
        "watch",
    ]):
        return "smartwatch"

    # Điện thoại
    if any(x in label for x in [
        "smartphone",
        "mobile phone",
        "cell phone",
        "phone",
    ]):
        return "phone"

    # Camera / webcam
    if any(x in label for x in [
        "webcam",
        "web camera",
        "camera",
        "security camera",
    ]):
        return "camera"

    if "laptop" in label:
        return "laptop"

    if any(x in label for x in [
        "headphones",
        "earphones",
    ]):
        return "headphones"

    # Váy/đầm phải ưu tiên trước clothing chung
    if any(x in label for x in [
        "dress",
        "gown",
        "women dress",
    ]):
        return "dress"

    if any(x in label for x in [
        "skirt",
    ]):
        return "dress"

    if any(x in label for x in [
        "shirt",
        "t shirt",
        "t-shirt",
        "blouse",
        "jacket",
        "hoodie",
    ]):
        return "top_clothing"

    if any(x in label for x in [
        "pants",
        "jeans",
    ]):
        return "bottom_clothing"

    if any(x in label for x in [
        "shoes",
        "sneakers",
    ]):
        return "shoes"

    if any(x in label for x in [
        "bag",
        "backpack",
        "handbag",
    ]):
        return "bag"

    if any(x in label for x in [
        "cosmetic",
        "bottle",
    ]):
        return "beauty"

    if "book" in label:
        return "book"

    return ""


def detect_family_from_candidate(candidate: ProductCandidate) -> str:
    text = candidate_text(candidate)

    # Smartwatch phải đặt trước phone/electronics
    if any(x in text for x in [
        "smartwatch",
        "smart watch",
        "dong ho thong minh",
        "dong ho dien tu",
        "apple watch",
        "galaxy watch",
        "watch",
        "fitness tracker",
        "vong deo thong minh",
    ]):
        return "smartwatch"

    # Camera / webcam
    if any(x in text for x in [
        "webcam",
        "web cam",
        "camera",
        "may anh",
        "camera an ninh",
        "security camera",
    ]):
        return "camera"

    # Phone
    if any(x in text for x in [
        "dien thoai",
        "smartphone",
        "iphone",
        "samsung",
        "oppo",
        "xiaomi",
        "realme",
        "vivo",
        "nokia",
        "mobile phone",
        "cell phone",
    ]):
        return "phone"

    if any(x in text for x in [
        "laptop",
        "may tinh xach tay",
        "notebook",
        "macbook",
    ]):
        return "laptop"

    if any(x in text for x in [
        "tai nghe",
        "headphone",
        "headphones",
        "earphone",
        "earphones",
        "airpods",
    ]):
        return "headphones"

    # Váy/đầm phải đặt trước áo/quần
    if any(x in text for x in [
        "vay",
        "dam",
        "chan vay",
        "dress",
        "skirt",
        "gown",
        "women dress",
    ]):
        return "dress"

    if any(x in text for x in [
        "ao",
        "shirt",
        "t shirt",
        "t-shirt",
        "blouse",
        "hoodie",
        "jacket",
    ]):
        return "top_clothing"

    if any(x in text for x in [
        "quan",
        "pants",
        "jeans",
        "jean",
        "kaki",
    ]):
        return "bottom_clothing"

    if any(x in text for x in [
        "giay",
        "sneaker",
        "shoes",
        "dep",
    ]):
        return "shoes"

    if any(x in text for x in [
        "tui",
        "balo",
        "bag",
        "backpack",
        "handbag",
    ]):
        return "bag"

    if any(x in text for x in [
        "my pham",
        "son",
        "kem",
        "nuoc hoa",
        "cosmetic",
        "skincare",
    ]):
        return "beauty"

    if any(x in text for x in [
        "sach",
        "book",
    ]):
        return "book"

    return ""


def group_from_family(family: str) -> str:
    if family in {
        "smartwatch",
        "phone",
        "camera",
        "laptop",
        "headphones",
    }:
        return "electronics"

    if family in {
        "dress",
        "top_clothing",
        "bottom_clothing",
    }:
        return "fashion"

    if family == "shoes":
        return "shoes"

    if family == "bag":
        return "bag"

    if family == "beauty":
        return "beauty"

    if family == "book":
        return "book"

    return ""


def family_match_score(query_family: str, candidate_family: str) -> float:
    if not query_family:
        return 0.0

    if candidate_family == query_family:
        return 0.55

    if not candidate_family:
        return -0.35

    # Sai family thì phạt rất mạnh.
    # Ví dụ smartwatch không được ra webcam, váy không được ra áo.
    return -1.50


def get_candidate_image_urls(
        candidate: ProductCandidate,
        max_images: int
) -> List[str]:
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

        if len(result) >= max_images:
            break

    return result


def candidate_to_dict(candidate: ProductCandidate) -> Dict[str, Any]:
    return {
        "productId": candidate.productId,
        "text": candidate.text,
        "imageUrl": candidate.imageUrl,
        "imageUrls": candidate.imageUrls,
        "categoryId": candidate.categoryId,
        "categoryName": candidate.categoryName,
        "brand": candidate.brand,
        "productName": candidate.productName,
        "soldCount": candidate.soldCount,
        "rating": candidate.rating,
        "reviewCount": candidate.reviewCount,
    }