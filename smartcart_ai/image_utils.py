import math
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


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


def _rgb_to_color_name(rgb: np.ndarray) -> str:
    r, g, b = [int(x) for x in rgb]

    # Đen / trắng / xám
    if r < 45 and g < 45 and b < 45:
        return "black"

    if r > 210 and g > 210 and b > 210:
        return "white"

    if abs(r - g) < 22 and abs(g - b) < 22 and 70 <= r <= 210:
        return "gray"

    # Các màu chính
    if r > 170 and g < 100 and b < 120:
        return "red"

    if r > 190 and g > 110 and b < 90:
        return "orange"

    if r > 180 and g > 160 and b < 110:
        return "yellow"

    if g > 130 and r < 130 and b < 140:
        return "green"

    if b > 140 and r < 130 and g < 150:
        return "blue"

    if r > 150 and b > 140 and g < 130:
        return "purple"

    if r > 190 and b > 150 and g < 170:
        return "pink"

    if r > 120 and g > 70 and b < 70:
        return "brown"

    return "mixed"


def extract_visual_attrs(image: Image.Image) -> Dict[str, object]:
    """
    Trích xuất thuộc tính thị giác đơn giản:
    - màu chủ đạo
    - có họa tiết hay không
    - dáng ảnh: dọc/ngang/vuông
    - tỷ lệ ảnh

    Dùng để tăng điểm cho váy/sản phẩm có màu, dáng, chi tiết giống nhau.
    """
    image = image.convert("RGB")

    width, height = image.size

    resized = image.resize((96, 96))
    arr = np.asarray(resized).astype("float32")

    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    std_rgb = arr.reshape(-1, 3).std(axis=0)

    color = _rgb_to_color_name(mean_rgb)

    gray = arr.mean(axis=2)

    dx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    dy = np.abs(gray[1:, :] - gray[:-1, :]).mean()

    edge_density = float((dx + dy) / 2.0)
    color_std = float(std_rgb.mean())

    patterned = edge_density > 18 or color_std > 54

    aspect_ratio = float(width / max(height, 1))

    if aspect_ratio < 0.58:
        shape = "long_vertical"
    elif aspect_ratio < 0.82:
        shape = "vertical"
    elif aspect_ratio <= 1.25:
        shape = "square"
    else:
        shape = "horizontal"

    return {
        "color": color,
        "patterned": bool(patterned),
        "edgeDensity": round(edge_density, 4),
        "colorStd": round(color_std, 4),
        "aspectRatio": round(aspect_ratio, 4),
        "shape": shape,
    }


def visual_attr_score(
        query_attrs: Dict[str, object],
        product_attrs: Dict[str, object],
        family: str
) -> float:
    """
    Cộng/trừ điểm dựa trên chi tiết ảnh:
    - màu giống
    - cùng có họa tiết hoặc cùng trơn
    - dáng ảnh giống
    - tỷ lệ ảnh gần nhau

    Với váy, pattern + dáng được cộng mạnh hơn.
    """
    if not query_attrs or not product_attrs:
        return 0.0

    score = 0.0

    query_color = query_attrs.get("color")
    product_color = product_attrs.get("color")

    if query_color and product_color:
        if query_color == product_color:
            score += 0.10
        elif "mixed" not in {query_color, product_color}:
            score -= 0.03

    query_patterned = bool(query_attrs.get("patterned"))
    product_patterned = bool(product_attrs.get("patterned"))

    if query_patterned == product_patterned:
        score += 0.06
    else:
        score -= 0.04

    query_shape = query_attrs.get("shape")
    product_shape = product_attrs.get("shape")

    if query_shape and product_shape:
        if query_shape == product_shape:
            score += 0.06

    try:
        q_ratio = float(query_attrs.get("aspectRatio") or 0)
        p_ratio = float(product_attrs.get("aspectRatio") or 0)

        if q_ratio > 0 and p_ratio > 0:
            diff = abs(math.log(q_ratio / p_ratio))

            if diff < 0.15:
                score += 0.08
            elif diff < 0.30:
                score += 0.04
            elif diff > 0.65:
                score -= 0.05

    except Exception:
        pass

    # Với váy: dáng + họa tiết rất quan trọng
    if family == "dress":
        if query_patterned == product_patterned:
            score += 0.08

        if query_shape == product_shape:
            score += 0.08

    return score