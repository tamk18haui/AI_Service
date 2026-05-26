from typing import List, Optional

from pydantic import BaseModel


class ProductCandidate(BaseModel):
    productId: int
    text: str = ""
    imageUrl: Optional[str] = None
    imageUrls: List[str] = []

    categoryId: Optional[int] = None
    categoryName: str = ""
    brand: str = ""
    productName: str = ""

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