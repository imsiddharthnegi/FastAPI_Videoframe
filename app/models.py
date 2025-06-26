from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class FrameInfo(BaseModel):
    """Information about an extracted frame"""
    frame_id: str
    timestamp: float
    path: str
    features: Optional[List[float]] = None

class VideoProcessResponse(BaseModel):
    """Response model for video processing"""
    message: str
    total_frames: int
    frames: List[FrameInfo]

class SimilarFrame(BaseModel):
    """Information about a similar frame"""
    frame_id: str
    timestamp: float
    path: str
    similarity_score: float
    features: List[float]

class SimilaritySearchResponse(BaseModel):
    """Response model for similarity search"""
    message: str
    query_features: List[float]
    similar_frames: List[SimilarFrame]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str

class FrameListResponse(BaseModel):
    """Response for listing frames"""
    frames: List[Dict[str, str]]
    total: int

