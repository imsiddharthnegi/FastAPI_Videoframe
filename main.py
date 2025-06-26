from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List, Optional
import json

from app.video_processor import VideoProcessor
from app.feature_extractor import FeatureExtractor
from app.vector_db import VectorDatabase
from app.models import VideoProcessResponse, SimilaritySearchResponse

# Initialize FastAPI app
app = FastAPI(
    title="Video Frame Analyzer",
    description="A FastAPI application for video processing, frame extraction, and similarity search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
video_processor = VideoProcessor()
feature_extractor = FeatureExtractor()
vector_db = VectorDatabase()

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("frames", exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Video Frame Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "upload_video": "/upload-video",
            "search_similar": "/search-similar",
            "get_frame": "/frame/{frame_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Video Frame Analyzer is running"}

@app.post("/upload-video", response_model=VideoProcessResponse)
async def upload_video(
    file: UploadFile = File(...),
    interval: float = Query(1.0, description="Frame extraction interval in seconds")
):
    """
    Upload a video file and extract frames at specified intervals
    """
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process video and extract frames
        frames_info = video_processor.extract_frames(file_path, interval)
        
        # Extract features for each frame
        feature_vectors = []
        for frame_info in frames_info:
            features = feature_extractor.extract_features(frame_info['path'])
            frame_info['features'] = features.tolist()
            feature_vectors.append(features)
        
        # Store in vector database
        vector_db.store_vectors(frames_info, feature_vectors)
        
        # Clean up uploaded video file
        os.remove(file_path)
        
        return VideoProcessResponse(
            message="Video processed successfully",
            total_frames=len(frames_info),
            frames=frames_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/search-similar", response_model=SimilaritySearchResponse)
async def search_similar_frames(
    file: UploadFile = File(...),
    limit: int = Query(5, description="Number of similar frames to return")
):
    """
    Upload an image and find similar frames in the database
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded image temporarily
        temp_path = f"uploads/temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract features from uploaded image
        query_features = feature_extractor.extract_features(temp_path)
        
        # Search for similar frames
        similar_frames = vector_db.search_similar(query_features, limit)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return SimilaritySearchResponse(
            message="Similar frames found",
            query_features=query_features.tolist(),
            similar_frames=similar_frames
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching similar frames: {str(e)}")

@app.get("/frame/{frame_id}")
async def get_frame(frame_id: str):
    """
    Get a specific frame image by ID
    """
    frame_path = f"frames/{frame_id}"
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="Frame not found")
    
    return FileResponse(frame_path)

@app.get("/frames")
async def list_frames():
    """
    List all available frames
    """
    try:
        frames = []
        for filename in os.listdir("frames"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                frames.append({
                    "id": filename,
                    "url": f"/frame/{filename}"
                })
        return {"frames": frames, "total": len(frames)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing frames: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

