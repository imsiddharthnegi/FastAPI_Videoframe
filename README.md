# Video Frame Analyzer

A comprehensive FastAPI application for video processing, frame extraction, feature vector computation, and similarity search using vector databases.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

Video Frame Analyzer is a modern, scalable application built with FastAPI that provides powerful video processing capabilities. The application extracts frames from uploaded videos, computes feature vectors using color histograms, stores them in a vector database (Qdrant), and enables similarity search functionality.

This project is ideal for applications requiring:
- Video content analysis
- Frame-based similarity search
- Video indexing and retrieval
- Computer vision preprocessing
- Media asset management

## Features

### Core Functionality
- **Video Processing**: Extract frames from uploaded videos at specified intervals
- **Feature Extraction**: Compute color histogram-based feature vectors from frames
- **Vector Storage**: Store feature vectors in Qdrant vector database for efficient retrieval
- **Similarity Search**: Find similar frames based on visual content using cosine similarity
- **RESTful API**: Complete REST API with automatic documentation via FastAPI

### Technical Features
- **Asynchronous Processing**: Built on FastAPI for high-performance async operations
- **Scalable Architecture**: Modular design with separate components for different functionalities
- **Comprehensive Testing**: Full test suite including unit tests and integration tests
- **Error Handling**: Robust error handling with detailed error messages
- **CORS Support**: Cross-origin resource sharing enabled for web applications
- **Automatic Documentation**: Interactive API documentation with Swagger UI




## Architecture

The Video Frame Analyzer follows a modular architecture with clear separation of concerns:

```
video-frame-analyzer/
├── app/
│   ├── __init__.py
│   ├── models.py          # Pydantic models for API requests/responses
│   ├── video_processor.py # Video processing and frame extraction
│   ├── feature_extractor.py # Feature vector computation
│   └── vector_db.py       # Vector database operations
├── main.py                # FastAPI application entry point
├── requirements.txt       # Python dependencies
├── frames/               # Directory for extracted frames
├── uploads/              # Directory for uploaded videos
└── README.md            # This file
```

### Component Overview

#### VideoProcessor (`app/video_processor.py`)
Handles all video-related operations including:
- Video file validation and metadata extraction
- Frame extraction at specified intervals using OpenCV
- Frame saving and management
- Video format support (MP4, AVI, MOV, etc.)

#### FeatureExtractor (`app/feature_extractor.py`)
Responsible for feature vector computation:
- Color histogram computation in multiple color spaces (HSV, RGB, LAB)
- Advanced feature extraction including texture and shape information
- Similarity computation using various metrics (cosine, euclidean, correlation)
- Batch processing capabilities for multiple images

#### VectorDatabase (`app/vector_db.py`)
Manages vector database operations:
- Qdrant client integration for vector storage and retrieval
- Collection management and configuration
- Similarity search with configurable parameters
- Data export/import functionality for backup and migration

#### API Models (`app/models.py`)
Defines Pydantic models for:
- Request validation and serialization
- Response formatting and documentation
- Type safety and automatic API documentation generation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 2GB of available disk space
- Internet connection for package installation

### System Dependencies

For Ubuntu/Debian systems:
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install libopencv-dev python3-opencv
```

For macOS:
```bash
brew install python3
brew install opencv
```

For Windows:
```bash
# Install Python 3.8+ from python.org
# OpenCV will be installed via pip
```

### Python Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd video-frame-analyzer
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

### Verify Installation

Run the following command to verify all dependencies are installed correctly:
```bash
python3 -c "import cv2, numpy, qdrant_client, fastapi; print('All dependencies installed successfully')"
```

## Quick Start

### 1. Start the Application

```bash
# Navigate to the project directory
cd video-frame-analyzer

# Start the FastAPI server
python3 main.py
```

The application will start on `http://localhost:8000` by default.

### 2. Access the API Documentation

Open your web browser and navigate to:
- **Interactive API Documentation**: `http://localhost:8000/docs`
- **Alternative Documentation**: `http://localhost:8000/redoc`

### 3. Test the Health Endpoint

```bash
curl -X GET "http://localhost:8000/health"
```

Expected response:
```json
{
  "status": "healthy",
  "message": "Video Frame Analyzer is running"
}
```

### 4. Upload and Process a Video

```bash
curl -X POST "http://localhost:8000/upload-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_video.mp4" \
  -F "interval=1.0"
```

### 5. Search for Similar Frames

```bash
curl -X POST "http://localhost:8000/search-similar" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@query_image.jpg" \
  -F "limit=5"
```


## API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, the API does not require authentication. For production deployments, consider implementing API key authentication or OAuth2.

### Content Types
- **Request**: `multipart/form-data` for file uploads, `application/json` for other requests
- **Response**: `application/json` for all responses except frame retrieval

### Endpoints

#### 1. Health Check
**GET** `/health`

Check the health status of the application.

**Parameters:** None

**Response:**
```json
{
  "status": "healthy",
  "message": "Video Frame Analyzer is running"
}
```

**Status Codes:**
- `200`: Service is healthy and running

---

#### 2. Root Information
**GET** `/`

Get basic information about the API and available endpoints.

**Parameters:** None

**Response:**
```json
{
  "message": "Video Frame Analyzer API",
  "version": "1.0.0",
  "endpoints": {
    "upload_video": "/upload-video",
    "search_similar": "/search-similar",
    "get_frame": "/frame/{frame_id}",
    "health": "/health"
  }
}
```

---

#### 3. Upload and Process Video
**POST** `/upload-video`

Upload a video file and extract frames at specified intervals.

**Parameters:**
- `file` (required): Video file to upload (multipart/form-data)
  - Supported formats: MP4, AVI, MOV, MKV, WMV
  - Maximum file size: 100MB (configurable)
- `interval` (optional): Frame extraction interval in seconds (default: 1.0)
  - Type: float
  - Range: 0.1 - 10.0 seconds

**Request Example:**
```bash
curl -X POST "http://localhost:8000/upload-video" \
  -F "file=@sample_video.mp4" \
  -F "interval=2.0"
```

**Response:**
```json
{
  "message": "Video processed successfully",
  "total_frames": 15,
  "frames": [
    {
      "frame_id": "frame_000000_abc123.jpg",
      "timestamp": 0.0,
      "path": "frames/frame_000000_abc123.jpg",
      "features": [0.123, 0.456, ...]
    },
    {
      "frame_id": "frame_000001_def456.jpg",
      "timestamp": 2.0,
      "path": "frames/frame_000001_def456.jpg",
      "features": [0.789, 0.012, ...]
    }
  ]
}
```

**Status Codes:**
- `200`: Video processed successfully
- `400`: Invalid file format or parameters
- `413`: File too large
- `500`: Processing error

---

#### 4. Search Similar Frames
**POST** `/search-similar`

Upload an image and find similar frames in the database.

**Parameters:**
- `file` (required): Image file for similarity search (multipart/form-data)
  - Supported formats: JPG, JPEG, PNG, BMP, TIFF
  - Maximum file size: 10MB
- `limit` (optional): Maximum number of similar frames to return (default: 5)
  - Type: integer
  - Range: 1 - 50

**Request Example:**
```bash
curl -X POST "http://localhost:8000/search-similar" \
  -F "file=@query_image.jpg" \
  -F "limit=10"
```

**Response:**
```json
{
  "message": "Similar frames found",
  "query_features": [0.123, 0.456, 0.789, ...],
  "similar_frames": [
    {
      "frame_id": "frame_000005_xyz789.jpg",
      "timestamp": 10.0,
      "path": "frames/frame_000005_xyz789.jpg",
      "similarity_score": 0.9876,
      "features": [0.124, 0.457, 0.788, ...]
    },
    {
      "frame_id": "frame_000003_mno345.jpg",
      "timestamp": 6.0,
      "path": "frames/frame_000003_mno345.jpg",
      "similarity_score": 0.8765,
      "features": [0.125, 0.458, 0.787, ...]
    }
  ]
}
```

**Status Codes:**
- `200`: Search completed successfully
- `400`: Invalid image format
- `404`: No similar frames found
- `500`: Search error

---

#### 5. Get Frame Image
**GET** `/frame/{frame_id}`

Retrieve a specific frame image by its ID.

**Parameters:**
- `frame_id` (required): The unique identifier of the frame
  - Type: string
  - Format: filename with extension (e.g., "frame_000001_abc123.jpg")

**Request Example:**
```bash
curl -X GET "http://localhost:8000/frame/frame_000001_abc123.jpg" \
  --output downloaded_frame.jpg
```

**Response:**
- **Content-Type**: `image/jpeg` or `image/png`
- **Body**: Binary image data

**Status Codes:**
- `200`: Frame retrieved successfully
- `404`: Frame not found
- `500`: Retrieval error

---

#### 6. List All Frames
**GET** `/frames`

Get a list of all available frames in the system.

**Parameters:** None

**Response:**
```json
{
  "frames": [
    {
      "id": "frame_000000_abc123.jpg",
      "url": "/frame/frame_000000_abc123.jpg"
    },
    {
      "id": "frame_000001_def456.jpg",
      "url": "/frame/frame_000001_def456.jpg"
    }
  ],
  "total": 25
}
```

**Status Codes:**
- `200`: Frames listed successfully
- `500`: Listing error

### Error Handling

All endpoints return consistent error responses in the following format:

```json
{
  "detail": "Error description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

Common error codes:
- `INVALID_FILE_FORMAT`: Unsupported file format
- `FILE_TOO_LARGE`: File exceeds size limit
- `PROCESSING_ERROR`: Error during video/image processing
- `DATABASE_ERROR`: Vector database operation failed
- `NOT_FOUND`: Requested resource not found


## Usage Examples

### Python Client Example

```python
import requests
import json

# Configuration
API_BASE_URL = "http://localhost:8000"

def upload_and_process_video(video_path, interval=1.0):
    """Upload a video and process it for frame extraction"""
    with open(video_path, 'rb') as video_file:
        files = {'file': (video_path, video_file, 'video/mp4')}
        data = {'interval': interval}
        
        response = requests.post(f"{API_BASE_URL}/upload-video", 
                               files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Successfully processed video: {result['total_frames']} frames extracted")
            return result
        else:
            print(f"Error: {response.text}")
            return None

def search_similar_frames(image_path, limit=5):
    """Search for frames similar to the given image"""
    with open(image_path, 'rb') as image_file:
        files = {'file': (image_path, image_file, 'image/jpeg')}
        data = {'limit': limit}
        
        response = requests.post(f"{API_BASE_URL}/search-similar",
                               files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Found {len(result['similar_frames'])} similar frames")
            for i, frame in enumerate(result['similar_frames']):
                print(f"  {i+1}. {frame['frame_id']} (score: {frame['similarity_score']:.4f})")
            return result
        else:
            print(f"Error: {response.text}")
            return None

def download_frame(frame_id, output_path):
    """Download a specific frame image"""
    response = requests.get(f"{API_BASE_URL}/frame/{frame_id}")
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Frame saved to {output_path}")
        return True
    else:
        print(f"Error downloading frame: {response.text}")
        return False

# Example usage
if __name__ == "__main__":
    # Process a video
    video_result = upload_and_process_video("sample_video.mp4", interval=2.0)
    
    # Search for similar frames
    search_result = search_similar_frames("query_image.jpg", limit=3)
    
    # Download a frame
    if search_result and search_result['similar_frames']:
        frame_id = search_result['similar_frames'][0]['frame_id']
        download_frame(frame_id, f"downloaded_{frame_id}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const API_BASE_URL = 'http://localhost:8000';

async function uploadVideo(videoPath, interval = 1.0) {
    try {
        const form = new FormData();
        form.append('file', fs.createReadStream(videoPath));
        form.append('interval', interval.toString());

        const response = await axios.post(`${API_BASE_URL}/upload-video`, form, {
            headers: form.getHeaders()
        });

        console.log(`Video processed: ${response.data.total_frames} frames extracted`);
        return response.data;
    } catch (error) {
        console.error('Error uploading video:', error.response?.data || error.message);
        return null;
    }
}

async function searchSimilarFrames(imagePath, limit = 5) {
    try {
        const form = new FormData();
        form.append('file', fs.createReadStream(imagePath));
        form.append('limit', limit.toString());

        const response = await axios.post(`${API_BASE_URL}/search-similar`, form, {
            headers: form.getHeaders()
        });

        console.log(`Found ${response.data.similar_frames.length} similar frames`);
        response.data.similar_frames.forEach((frame, index) => {
            console.log(`  ${index + 1}. ${frame.frame_id} (score: ${frame.similarity_score.toFixed(4)})`);
        });

        return response.data;
    } catch (error) {
        console.error('Error searching frames:', error.response?.data || error.message);
        return null;
    }
}

// Example usage
async function main() {
    const videoResult = await uploadVideo('sample_video.mp4', 2.0);
    const searchResult = await searchSimilarFrames('query_image.jpg', 3);
}

main();
```

### cURL Examples

#### Upload a video with 0.5-second intervals:
```bash
curl -X POST "http://localhost:8000/upload-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/video.mp4" \
  -F "interval=0.5"
```

#### Search for top 10 similar frames:
```bash
curl -X POST "http://localhost:8000/search-similar" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/query_image.jpg" \
  -F "limit=10"
```

#### Download a specific frame:
```bash
curl -X GET "http://localhost:8000/frame/frame_000001_abc123.jpg" \
  --output downloaded_frame.jpg
```

#### Get list of all frames:
```bash
curl -X GET "http://localhost:8000/frames" \
  -H "accept: application/json"
```

## Configuration

### Environment Variables

The application supports the following environment variables for configuration:

```bash
# Server Configuration
HOST=0.0.0.0                    # Server host (default: 0.0.0.0)
PORT=8000                       # Server port (default: 8000)
WORKERS=1                       # Number of worker processes (default: 1)

# File Upload Configuration
MAX_UPLOAD_SIZE=104857600       # Max upload size in bytes (default: 100MB)
UPLOAD_DIR=uploads              # Upload directory (default: uploads)
FRAMES_DIR=frames               # Frames directory (default: frames)

# Feature Extraction Configuration
FEATURE_BINS=32                 # Histogram bins (default: 32)
COLOR_SPACE=HSV                 # Color space (HSV, RGB, LAB) (default: HSV)

# Vector Database Configuration
VECTOR_DB_HOST=localhost        # Qdrant host (default: localhost)
VECTOR_DB_PORT=6333            # Qdrant port (default: 6333)
COLLECTION_NAME=video_frames    # Collection name (default: video_frames)

# Logging Configuration
LOG_LEVEL=INFO                  # Log level (DEBUG, INFO, WARNING, ERROR)
LOG_FILE=app.log               # Log file path (optional)
```

### Configuration File

Create a `config.py` file for advanced configuration:

```python
import os

class Config:
    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # File settings
    MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', 104857600))  # 100MB
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'uploads')
    FRAMES_DIR = os.getenv('FRAMES_DIR', 'frames')
    
    # Feature extraction settings
    FEATURE_BINS = int(os.getenv('FEATURE_BINS', 32))
    COLOR_SPACE = os.getenv('COLOR_SPACE', 'HSV')
    
    # Vector database settings
    VECTOR_DB_HOST = os.getenv('VECTOR_DB_HOST', 'localhost')
    VECTOR_DB_PORT = int(os.getenv('VECTOR_DB_PORT', 6333))
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'video_frames')
    
    # Similarity search settings
    DEFAULT_SEARCH_LIMIT = int(os.getenv('DEFAULT_SEARCH_LIMIT', 5))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.0))
```

### Docker Configuration

Create a `docker-compose.yml` for containerized deployment:

```yaml
version: '3.8'

services:
  video-frame-analyzer:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - VECTOR_DB_HOST=qdrant
    volumes:
      - ./uploads:/app/uploads
      - ./frames:/app/frames
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
```

Corresponding `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads frames

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
```


## Development

### Development Setup

1. **Clone the repository and set up the environment:**
```bash
git clone <repository-url>
cd video-frame-analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Install development dependencies:**
```bash
pip install pytest pytest-asyncio httpx black flake8 mypy
```

3. **Set up pre-commit hooks (optional):**
```bash
pip install pre-commit
pre-commit install
```

### Code Structure

The application follows a clean architecture pattern:

- **`main.py`**: FastAPI application setup and route definitions
- **`app/models.py`**: Pydantic models for request/response validation
- **`app/video_processor.py`**: Video processing logic and frame extraction
- **`app/feature_extractor.py`**: Feature vector computation and similarity metrics
- **`app/vector_db.py`**: Vector database operations and management

### Adding New Features

#### Adding a New Feature Extraction Method

1. Extend the `FeatureExtractor` class in `app/feature_extractor.py`:
```python
def extract_texture_features(self, image_path: str) -> np.ndarray:
    """Extract texture-based features using Local Binary Patterns"""
    # Implementation here
    pass
```

2. Update the API models in `app/models.py` if needed
3. Add corresponding tests in the test suite
4. Update the documentation

#### Adding a New Similarity Metric

1. Add the new metric to the `compute_similarity` method:
```python
def compute_similarity(self, features1: np.ndarray, features2: np.ndarray, 
                      method: str = 'cosine') -> float:
    if method == 'new_metric':
        # Implementation here
        pass
```

2. Update the vector database search functionality if needed
3. Add tests and documentation

### Code Style and Standards

The project follows PEP 8 style guidelines with the following tools:

- **Black**: Code formatting
- **Flake8**: Linting and style checking
- **MyPy**: Type checking
- **isort**: Import sorting

Run code quality checks:
```bash
black --check .
flake8 .
mypy .
isort --check-only .
```

Format code:
```bash
black .
isort .
```

## Testing

### Test Suite Overview

The project includes comprehensive testing with multiple levels:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions and API endpoints
3. **End-to-End Tests**: Test complete workflows from video upload to similarity search

### Running Tests

#### Run all tests:
```bash
pytest
```

#### Run with coverage:
```bash
pytest --cov=app --cov-report=html
```

#### Run specific test categories:
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# API tests only
pytest tests/api/
```

### Test Structure

```
tests/
├── unit/
│   ├── test_video_processor.py
│   ├── test_feature_extractor.py
│   └── test_vector_db.py
├── integration/
│   ├── test_api_endpoints.py
│   └── test_workflow.py
└── fixtures/
    ├── sample_video.mp4
    └── sample_images/
```

### Writing Tests

Example unit test:
```python
import pytest
from app.feature_extractor import FeatureExtractor

def test_feature_extraction():
    extractor = FeatureExtractor(hist_bins=32)
    features = extractor.extract_features("test_image.jpg")
    
    assert features.shape == (96,)  # 32 bins * 3 channels
    assert features.dtype == np.float32
    assert np.sum(features) == pytest.approx(3.0, rel=1e-3)
```

Example integration test:
```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_video_endpoint():
    with open("test_video.mp4", "rb") as video_file:
        response = client.post(
            "/upload-video",
            files={"file": ("test.mp4", video_file, "video/mp4")},
            data={"interval": 1.0}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "total_frames" in data
    assert data["total_frames"] > 0
```

### Performance Testing

Use the included performance test script:
```bash
python tests/performance/load_test.py
```

This script tests:
- Concurrent video uploads
- Similarity search performance
- Memory usage under load
- Response time metrics

## Deployment

### Production Deployment

#### Using Docker

1. **Build the Docker image:**
```bash
docker build -t video-frame-analyzer .
```

2. **Run with Docker Compose:**
```bash
docker-compose up -d
```

#### Using Systemd (Linux)

1. **Create a systemd service file** (`/etc/systemd/system/video-frame-analyzer.service`):
```ini
[Unit]
Description=Video Frame Analyzer API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/video-frame-analyzer
Environment=PATH=/opt/video-frame-analyzer/venv/bin
ExecStart=/opt/video-frame-analyzer/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

2. **Enable and start the service:**
```bash
sudo systemctl enable video-frame-analyzer
sudo systemctl start video-frame-analyzer
```

#### Using Nginx as Reverse Proxy

Configure Nginx to proxy requests to the FastAPI application:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Cloud Deployment

#### AWS Deployment

1. **Using AWS Lambda with Mangum:**
```python
from mangum import Mangum
from main import app

handler = Mangum(app)
```

2. **Using AWS ECS:**
- Build and push Docker image to ECR
- Create ECS task definition
- Deploy to ECS cluster

#### Google Cloud Platform

1. **Using Cloud Run:**
```bash
gcloud run deploy video-frame-analyzer \
  --image gcr.io/PROJECT-ID/video-frame-analyzer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Deployment

1. **Using Azure Container Instances:**
```bash
az container create \
  --resource-group myResourceGroup \
  --name video-frame-analyzer \
  --image myregistry.azurecr.io/video-frame-analyzer:latest \
  --ports 8000
```

### Scaling Considerations

#### Horizontal Scaling
- Use multiple application instances behind a load balancer
- Implement shared storage for frames (AWS S3, Google Cloud Storage)
- Use external Qdrant cluster for vector database

#### Vertical Scaling
- Increase CPU and memory allocation
- Optimize OpenCV operations for multi-threading
- Use GPU acceleration for feature extraction

#### Performance Optimization
- Implement caching for frequently accessed frames
- Use async processing for video uploads
- Implement request queuing for high-load scenarios

## Monitoring and Logging

### Application Monitoring

Add monitoring endpoints:
```python
@app.get("/metrics")
async def get_metrics():
    return {
        "uptime": get_uptime(),
        "processed_videos": get_video_count(),
        "total_frames": get_frame_count(),
        "database_size": get_db_size()
    }
```

### Logging Configuration

Configure structured logging:
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName
        }
        return json.dumps(log_entry)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
```

### Health Checks

Implement comprehensive health checks:
```python
@app.get("/health/detailed")
async def detailed_health_check():
    return {
        "status": "healthy",
        "checks": {
            "database": check_database_connection(),
            "storage": check_storage_availability(),
            "memory": check_memory_usage(),
            "disk": check_disk_space()
        }
    }
```

## Security Considerations

### Input Validation
- File type validation for uploads
- File size limits
- Content scanning for malicious files

### API Security
- Rate limiting implementation
- API key authentication
- CORS configuration
- Input sanitization

### Data Protection
- Secure file storage
- Data encryption at rest
- Secure deletion of temporary files

## Troubleshooting

### Common Issues

#### 1. OpenCV Installation Issues
```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev python3-opencv

# macOS
brew install opencv

# If still having issues, try:
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 2. Qdrant Connection Issues
```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant:latest
```

#### 3. Memory Issues with Large Videos
- Reduce frame extraction interval
- Process videos in smaller chunks
- Increase system memory allocation

#### 4. Slow Similarity Search
- Reduce vector dimensions
- Implement approximate search
- Use GPU acceleration

### Debug Mode

Enable debug mode for detailed error information:
```bash
export DEBUG=true
python main.py
```

### Log Analysis

Common log patterns to monitor:
- Video processing errors
- Database connection issues
- Memory usage warnings
- API response times

## Contributing

We welcome contributions to the Video Frame Analyzer project! Please follow these guidelines:

### Getting Started
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Contribution Guidelines
- Follow the existing code style
- Add comprehensive tests
- Update documentation
- Include meaningful commit messages
- Ensure backward compatibility

### Reporting Issues
When reporting issues, please include:
- Python version and OS
- Error messages and stack traces
- Steps to reproduce the issue
- Sample files if applicable

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **FastAPI**: For the excellent web framework
- **OpenCV**: For powerful computer vision capabilities
- **Qdrant**: For efficient vector database operations
- **NumPy**: For numerical computing support

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review existing issues and discussions

---

**Video Frame Analyzer** - Empowering video analysis with modern AI and vector search capabilities.

