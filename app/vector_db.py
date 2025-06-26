from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import numpy as np
import uuid
import json
import os

class VectorDatabase:
    """
    Handles vector database operations using Qdrant
    """
    
    def __init__(self, collection_name: str = "video_frames", vector_size: int = 96):
        """
        Initialize VectorDatabase
        
        Args:
            collection_name: Name of the collection to store vectors
            vector_size: Dimension of feature vectors
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize Qdrant client (in-memory for development)
        self.client = QdrantClient(":memory:")
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """
        Create a collection in Qdrant if it doesn't exist
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def store_vectors(self, frames_info: List[Dict], feature_vectors: List[np.ndarray]) -> List[str]:
        """
        Store feature vectors with metadata in the database
        
        Args:
            frames_info: List of frame information dictionaries
            feature_vectors: List of feature vectors
            
        Returns:
            List of point IDs that were stored
        """
        if len(frames_info) != len(feature_vectors):
            raise ValueError("Number of frames and feature vectors must match")
        
        points = []
        point_ids = []
        
        for frame_info, features in zip(frames_info, feature_vectors):
            # Generate unique point ID
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Prepare metadata
            payload = {
                "frame_id": frame_info["frame_id"],
                "timestamp": frame_info["timestamp"],
                "path": frame_info["path"],
                "frame_number": frame_info.get("frame_number", 0)
            }
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=features.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Store points in batch
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Stored {len(points)} vectors in database")
            return point_ids
            
        except Exception as e:
            print(f"Error storing vectors: {e}")
            raise
    
    def search_similar(self, query_vector: np.ndarray, limit: int = 5, 
                      score_threshold: float = 0.0) -> List[Dict]:
        """
        Search for similar vectors in the database
        
        Args:
            query_vector: Query feature vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar frames with metadata and scores
        """
        try:
            # Perform similarity search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            similar_frames = []
            for result in search_results:
                frame_data = {
                    "frame_id": result.payload["frame_id"],
                    "timestamp": result.payload["timestamp"],
                    "path": result.payload["path"],
                    "similarity_score": float(result.score),
                    "features": result.vector if result.vector else []
                }
                similar_frames.append(frame_data)
            
            print(f"Found {len(similar_frames)} similar frames")
            return similar_frames
            
        except Exception as e:
            print(f"Error searching similar vectors: {e}")
            return []
    
    def get_vector_by_frame_id(self, frame_id: str) -> Optional[Dict]:
        """
        Retrieve a specific vector by frame ID
        
        Args:
            frame_id: Frame identifier
            
        Returns:
            Frame data with vector if found, None otherwise
        """
        try:
            # Search by frame_id in payload
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="frame_id",
                            match=MatchValue(value=frame_id)
                        )
                    ]
                ),
                limit=1,
                with_vectors=True
            )
            
            if search_results[0]:  # search_results is a tuple (points, next_page_offset)
                result = search_results[0][0]  # First point
                return {
                    "frame_id": result.payload["frame_id"],
                    "timestamp": result.payload["timestamp"],
                    "path": result.payload["path"],
                    "features": result.vector
                }
            
            return None
            
        except Exception as e:
            print(f"Error retrieving vector by frame_id: {e}")
            return None
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection
        
        Returns:
            Collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
            
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def delete_vectors(self, point_ids: List[str]) -> bool:
        """
        Delete vectors by point IDs
        
        Args:
            point_ids: List of point IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            print(f"Deleted {len(point_ids)} vectors")
            return True
            
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all vectors from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self._create_collection()
            print(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def export_vectors(self, output_file: str) -> bool:
        """
        Export all vectors and metadata to a JSON file
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Retrieve all points
            all_points = []
            offset = None
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_vectors=True
                )
                
                points, next_offset = scroll_result
                
                for point in points:
                    point_data = {
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload
                    }
                    all_points.append(point_data)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            # Save to JSON file
            with open(output_file, 'w') as f:
                json.dump(all_points, f, indent=2)
            
            print(f"Exported {len(all_points)} vectors to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error exporting vectors: {e}")
            return False
    
    def import_vectors(self, input_file: str) -> bool:
        """
        Import vectors and metadata from a JSON file
        
        Args:
            input_file: Path to input JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(input_file):
                print(f"Input file not found: {input_file}")
                return False
            
            with open(input_file, 'r') as f:
                points_data = json.load(f)
            
            # Convert to PointStruct objects
            points = []
            for point_data in points_data:
                point = PointStruct(
                    id=point_data["id"],
                    vector=point_data["vector"],
                    payload=point_data["payload"]
                )
                points.append(point)
            
            # Upsert points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"Imported {len(points)} vectors from {input_file}")
            return True
            
        except Exception as e:
            print(f"Error importing vectors: {e}")
            return False

