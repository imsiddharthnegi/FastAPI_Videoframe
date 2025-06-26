import cv2
import os
import uuid
from typing import List, Dict
import numpy as np

class VideoProcessor:
    """
    Handles video processing and frame extraction functionality
    """
    
    def __init__(self, frames_dir: str = "frames"):
        """
        Initialize VideoProcessor
        
        Args:
            frames_dir: Directory to save extracted frames
        """
        self.frames_dir = frames_dir
        os.makedirs(frames_dir, exist_ok=True)
    
    def extract_frames(self, video_path: str, interval: float = 1.0) -> List[Dict]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to the video file
            interval: Time interval between frames in seconds
            
        Returns:
            List of dictionaries containing frame information
        """
        frames_info = []
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video info: FPS={fps}, Total frames={total_frames}, Duration={duration:.2f}s")
        
        # Calculate frame interval
        frame_interval = int(fps * interval)
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frame_id = f"frame_{extracted_count:06d}_{uuid.uuid4().hex[:8]}.jpg"
                frame_path = os.path.join(self.frames_dir, frame_id)
                
                # Save frame as image
                cv2.imwrite(frame_path, frame)
                
                # Store frame information
                frame_info = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "path": frame_path,
                    "frame_number": frame_count
                }
                
                frames_info.append(frame_info)
                extracted_count += 1
                
                print(f"Extracted frame {extracted_count} at {timestamp:.2f}s")
            
            frame_count += 1
        
        cap.release()
        
        print(f"Total frames extracted: {len(frames_info)}")
        return frames_info
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get basic information about a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration,
            "format": os.path.splitext(video_path)[1].lower()
        }
    
    def validate_video_file(self, file_path: str) -> bool:
        """
        Validate if the file is a valid video file
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if valid video file, False otherwise
        """
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False
            
            # Try to read first frame
            ret, _ = cap.read()
            cap.release()
            
            return ret
        except Exception:
            return False
    
    def cleanup_frames(self, frame_ids: List[str] = None):
        """
        Clean up extracted frame files
        
        Args:
            frame_ids: List of specific frame IDs to delete. If None, deletes all frames.
        """
        if frame_ids is None:
            # Delete all frames
            for filename in os.listdir(self.frames_dir):
                file_path = os.path.join(self.frames_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            # Delete specific frames
            for frame_id in frame_ids:
                file_path = os.path.join(self.frames_dir, frame_id)
                if os.path.exists(file_path):
                    os.remove(file_path)

