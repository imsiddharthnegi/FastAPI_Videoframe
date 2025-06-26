import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

class FeatureExtractor:
    """
    Handles feature extraction from images using color histograms
    """
    
    def __init__(self, hist_bins: int = 32, color_space: str = 'HSV'):
        """
        Initialize FeatureExtractor
        
        Args:
            hist_bins: Number of bins for histogram computation
            color_space: Color space to use ('HSV', 'RGB', 'LAB')
        """
        self.hist_bins = hist_bins
        self.color_space = color_space
        
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract feature vector from an image using color histograms
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert color space if needed
        if self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.color_space == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Compute color histogram for each channel
        histograms = []
        
        for i in range(3):  # 3 channels
            hist = cv2.calcHist([image], [i], None, [self.hist_bins], [0, 256])
            hist = hist.flatten()
            # Normalize histogram
            hist = hist / (hist.sum() + 1e-7)  # Add small epsilon to avoid division by zero
            histograms.extend(hist)
        
        feature_vector = np.array(histograms, dtype=np.float32)
        
        return feature_vector
    
    def extract_features_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Extract features from multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of feature vectors
        """
        features = []
        for path in image_paths:
            try:
                feature = self.extract_features(path)
                features.append(feature)
            except Exception as e:
                print(f"Error extracting features from {path}: {e}")
                # Return zero vector for failed extractions
                features.append(np.zeros(self.hist_bins * 3, dtype=np.float32))
        
        return features
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray, 
                          method: str = 'cosine') -> float:
        """
        Compute similarity between two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            method: Similarity method ('cosine', 'euclidean', 'correlation')
            
        Returns:
            Similarity score (higher means more similar)
        """
        if method == 'cosine':
            # Cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        elif method == 'euclidean':
            # Euclidean distance (convert to similarity)
            distance = np.linalg.norm(features1 - features2)
            similarity = 1.0 / (1.0 + distance)
            return float(similarity)
        
        elif method == 'correlation':
            # Correlation coefficient
            correlation = np.corrcoef(features1, features2)[0, 1]
            if np.isnan(correlation):
                return 0.0
            return float(correlation)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def get_feature_dimension(self) -> int:
        """
        Get the dimension of feature vectors
        
        Returns:
            Feature vector dimension
        """
        return self.hist_bins * 3  # 3 channels
    
    def extract_advanced_features(self, image_path: str) -> np.ndarray:
        """
        Extract more advanced features including texture and shape information
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Advanced feature vector
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        features = []
        
        # 1. Color histogram features (as before)
        color_features = self.extract_features(image_path)
        features.extend(color_features)
        
        # 2. Texture features using Local Binary Pattern
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple texture measure: standard deviation of pixel intensities
        texture_std = np.std(gray)
        features.append(texture_std)
        
        # 3. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # 4. Brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        features.extend([brightness, contrast])
        
        # 5. Color moments (mean, std, skewness for each channel)
        for i in range(3):
            channel = image[:, :, i].flatten()
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            # Simple skewness approximation
            skew_val = np.mean((channel - mean_val) ** 3) / (std_val ** 3 + 1e-7)
            features.extend([mean_val, std_val, skew_val])
        
        return np.array(features, dtype=np.float32)
    
    def visualize_histogram(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualize color histogram of an image
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the histogram plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create histogram plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Show original image
            axes[0, 0].imshow(image_rgb)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Plot histograms for each channel
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
                if i == 0:
                    axes[0, 1].plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
                elif i == 1:
                    axes[1, 0].plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
                else:
                    axes[1, 1].plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
            
            # Set titles and labels
            axes[0, 1].set_title('Red Channel Histogram')
            axes[0, 1].set_xlabel('Pixel Intensity')
            axes[0, 1].set_ylabel('Frequency')
            
            axes[1, 0].set_title('Green Channel Histogram')
            axes[1, 0].set_xlabel('Pixel Intensity')
            axes[1, 0].set_ylabel('Frequency')
            
            axes[1, 1].set_title('Blue Channel Histogram')
            axes[1, 1].set_xlabel('Pixel Intensity')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Histogram saved to: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for histogram visualization")
        except Exception as e:
            print(f"Error creating histogram visualization: {e}")

