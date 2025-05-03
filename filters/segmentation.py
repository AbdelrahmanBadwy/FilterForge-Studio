import numpy as np
import cv2

def threshold_segment(image, threshold=127, method='binary'):
    """Apply thresholding for image segmentation
    
    Args:
        image: input grayscale image
        threshold: threshold value (0-255)
        method: thresholding method ('binary', 'binary_inv', 'otsu')
        
    Returns:
        segmented image
    """
    if method == 'binary':
        _, segmented = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    elif method == 'binary_inv':
        _, segmented = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    elif method == 'otsu':
        _, segmented = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
    
    return segmented

def adaptive_threshold(image, block_size=11, c=2, method='mean'):
    """Apply adaptive thresholding
    
    Args:
        image: input grayscale image
        block_size: size of pixel neighborhood
        c: constant subtracted from mean/gaussian
        method: 'mean' or 'gaussian'
        
    Returns:
        segmented image
    """
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
        
    if method == 'mean':
        segmented = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, block_size, c)
    elif method == 'gaussian':
        segmented = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, block_size, c)
    else:
        raise ValueError(f"Unknown adaptive method: {method}")
    
    return segmented

def detect_corners(image, max_corners=50, quality_level=0.01, min_distance=10):
    """Detect corners in image using Shi-Tomasi algorithm
    
    Args:
        image: input grayscale image
        max_corners: maximum number of corners to detect
        quality_level: minimum quality of corner
        min_distance: minimum Euclidean distance between corners
        
    Returns:
        original image and list of corner coordinates
    """
    corners = cv2.goodFeaturesToTrack(image, max_corners, quality_level, min_distance)
    
    if corners is not None:
        corners = np.int0(corners)
        corners = [c[0] for c in corners]  # Extract coordinates from nested array
    else:
        corners = []
    
    return corners

def detect_blobs(image, min_threshold=10, max_threshold=200, min_area=100, max_area=5000):
    """Detect blobs in image using SimpleBlobDetector
    
    Args:
        image: input grayscale image
        min_threshold: minimum threshold for blob detection
        max_threshold: maximum threshold for blob detection
        min_area: minimum area of blobs
        max_area: maximum area of blobs
        
    Returns:
        list of keypoints
    """
    # Set up parameters for blob detector
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    
    # Filter by area
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(image)
    
    return keypoints

def detect_edges(image, threshold1=100, threshold2=200):
    """Detect edges using Canny edge detector
    
    Args:
        image: input grayscale image
        threshold1: lower threshold
        threshold2: upper threshold
        
    Returns:
        edge image
    """
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges

def watershed_segmentation(image):
    """Apply watershed algorithm for segmentation
    
    Args:
        image: input grayscale image
        
    Returns:
        segmented image with markers
    """
    # Threshold the image
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that background is not 0, but 1
    markers = markers + 1
    
    # Mark the unknown region with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    
    # Create segmentation result visualization
    segmented = np.zeros_like(image)
    segmented[markers == -1] = 255  # Boundaries
    
    return segmented, markers

def overlay_segmentation(image, segmentation, color=(0, 0, 255)):
    """Overlay segmentation on original image
    
    Args:
        image: original grayscale image
        segmentation: binary segmentation mask
        color: overlay color (B, G, R)
        
    Returns:
        image with overlay
    """
    # Convert grayscale to color if needed
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    
    # Create mask from segmentation
    mask = segmentation.astype(bool)
    
    # Create color overlay
    overlay = np.zeros_like(image_color)
    overlay[mask] = color
    
    # Blend original image with overlay
    result = cv2.addWeighted(image_color, 1.0, overlay, 0.5, 0)
    
    return result 