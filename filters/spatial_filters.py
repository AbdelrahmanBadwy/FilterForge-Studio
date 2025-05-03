import numpy as np
import cv2

def laplacian_filter(image):
    """Apply Laplacian filter for edge detection
    
    Args:
        image: input grayscale image
    
    Returns:
        filtered image
    """
    # Apply Laplacian filter using OpenCV
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Convert back to uint8 and normalize
    laplacian = np.abs(laplacian)
    laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
    
    return laplacian

def high_boost_filter(image, alpha=1.5):
    """Apply high-boost filtering
    
    Args:
        image: input grayscale image
        alpha: boost factor (typically > 1)
    
    Returns:
        filtered image
    """
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Calculate unsharp mask
    mask = image.astype(float) - blurred.astype(float)
    
    # Add weighted mask to original image
    high_boost = image.astype(float) + alpha * mask
    
    # Normalize result to 0-255 range
    high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)
    
    return high_boost

def apply_custom_kernel(image, kernel):
    """Apply custom kernel to image
    
    Args:
        image: input grayscale image
        kernel: 2D numpy array representing the kernel
        
    Returns:
        filtered image
    """
    # Apply custom kernel using filter2D
    result = cv2.filter2D(image, -1, kernel)
    return result

def sobel_filter(image, direction='both'):
    """Apply Sobel filter for edge detection
    
    Args:
        image: input grayscale image
        direction: 'x', 'y', or 'both'
        
    Returns:
        filtered image
    """
    # Apply Sobel filter
    if direction == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    elif direction == 'y':
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    else:  # both
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalize result
    sobel = np.abs(sobel)
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)
    
    return sobel

def canny_edge_detector(image, threshold1=100, threshold2=200):
    """Apply Canny edge detector
    
    Args:
        image: input grayscale image
        threshold1: lower threshold
        threshold2: upper threshold
        
    Returns:
        edge image
    """
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter for edge-preserving smoothing
    
    Args:
        image: input grayscale image
        d: diameter of each pixel neighborhood
        sigma_color: filter sigma in color space
        sigma_space: filter sigma in coordinate space
        
    Returns:
        filtered image
    """
    result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return result

def median_filter(image, ksize=5):
    """Apply median filter for noise reduction
    
    Args:
        image: input grayscale image
        ksize: kernel size (must be odd)
        
    Returns:
        filtered image
    """
    # Ensure kernel size is odd
    if ksize % 2 == 0:
        ksize += 1
    
    result = cv2.medianBlur(image, ksize)
    return result

# Common kernels
KERNELS = {
    'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    'edge_detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'box_blur': np.ones((3, 3)) / 9,
    'gaussian_blur': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
    'prewitt_x': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    'prewitt_y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
} 