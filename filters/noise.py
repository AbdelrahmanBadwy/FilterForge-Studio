import numpy as np
import cv2
from .spatial_filters import median_filter, bilateral_filter

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to image
    
    Args:
        image: input grayscale image
        mean: mean of Gaussian distribution
        sigma: standard deviation of Gaussian distribution
        
    Returns:
        noisy image
    """
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """Add salt and pepper noise to image
    
    Args:
        image: input grayscale image
        salt_prob: probability of salt noise
        pepper_prob: probability of pepper noise
        
    Returns:
        noisy image
    """
    noisy = np.copy(image)
    
    # Salt noise (white pixels)
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy[salt_mask] = 255
    
    # Pepper noise (black pixels)
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy[pepper_mask] = 0
    
    return noisy

def add_speckle_noise(image, intensity=0.1):
    """Add speckle noise to image
    
    Args:
        image: input grayscale image
        intensity: intensity of speckle noise
        
    Returns:
        noisy image
    """
    row, col = image.shape
    speckle = intensity * np.random.randn(row, col)
    noisy = image + image * speckle
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_poisson_noise(image):
    """Add Poisson noise to image
    
    Args:
        image: input grayscale image
        
    Returns:
        noisy image
    """
    # Poisson noise is signal-dependent
    # Normalize image to 0-1 range
    norm_image = image / 255.0
    
    # Generate Poisson noise
    noisy_norm = np.random.poisson(norm_image * 255) / 255
    
    # Scale back to 0-255 range
    noisy = (noisy_norm * 255).astype(np.uint8)
    
    return noisy

def remove_noise(image, method='median', **kwargs):
    """Remove noise from image using specified method
    
    Args:
        image: input noisy grayscale image
        method: denoising method ('median', 'bilateral', 'gaussian', 'nlm')
        **kwargs: additional parameters for specific methods
        
    Returns:
        denoised image
    """
    if method == 'median':
        ksize = kwargs.get('ksize', 5)
        return median_filter(image, ksize)
    
    elif method == 'bilateral':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return bilateral_filter(image, d, sigma_color, sigma_space)
    
    elif method == 'gaussian':
        ksize = kwargs.get('ksize', (5, 5))
        sigma = kwargs.get('sigma', 0)
        return cv2.GaussianBlur(image, ksize, sigma)
    
    elif method == 'nlm':  # Non-local means
        h = kwargs.get('h', 10)  # Filter strength
        template_window_size = kwargs.get('template_window_size', 7)
        search_window_size = kwargs.get('search_window_size', 21)
        return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
    
    else:
        raise ValueError(f"Unknown denoising method: {method}") 