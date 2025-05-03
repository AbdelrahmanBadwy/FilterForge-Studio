#!/usr/bin/env python3
"""
FilterForge Studio - Image Processing Tool

This is the main entry point for the application. 
It parses command-line arguments and launches either the GUI or CLI mode.
"""

import sys
from utils.cli import create_parser, process_args
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def create_frequency_filters(image, filter_type, cutoff_freq, order=2, boost_factor=None):
    """Apply frequency domain filters (Low Pass and High Pass)
    
    Args:
        image: input grayscale image
        filter_type: string indicating filter type 
            ('ideal_low', 'butterworth_low', 'gaussian_low',
             'ideal_high', 'butterworth_high', 'gaussian_high')
        cutoff_freq: cutoff frequency (radius)
        order: order for Butterworth filter
        boost_factor: factor for high-boost filtering (only for high pass)
    
    Returns:
        filtered image
    """
    # Get image dimensions
    rows, cols = image.shape
    
    # Create meshgrid for filter calculation
    u = np.fft.fftshift(np.fft.fftfreq(cols))
    v = np.fft.fftshift(np.fft.fftfreq(rows))
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Calculate distance from origin (center frequency)
    D = np.sqrt(u_grid**2 + v_grid**2)
    
    # Normalize cutoff frequency
    cutoff_freq = cutoff_freq / max(rows, cols)
    
    # Create filter based on specified type
    if filter_type == 'ideal_low':
        H = (D <= cutoff_freq).astype(float)
    elif filter_type == 'butterworth_low':
        H = 1 / (1 + (D / cutoff_freq)**(2 * order))
    elif filter_type == 'gaussian_low':
        H = np.exp(-(D**2) / (2 * cutoff_freq**2))
    elif filter_type == 'ideal_high':
        H = (D > cutoff_freq).astype(float)
    elif filter_type == 'butterworth_high':
        H = 1 - 1 / (1 + (D / cutoff_freq)**(2 * order))
    elif filter_type == 'gaussian_high':
        H = 1 - np.exp(-(D**2) / (2 * cutoff_freq**2))
    else:
        raise ValueError("Invalid filter type")
    
    # Apply high-boost if requested
    if boost_factor is not None and 'high' in filter_type:
        H = boost_factor + H
    
    # Apply filter in frequency domain
    f_transform = np.fft.fftshift(np.fft.fft2(image))
    filtered_f = f_transform * H
    filtered_image = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_f)))
    
    # Normalize result to 0-255 range
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image

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

def display_results(original, filtered_images, titles):
    """Display original and filtered images
    
    Args:
        original: original image
        filtered_images: list of filtered images
        titles: list of titles for each image
    """
    plt.figure(figsize=(15, 10))
    
    # Display original image
    plt.subplot(3, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Display filtered images
    for i, (img, title) in enumerate(zip(filtered_images, titles)):
        plt.subplot(3, 3, i + 2)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main entry point for the application"""
    # Create argument parser
    parser = create_parser()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process arguments
    return process_args(args)

if __name__ == "__main__":
    sys.exit(main())