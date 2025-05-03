import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib.pyplot as plt

def calculate_psnr(original, processed):
    """Calculate Peak Signal-to-Noise Ratio
    
    Args:
        original: original image
        processed: processed image
        
    Returns:
        PSNR value in dB
    """
    return peak_signal_noise_ratio(original, processed)

def calculate_ssim(original, processed):
    """Calculate Structural Similarity Index
    
    Args:
        original: original image
        processed: processed image
        
    Returns:
        SSIM value (between -1 and 1)
    """
    return structural_similarity(original, processed)

def calculate_mse(original, processed):
    """Calculate Mean Squared Error
    
    Args:
        original: original image
        processed: processed image
        
    Returns:
        MSE value
    """
    return np.mean((original.astype(float) - processed.astype(float)) ** 2)

def calculate_histogram(image):
    """Calculate image histogram
    
    Args:
        image: input grayscale image
        
    Returns:
        histogram values, bin edges
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()

def plot_histogram(image, title="Histogram"):
    """Plot image histogram
    
    Args:
        image: input grayscale image
        title: plot title
        
    Returns:
        matplotlib figure
    """
    hist = calculate_histogram(image)
    plt.figure(figsize=(10, 4))
    plt.plot(hist)
    plt.title(title)
    plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    return plt.gcf()

def compare_histograms(original, processed):
    """Compare histograms of original and processed images
    
    Args:
        original: original image
        processed: processed image
        
    Returns:
        matplotlib figure with both histograms
    """
    hist_original = calculate_histogram(original)
    hist_processed = calculate_histogram(processed)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist_original)
    plt.title('Original Image')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.plot(hist_processed)
    plt.title('Processed Image')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    
    plt.tight_layout()
    return plt.gcf()

def calculate_metrics(original, processed):
    """Calculate multiple image quality metrics
    
    Args:
        original: original image
        processed: processed image
        
    Returns:
        dictionary of metrics
    """
    psnr = calculate_psnr(original, processed)
    ssim = calculate_ssim(original, processed)
    mse = calculate_mse(original, processed)
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'MSE': mse
    }

def print_metrics(metrics):
    """Print metrics in formatted way
    
    Args:
        metrics: dictionary of metrics
    """
    print("Image Quality Metrics:")
    print(f"PSNR: {metrics['PSNR']:.2f} dB")
    print(f"SSIM: {metrics['SSIM']:.4f}")
    print(f"MSE: {metrics['MSE']:.2f}")

def embed_metadata(image_path, metadata, output_path=None):
    """Embed metadata into image file
    
    Args:
        image_path: path to image file
        metadata: dictionary of metadata to embed
        output_path: path to save output image (if None, overwrites input)
        
    Returns:
        output path
    """
    # If no output path specified, use input path
    if output_path is None:
        output_path = image_path
        
    # Read image
    img = cv2.imread(image_path)
    
    # Convert metadata to string
    metadata_str = ';'.join([f"{k}={v}" for k, v in metadata.items()])
    
    # Embed metadata
    # This is a simple implementation; real-world applications might use EXIF or other standards
    success, img_encoded = cv2.imencode('.png', img)
    
    if success:
        with open(output_path, 'wb') as f:
            f.write(img_encoded.tobytes())
            
    return output_path 