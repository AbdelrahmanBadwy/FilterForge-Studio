import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from PIL import Image
import io

def load_image(path, as_grayscale=True):
    """Load image from path
    
    Args:
        path: image file path
        as_grayscale: whether to load as grayscale
        
    Returns:
        loaded image as numpy array
    """
    if as_grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(path)
    
def save_image(image, path, params=None):
    """Save image to path
    
    Args:
        image: image as numpy array
        path: output file path
        params: optional quality parameters for different formats
        
    Returns:
        True if successful
    """
    if params is None:
        return cv2.imwrite(path, image)
    else:
        return cv2.imwrite(path, image, params)
    
def resize_image(image, width=None, height=None, keep_aspect_ratio=True):
    """Resize image
    
    Args:
        image: input image
        width: target width (if None, calculated from height and aspect ratio)
        height: target height (if None, calculated from width and aspect ratio)
        keep_aspect_ratio: whether to maintain aspect ratio
        
    Returns:
        resized image
    """
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if keep_aspect_ratio:
        if width is None:
            aspect_ratio = w / h
            width = int(height * aspect_ratio)
        elif height is None:
            aspect_ratio = h / w
            height = int(width * aspect_ratio)
        else:
            # Both width and height provided, determine which dimension to adjust
            aspect_ratio = w / h
            if width / height > aspect_ratio:
                width = int(height * aspect_ratio)
            else:
                height = int(width / aspect_ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV image (numpy array)
    """
    # Convert PIL Image to numpy array
    np_array = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(np_array.shape) == 3 and np_array.shape[2] == 3:
        return cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
    else:
        return np_array

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format
    
    Args:
        cv2_image: OpenCV image (numpy array)
        
    Returns:
        PIL Image object
    """
    # Convert BGR to RGB if color image
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(cv2_image)

def load_kernel_from_csv(filepath):
    """Load custom kernel from CSV file
    
    Args:
        filepath: path to CSV file
        
    Returns:
        numpy array representing kernel
    """
    kernel = []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert strings to floats
            kernel.append([float(x) for x in row])
    
    return np.array(kernel)

def save_kernel_to_csv(kernel, filepath):
    """Save kernel to CSV file
    
    Args:
        kernel: numpy array representing kernel
        filepath: output file path
        
    Returns:
        True if successful
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in kernel:
            writer.writerow(row)
    
    return True

def create_kernel_preview(kernel):
    """Create visual preview of kernel
    
    Args:
        kernel: numpy array representing kernel
        
    Returns:
        image with kernel visualization
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(kernel, cmap='viridis')
    plt.colorbar(label='Weight')
    
    # Add text annotations
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            plt.text(j, i, f"{kernel[i, j]:.2f}", 
                    ha="center", va="center", 
                    color="white" if abs(kernel[i, j]) < 0.7 * kernel.max() else "black")
    
    plt.title("Kernel Visualization")
    plt.tight_layout()
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Convert buffer to image
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    # Convert RGB to BGR (OpenCV format)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def create_matplotlib_figure():
    """Create a new matplotlib figure
    
    Returns:
        matplotlib figure
    """
    return plt.figure(figsize=(10, 8))

def figure_to_image(fig):
    """Convert matplotlib figure to image
    
    Args:
        fig: matplotlib figure
        
    Returns:
        image as numpy array
    """
    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    
    # Convert buffer to image
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    # Convert RGB to BGR (OpenCV format)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img 