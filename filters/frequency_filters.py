import numpy as np
import cv2

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
    
    return filtered_image, H, f_transform

def visualize_frequency_domain(f_transform, filter_mask=None):
    """Visualize frequency domain representation
    
    Args:
        f_transform: Fourier transform of the image
        filter_mask: Optional filter mask to display
        
    Returns:
        magnitude_spectrum: Log-scaled magnitude spectrum image
        phase_spectrum: Phase spectrum image
        filter_view: Filter mask visualization (if provided)
    """
    # Compute magnitude spectrum (log scale)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform) + 1)
    magnitude_spectrum = np.clip(magnitude_spectrum, 0, 255)
    magnitude_spectrum = ((magnitude_spectrum - magnitude_spectrum.min()) / 
                          (magnitude_spectrum.max() - magnitude_spectrum.min()) * 255).astype(np.uint8)
    
    # Compute phase spectrum
    phase_spectrum = np.angle(f_transform)
    phase_spectrum = ((phase_spectrum - phase_spectrum.min()) / 
                      (phase_spectrum.max() - phase_spectrum.min()) * 255).astype(np.uint8)
    
    # Visualize filter mask if provided
    filter_view = None
    if filter_mask is not None:
        filter_view = (filter_mask * 255).astype(np.uint8)
    
    return magnitude_spectrum, phase_spectrum, filter_view 