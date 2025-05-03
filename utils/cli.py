import argparse
import os
import sys
from filters import frequency_filters, spatial_filters, noise, segmentation

def create_parser():
    """Create argument parser for command-line interface
    
    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='FilterForge Studio - Image Processing Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main options
    parser.add_argument('--gui', action='store_true', 
                        help='Launch GUI (default if no arguments provided)')
    parser.add_argument('--cli', action='store_true',
                        help='Run in command-line mode')
    
    # Input/output options
    parser.add_argument('-i', '--input', type=str,
                        help='Input image or directory path')
    parser.add_argument('-o', '--output', type=str,
                        help='Output image or directory path')
    parser.add_argument('--batch', action='store_true',
                        help='Process a directory of images in batch mode')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for batch processing')
    
    # Filter options
    filter_group = parser.add_argument_group('Filter Options')
    filter_group.add_argument('--filter', type=str, choices=[
        'ideal_low', 'butterworth_low', 'gaussian_low',
        'ideal_high', 'butterworth_high', 'gaussian_high',
        'laplacian', 'high_boost', 'sobel', 'canny',
        'median', 'bilateral', 'custom'
    ], help='Filter type to apply')
    
    # Filter parameters
    filter_params = parser.add_argument_group('Filter Parameters')
    filter_params.add_argument('--cutoff', type=int, default=50,
                               help='Cutoff frequency for frequency domain filters')
    filter_params.add_argument('--order', type=int, default=2,
                               help='Order for Butterworth filter')
    filter_params.add_argument('--boost', type=float, default=1.5,
                               help='Boost factor for high-boost filter')
    filter_params.add_argument('--kernel', type=str,
                               help='Path to CSV file with custom kernel')
    filter_params.add_argument('--kernel-type', type=str, choices=list(spatial_filters.KERNELS.keys()),
                               help='Predefined kernel type')
    
    # Noise options
    noise_group = parser.add_argument_group('Noise Options')
    noise_group.add_argument('--add-noise', type=str, choices=[
        'gaussian', 'salt_pepper', 'speckle', 'poisson'
    ], help='Add noise to image')
    noise_group.add_argument('--noise-params', type=str,
                             help='Noise parameters in format "param1=value1,param2=value2"')
    noise_group.add_argument('--denoise', type=str, choices=[
        'median', 'bilateral', 'gaussian', 'nlm'
    ], help='Apply denoising')
    
    # Segmentation options
    seg_group = parser.add_argument_group('Segmentation Options')
    seg_group.add_argument('--segment', type=str, choices=[
        'threshold', 'adaptive', 'otsu', 'watershed'
    ], help='Apply segmentation')
    seg_group.add_argument('--threshold', type=int, default=127,
                           help='Threshold value for segmentation')
    seg_group.add_argument('--overlay', action='store_true',
                           help='Overlay segmentation on original image')
    
    # Feature detection options
    feat_group = parser.add_argument_group('Feature Detection Options')
    feat_group.add_argument('--detect-corners', action='store_true',
                            help='Detect corners in image')
    feat_group.add_argument('--detect-blobs', action='store_true',
                            help='Detect blobs in image')
    feat_group.add_argument('--max-corners', type=int, default=50,
                            help='Maximum number of corners to detect')
    
    # Visualization options
    vis_group = parser.add_argument_group('Visualization Options')
    vis_group.add_argument('--show-spectrum', action='store_true',
                           help='Show frequency spectrum')
    vis_group.add_argument('--show-histogram', action='store_true',
                           help='Show image histogram')
    vis_group.add_argument('--show-metrics', action='store_true',
                           help='Show image quality metrics')
    
    # Format options
    format_group = parser.add_argument_group('Format Options')
    format_group.add_argument('--format', type=str, choices=['jpg', 'png', 'tiff'],
                              help='Output image format')
    format_group.add_argument('--quality', type=int, default=95,
                              help='JPEG output quality (0-100)')
    
    return parser

def parse_noise_params(params_str):
    """Parse noise parameters from string
    
    Args:
        params_str: string in format "param1=value1,param2=value2"
        
    Returns:
        dictionary of parameters
    """
    if not params_str:
        return {}
        
    params = {}
    pairs = params_str.split(',')
    
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=')
            
            # Convert numeric values
            try:
                # Try as integer
                value = int(value)
            except ValueError:
                try:
                    # Try as float
                    value = float(value)
                except ValueError:
                    # Keep as string
                    pass
            
            params[key.strip()] = value
    
    return params

def process_args(args):
    """Process command-line arguments and run appropriate action
    
    Args:
        args: parsed command-line arguments
        
    Returns:
        0 on success, non-zero on error
    """
    # Import here to avoid circular imports
    from utils.batch_processor import BatchProcessor
    from utils.image_utils import load_image, save_image, load_kernel_from_csv
    
    # Determine if GUI should be launched
    if not args.cli and (args.gui or len(sys.argv) <= 1):
        print("Launching GUI...")
        # Import and run GUI (will be implemented in gui module)
        from gui.app import run_app
        return run_app()
    
    # CLI mode requires input path
    if not args.input:
        print("Error: Input path is required in CLI mode")
        return 1
    
    # Determine if batch processing should be used
    is_batch = args.batch or os.path.isdir(args.input)
    
    if is_batch:
        # Batch processing
        if not args.output:
            output_dir = args.input + '_processed'
        else:
            output_dir = args.output
            
        processor = BatchProcessor(args.input, output_dir, args.workers)
        
        # Determine which function to use
        if args.filter:
            if args.filter in ['ideal_low', 'butterworth_low', 'gaussian_low',
                             'ideal_high', 'butterworth_high', 'gaussian_high']:
                # Frequency domain filter
                process_func = lambda img: frequency_filters.create_frequency_filters(
                    img, args.filter, args.cutoff, args.order)[0]
                
            elif args.filter == 'laplacian':
                process_func = spatial_filters.laplacian_filter
                
            elif args.filter == 'high_boost':
                process_func = lambda img: spatial_filters.high_boost_filter(img, args.boost)
                
            elif args.filter == 'sobel':
                process_func = lambda img: spatial_filters.sobel_filter(img, 'both')
                
            elif args.filter == 'canny':
                process_func = spatial_filters.canny_edge_detector
                
            elif args.filter == 'median':
                process_func = lambda img: spatial_filters.median_filter(img, 5)
                
            elif args.filter == 'bilateral':
                process_func = spatial_filters.bilateral_filter
                
            elif args.filter == 'custom':
                if args.kernel:
                    kernel = load_kernel_from_csv(args.kernel)
                elif args.kernel_type:
                    kernel = spatial_filters.KERNELS[args.kernel_type]
                else:
                    print("Error: Custom filter requires --kernel or --kernel-type")
                    return 1
                    
                process_func = lambda img: spatial_filters.apply_custom_kernel(img, kernel)
        
        elif args.add_noise:
            # Parse noise parameters
            noise_params = parse_noise_params(args.noise_params)
            
            if args.add_noise == 'gaussian':
                mean = noise_params.get('mean', 0)
                sigma = noise_params.get('sigma', 25)
                process_func = lambda img: noise.add_gaussian_noise(img, mean, sigma)
                
            elif args.add_noise == 'salt_pepper':
                salt_prob = noise_params.get('salt', 0.02)
                pepper_prob = noise_params.get('pepper', 0.02)
                process_func = lambda img: noise.add_salt_pepper_noise(img, salt_prob, pepper_prob)
                
            elif args.add_noise == 'speckle':
                intensity = noise_params.get('intensity', 0.1)
                process_func = lambda img: noise.add_speckle_noise(img, intensity)
                
            elif args.add_noise == 'poisson':
                process_func = noise.add_poisson_noise
        
        elif args.denoise:
            # Parse denoise parameters
            denoise_params = parse_noise_params(args.noise_params or '')
            process_func = lambda img: noise.remove_noise(img, args.denoise, **denoise_params)
            
        elif args.segment:
            if args.segment == 'threshold':
                process_func = lambda img: segmentation.threshold_segment(img, args.threshold, 'binary')
                
            elif args.segment == 'adaptive':
                process_func = lambda img: segmentation.adaptive_threshold(img)
                
            elif args.segment == 'otsu':
                process_func = lambda img: segmentation.threshold_segment(img, 0, 'otsu')
                
            elif args.segment == 'watershed':
                process_func = lambda img: segmentation.watershed_segmentation(img)[0]
        
        else:
            print("Error: No processing operation specified")
            return 1
            
        # Process batch
        results = processor.process_batch(process_func, file_format=args.format)
        
        # Show statistics
        processor.get_processing_stats(results)
        
    else:
        # Single image processing
        img = load_image(args.input)
        
        if img is None:
            print(f"Error: Could not read image from {args.input}")
            return 1
            
        # Determine output path
        if not args.output:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_processed{ext}"
            
        # Determine which operation to perform
        processed_img = img  # Default (no operation)
        
        if args.filter:
            if args.filter in ['ideal_low', 'butterworth_low', 'gaussian_low',
                             'ideal_high', 'butterworth_high', 'gaussian_high']:
                # Frequency domain filter
                processed_img, _, _ = frequency_filters.create_frequency_filters(
                    img, args.filter, args.cutoff, args.order, 
                    boost_factor=args.boost if 'high' in args.filter else None)
                
                # Show spectrum if requested
                if args.show_spectrum:
                    # Import display functionality
                    import matplotlib.pyplot as plt
                    _, _, f_transform = frequency_filters.create_frequency_filters(
                        img, args.filter, args.cutoff, args.order)
                    magnitude, phase, _ = frequency_filters.visualize_frequency_domain(f_transform)
                    
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(magnitude, cmap='viridis')
                    plt.title('Magnitude Spectrum (log scale)')
                    plt.colorbar()
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(phase, cmap='hsv')
                    plt.title('Phase Spectrum')
                    plt.colorbar()
                    
                    plt.tight_layout()
                    plt.show()
                
            elif args.filter == 'laplacian':
                processed_img = spatial_filters.laplacian_filter(img)
                
            elif args.filter == 'high_boost':
                processed_img = spatial_filters.high_boost_filter(img, args.boost)
                
            elif args.filter == 'sobel':
                processed_img = spatial_filters.sobel_filter(img, 'both')
                
            elif args.filter == 'canny':
                processed_img = spatial_filters.canny_edge_detector(img)
                
            elif args.filter == 'median':
                processed_img = spatial_filters.median_filter(img, 5)
                
            elif args.filter == 'bilateral':
                processed_img = spatial_filters.bilateral_filter(img)
                
            elif args.filter == 'custom':
                if args.kernel:
                    kernel = load_kernel_from_csv(args.kernel)
                elif args.kernel_type:
                    kernel = spatial_filters.KERNELS[args.kernel_type]
                else:
                    print("Error: Custom filter requires --kernel or --kernel-type")
                    return 1
                    
                processed_img = spatial_filters.apply_custom_kernel(img, kernel)
        
        elif args.add_noise:
            # Parse noise parameters
            noise_params = parse_noise_params(args.noise_params)
            
            if args.add_noise == 'gaussian':
                mean = noise_params.get('mean', 0)
                sigma = noise_params.get('sigma', 25)
                processed_img = noise.add_gaussian_noise(img, mean, sigma)
                
            elif args.add_noise == 'salt_pepper':
                salt_prob = noise_params.get('salt', 0.02)
                pepper_prob = noise_params.get('pepper', 0.02)
                processed_img = noise.add_salt_pepper_noise(img, salt_prob, pepper_prob)
                
            elif args.add_noise == 'speckle':
                intensity = noise_params.get('intensity', 0.1)
                processed_img = noise.add_speckle_noise(img, intensity)
                
            elif args.add_noise == 'poisson':
                processed_img = noise.add_poisson_noise(img)
        
        elif args.denoise:
            # Parse denoise parameters
            denoise_params = parse_noise_params(args.noise_params or '')
            processed_img = noise.remove_noise(img, args.denoise, **denoise_params)
            
        elif args.segment:
            if args.segment == 'threshold':
                seg_img = segmentation.threshold_segment(img, args.threshold, 'binary')
                
            elif args.segment == 'adaptive':
                seg_img = segmentation.adaptive_threshold(img)
                
            elif args.segment == 'otsu':
                seg_img = segmentation.threshold_segment(img, 0, 'otsu')
                
            elif args.segment == 'watershed':
                seg_img, _ = segmentation.watershed_segmentation(img)
                
            # Overlay if requested
            if args.overlay:
                processed_img = segmentation.overlay_segmentation(img, seg_img)
            else:
                processed_img = seg_img
        
        elif args.detect_corners:
            corners = segmentation.detect_corners(img, max_corners=args.max_corners)
            # Create a color version for visualization
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in corners:
                cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
            processed_img = vis_img
            
        elif args.detect_blobs:
            keypoints = segmentation.detect_blobs(img)
            # Create a color version for visualization
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vis_img = cv2.drawKeypoints(vis_img, keypoints, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            processed_img = vis_img
        
        # Show histogram if requested
        if args.show_histogram:
            # Import display functionality
            from utils.metrics import compare_histograms
            import matplotlib.pyplot as plt
            
            if np.array_equal(img, processed_img):
                # Only show original histogram
                from utils.metrics import plot_histogram
                plt.figure(figsize=(10, 4))
                hist = plot_histogram(img, "Image Histogram")
                plt.show()
            else:
                # Compare original and processed histograms
                fig = compare_histograms(img, processed_img)
                plt.show()
        
        # Show metrics if requested
        if args.show_metrics and not np.array_equal(img, processed_img):
            from utils.metrics import calculate_metrics, print_metrics
            metrics = calculate_metrics(img, processed_img)
            print_metrics(metrics)
        
        # Save output image
        if args.format:
            base, _ = os.path.splitext(args.output)
            args.output = f"{base}.{args.format}"
            
        # Set quality for JPEG format
        params = None
        if args.format == 'jpg' or args.output.lower().endswith('.jpg') or args.output.lower().endswith('.jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
            
        success = save_image(processed_img, args.output, params)
        
        if success:
            print(f"Saved processed image to {args.output}")
        else:
            print(f"Error: Could not save image to {args.output}")
            return 1
    
    return 0 