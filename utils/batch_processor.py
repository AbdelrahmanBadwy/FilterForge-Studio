import os
import cv2
import numpy as np
import time
import concurrent.futures
from tqdm import tqdm
from .metrics import embed_metadata

class BatchProcessor:
    """Class for batch processing of multiple images"""
    
    def __init__(self, input_dir, output_dir=None, num_workers=4):
        """Initialize batch processor
        
        Args:
            input_dir: directory with input images
            output_dir: directory for output images (if None, uses input_dir + '_processed')
            num_workers: number of worker threads for parallel processing
        """
        self.input_dir = input_dir
        
        # Set default output directory if not specified
        if output_dir is None:
            self.output_dir = input_dir + '_processed'
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.num_workers = num_workers
        self.image_files = self._get_image_files()
        
    def _get_image_files(self):
        """Get list of image files in input directory
        
        Returns:
            list of image file paths
        """
        valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        image_files = []
        
        for filename in os.listdir(self.input_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(os.path.join(self.input_dir, filename))
                
        return image_files
    
    def _process_single_image(self, image_path, process_func, args=None, kwargs=None):
        """Process a single image
        
        Args:
            image_path: path to input image
            process_func: function to apply to image
            args: positional arguments for process_func
            kwargs: keyword arguments for process_func
            
        Returns:
            tuple of (output_path, processing_time, metadata)
        """
        # Default args and kwargs
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
            
        # Get output filename
        base_name = os.path.basename(image_path)
        output_path = os.path.join(self.output_dir, base_name)
        
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        
        # Process image and measure time
        start_time = time.time()
        processed_img = process_func(img, *args, **kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Handle case where process_func returns multiple values
        if isinstance(processed_img, tuple):
            processed_img = processed_img[0]  # Assume first return value is the image
        
        # Save processed image
        cv2.imwrite(output_path, processed_img)
        
        # Prepare metadata
        metadata = {
            'original_file': base_name,
            'processing_time': f"{processing_time:.3f}s",
            'process_func': process_func.__name__,
        }
        
        # Add function arguments to metadata
        for key, value in kwargs.items():
            if isinstance(value, (int, float, str, bool)):
                metadata[key] = value
                
        # Embed metadata
        embed_metadata(output_path, metadata)
        
        return output_path, processing_time, metadata
    
    def process_batch(self, process_func, args=None, kwargs=None, file_format=None):
        """Process all images in batch
        
        Args:
            process_func: function to apply to each image
            args: positional arguments for process_func
            kwargs: keyword arguments for process_func
            file_format: output file format ('jpg', 'png', 'tiff', or None to keep original)
            
        Returns:
            list of processing results
        """
        # Default args and kwargs
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
            
        results = []
        
        print(f"Processing {len(self.image_files)} images...")
        
        # Sequential processing
        if self.num_workers <= 1:
            for image_path in tqdm(self.image_files):
                result = self._process_single_image(image_path, process_func, args, kwargs)
                if result is not None:
                    results.append(result)
        
        # Parallel processing
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_path = {
                    executor.submit(self._process_single_image, path, process_func, args, kwargs): path
                    for path in self.image_files
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(self.image_files)):
                    result = future.result()
                    if result is not None:
                        results.append(result)
        
        # Change file format if specified
        if file_format is not None:
            self._convert_batch_format(file_format)
                        
        return results
    
    def _convert_batch_format(self, format_name):
        """Convert all processed images to specified format
        
        Args:
            format_name: target format ('jpg', 'png', 'tiff')
        """
        valid_formats = {'jpg', 'jpeg', 'png', 'tiff'}
        
        if format_name.lower() not in valid_formats:
            raise ValueError(f"Invalid format: {format_name}. Valid formats are: {valid_formats}")
            
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            name, ext = os.path.splitext(filename)
            
            if ext.lower()[1:] != format_name.lower():
                # Read image
                img = cv2.imread(file_path)
                
                if img is not None:
                    # Create new filename
                    new_path = os.path.join(self.output_dir, f"{name}.{format_name}")
                    
                    # Save in new format
                    cv2.imwrite(new_path, img)
                    
                    # Remove old file
                    os.remove(file_path)
    
    def get_processing_stats(self, results):
        """Calculate and print processing statistics
        
        Args:
            results: list of processing results from process_batch
            
        Returns:
            dictionary of statistics
        """
        if not results:
            return {}
            
        # Extract processing times
        times = [result[1] for result in results]
        
        stats = {
            'total_images': len(results),
            'total_time': sum(times),
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times)
        }
        
        # Print statistics
        print("\nBatch Processing Statistics:")
        print(f"Total images processed: {stats['total_images']}")
        print(f"Total processing time: {stats['total_time']:.2f} seconds")
        print(f"Average processing time: {stats['avg_time']:.2f} seconds per image")
        print(f"Min processing time: {stats['min_time']:.2f} seconds")
        print(f"Max processing time: {stats['max_time']:.2f} seconds")
        
        return stats 