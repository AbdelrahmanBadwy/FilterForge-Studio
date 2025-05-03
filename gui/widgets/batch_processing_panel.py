import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import time
from utils.batch_processor import BatchProcessor
from filters import frequency_filters, spatial_filters, noise, segmentation

class BatchProcessingPanel(ttk.Frame):
    """Panel for batch processing of multiple images"""
    
    def __init__(self, parent, app):
        """Initialize panel
        
        Args:
            parent: parent widget
            app: main application instance
        """
        super().__init__(parent, padding=10)
        self.app = app
        self.input_dir = ""
        self.output_dir = ""
        self.processor = None
        self.processing_thread = None
        self.is_processing = False
        self.create_widgets()
    
    def create_widgets(self):
        """Create panel widgets"""
        # Directory selection
        dir_frame = ttk.LabelFrame(self, text="Directories", padding=5)
        dir_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(dir_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_dir_var = tk.StringVar()
        input_entry = ttk.Entry(dir_frame, textvariable=self.input_dir_var, width=40)
        input_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        input_button = ttk.Button(dir_frame, text="Browse...", command=self.browse_input_dir)
        input_button.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(dir_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        output_entry = ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=40)
        output_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        output_button = ttk.Button(dir_frame, text="Browse...", command=self.browse_output_dir)
        output_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Processing options
        options_frame = ttk.LabelFrame(self, text="Processing Options", padding=5)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Filter type selection
        ttk.Label(options_frame, text="Filter Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.filter_type = tk.StringVar(value="gaussian_low")
        filter_combo = ttk.Combobox(options_frame, textvariable=self.filter_type, width=20)
        filter_combo['values'] = (
            # Frequency domain filters
            "ideal_low", "butterworth_low", "gaussian_low",
            "ideal_high", "butterworth_high", "gaussian_high",
            # Spatial filters
            "laplacian", "high_boost", "sobel", "canny",
            "median", "bilateral",
            # Noise operations
            "add_gaussian_noise", "add_salt_pepper_noise", "add_speckle_noise", "add_poisson_noise",
            "denoise_median", "denoise_bilateral", "denoise_gaussian", "denoise_nlm",
            # Segmentation
            "threshold", "adaptive_threshold", "otsu", "watershed",
            # Feature detection
            "detect_corners", "detect_blobs", "detect_edges"
        )
        filter_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Parameters frame (will update based on selected filter)
        self.params_frame = ttk.LabelFrame(options_frame, text="Filter Parameters", padding=5)
        self.params_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        # Number of worker threads
        ttk.Label(options_frame, text="Worker Threads:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.num_workers_var = tk.IntVar(value=4)
        workers_spinner = ttk.Spinbox(options_frame, from_=1, to=16, textvariable=self.num_workers_var, width=5)
        workers_spinner.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Output format
        ttk.Label(options_frame, text="Output Format:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_format = tk.StringVar(value="keep")
        format_combo = ttk.Combobox(options_frame, textvariable=self.output_format, width=10)
        format_combo['values'] = ("keep", "jpg", "png", "tiff")
        format_combo.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # JPEG quality (only used for JPEG output)
        ttk.Label(options_frame, text="JPEG Quality:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.jpeg_quality = tk.IntVar(value=95)
        quality_scale = ttk.Scale(options_frame, from_=1, to=100, variable=self.jpeg_quality,
                                 orient=tk.HORIZONTAL, length=200)
        quality_scale.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        quality_label = ttk.Label(options_frame, textvariable=self.jpeg_quality)
        quality_label.grid(row=4, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Add trace to filter type to update parameter widgets
        self.filter_type.trace_add("write", self.update_param_widgets)
        
        # Create initial parameter widgets
        self.update_param_widgets()
        
        # Status and progress
        status_frame = ttk.LabelFrame(self, text="Status", padding=5)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Results frame for displaying statistics
        self.results_frame = ttk.LabelFrame(self, text="Results", padding=5)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollable text widget for results
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD, width=50, height=10)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        results_scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=results_scrollbar.set)
        self.results_text.config(state=tk.DISABLED)
    
    def browse_input_dir(self):
        """Browse for input directory"""
        input_dir = filedialog.askdirectory(
            initialdir=self.app.current_dir if hasattr(self.app, 'current_dir') else None,
            title="Select Input Directory"
        )
        
        if input_dir:
            self.input_dir = input_dir
            self.input_dir_var.set(input_dir)
            
            # Set default output directory
            if not self.output_dir:
                self.output_dir = input_dir + "_processed"
                self.output_dir_var.set(self.output_dir)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        output_dir = filedialog.askdirectory(
            initialdir=self.app.current_dir if hasattr(self.app, 'current_dir') else None,
            title="Select Output Directory"
        )
        
        if output_dir:
            self.output_dir = output_dir
            self.output_dir_var.set(output_dir)
    
    def update_param_widgets(self, *args):
        """Update parameter widgets based on selected filter type"""
        # Clear current parameter widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        filter_type = self.filter_type.get()
        
        # Create appropriate parameter widgets based on filter type
        if filter_type in ["ideal_low", "butterworth_low", "gaussian_low",
                         "ideal_high", "butterworth_high", "gaussian_high"]:
            # Frequency domain filter parameters
            ttk.Label(self.params_frame, text="Cutoff Frequency:").grid(row=0, column=0, sticky=tk.W, pady=2)
            self.cutoff_var = tk.IntVar(value=50)
            cutoff_scale = ttk.Scale(self.params_frame, from_=1, to=150, variable=self.cutoff_var,
                                    orient=tk.HORIZONTAL, length=200)
            cutoff_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            cutoff_label = ttk.Label(self.params_frame, textvariable=self.cutoff_var)
            cutoff_label.grid(row=0, column=2, padx=5, pady=2)
            
            if "butterworth" in filter_type:
                ttk.Label(self.params_frame, text="Order:").grid(row=1, column=0, sticky=tk.W, pady=2)
                self.order_var = tk.IntVar(value=2)
                order_scale = ttk.Scale(self.params_frame, from_=1, to=10, variable=self.order_var,
                                      orient=tk.HORIZONTAL, length=200)
                order_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
                order_label = ttk.Label(self.params_frame, textvariable=self.order_var)
                order_label.grid(row=1, column=2, padx=5, pady=2)
            
            if "high" in filter_type:
                ttk.Label(self.params_frame, text="Boost Factor:").grid(row=2, column=0, sticky=tk.W, pady=2)
                self.boost_var = tk.DoubleVar(value=1.5)
                boost_scale = ttk.Scale(self.params_frame, from_=1.0, to=5.0, variable=self.boost_var,
                                      orient=tk.HORIZONTAL, length=200)
                boost_scale.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
                
                # Use StringVar to format the float value
                self.boost_str_var = tk.StringVar()
                self.boost_var.trace_add("write", lambda *args: self.boost_str_var.set(f"{self.boost_var.get():.1f}"))
                self.boost_str_var.set(f"{self.boost_var.get():.1f}")
                
                boost_label = ttk.Label(self.params_frame, textvariable=self.boost_str_var)
                boost_label.grid(row=2, column=2, padx=5, pady=2)
                
        elif filter_type == "high_boost":
            ttk.Label(self.params_frame, text="Boost Factor:").grid(row=0, column=0, sticky=tk.W, pady=2)
            self.boost_var = tk.DoubleVar(value=1.5)
            boost_scale = ttk.Scale(self.params_frame, from_=1.0, to=5.0, variable=self.boost_var,
                                  orient=tk.HORIZONTAL, length=200)
            boost_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            # Use StringVar to format the float value
            self.boost_str_var = tk.StringVar()
            self.boost_var.trace_add("write", lambda *args: self.boost_str_var.set(f"{self.boost_var.get():.1f}"))
            self.boost_str_var.set(f"{self.boost_var.get():.1f}")
            
            boost_label = ttk.Label(self.params_frame, textvariable=self.boost_str_var)
            boost_label.grid(row=0, column=2, padx=5, pady=2)
            
        elif filter_type == "median":
            ttk.Label(self.params_frame, text="Kernel Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
            self.ksize_var = tk.IntVar(value=5)
            ksize_values = [3, 5, 7, 9, 11, 13, 15]
            ksize_combo = ttk.Combobox(self.params_frame, textvariable=self.ksize_var,
                                     values=ksize_values, width=5, state="readonly")
            ksize_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif filter_type == "bilateral":
            ttk.Label(self.params_frame, text="Diameter:").grid(row=0, column=0, sticky=tk.W, pady=2)
            self.d_var = tk.IntVar(value=9)
            d_scale = ttk.Scale(self.params_frame, from_=5, to=15, variable=self.d_var,
                              orient=tk.HORIZONTAL, length=200)
            d_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            d_label = ttk.Label(self.params_frame, textvariable=self.d_var)
            d_label.grid(row=0, column=2, padx=5, pady=2)
            
            ttk.Label(self.params_frame, text="Sigma Color:").grid(row=1, column=0, sticky=tk.W, pady=2)
            self.sigma_color_var = tk.IntVar(value=75)
            sigma_color_scale = ttk.Scale(self.params_frame, from_=10, to=150, variable=self.sigma_color_var,
                                        orient=tk.HORIZONTAL, length=200)
            sigma_color_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
            sigma_color_label = ttk.Label(self.params_frame, textvariable=self.sigma_color_var)
            sigma_color_label.grid(row=1, column=2, padx=5, pady=2)
            
            ttk.Label(self.params_frame, text="Sigma Space:").grid(row=2, column=0, sticky=tk.W, pady=2)
            self.sigma_space_var = tk.IntVar(value=75)
            sigma_space_scale = ttk.Scale(self.params_frame, from_=10, to=150, variable=self.sigma_space_var,
                                        orient=tk.HORIZONTAL, length=200)
            sigma_space_scale.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
            sigma_space_label = ttk.Label(self.params_frame, textvariable=self.sigma_space_var)
            sigma_space_label.grid(row=2, column=2, padx=5, pady=2)
            
        # Continue with more filter types as needed
        # This would be a large method with handlers for all filter types
        # For brevity, only a few filter types are implemented here
    
    def get_process_function(self):
        """Get the appropriate processing function based on selected filter type
        
        Returns:
            function to use for processing
            args and kwargs for the function
        """
        filter_type = self.filter_type.get()
        args = []
        kwargs = {}
        
        # Frequency domain filters
        if filter_type in ["ideal_low", "butterworth_low", "gaussian_low",
                         "ideal_high", "butterworth_high", "gaussian_high"]:
            
            func = lambda img: frequency_filters.create_frequency_filters(
                img, filter_type, self.cutoff_var.get(), 
                order=self.order_var.get() if "butterworth" in filter_type else 2,
                boost_factor=self.boost_var.get() if "high" in filter_type else None
            )[0]
            
        # Spatial filters
        elif filter_type == "high_boost":
            func = lambda img: spatial_filters.high_boost_filter(img, self.boost_var.get())
            
        elif filter_type == "laplacian":
            func = spatial_filters.laplacian_filter
            
        elif filter_type == "sobel":
            func = lambda img: spatial_filters.sobel_filter(img, 'both')
            
        elif filter_type == "canny":
            func = spatial_filters.canny_edge_detector
            
        elif filter_type == "median":
            func = lambda img: spatial_filters.median_filter(img, self.ksize_var.get())
            
        elif filter_type == "bilateral":
            func = lambda img: spatial_filters.bilateral_filter(
                img, self.d_var.get(), self.sigma_color_var.get(), self.sigma_space_var.get())
        
        # Add more filter types as needed
        else:
            # Default to identity function
            func = lambda img: img
            
        return func, args, kwargs
    
    def start_processing(self):
        """Start batch processing"""
        # Check if input directory is set
        if not self.input_dir or not os.path.isdir(self.input_dir):
            messagebox.showerror("Error", "Please select a valid input directory")
            return
        
        # Check if output directory is set
        if not self.output_dir:
            self.output_dir = self.input_dir + "_processed"
            self.output_dir_var.set(self.output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get processing function
        process_func, args, kwargs = self.get_process_function()
        
        # Determine output format
        file_format = None if self.output_format.get() == "keep" else self.output_format.get()
        
        # Update UI
        self.status_var.set("Initializing...")
        self.progress_var.set(0)
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Clear results text
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Create batch processor
        self.processor = BatchProcessor(self.input_dir, self.output_dir, self.num_workers_var.get())
        
        # Start processing in a separate thread
        self.is_processing = True
        
        def process():
            try:
                # Get list of image files
                num_files = len(self.processor.image_files)
                
                if num_files == 0:
                    # No image files found
                    self.root.after(0, lambda: self.update_status("No image files found in the input directory"))
                    self.root.after(0, self.processing_complete)
                    return
                
                # Add processing info to results
                self.root.after(0, lambda: self.add_result_text(
                    f"Starting batch processing with {num_files} files\n"
                    f"Input directory: {self.input_dir}\n"
                    f"Output directory: {self.output_dir}\n"
                    f"Filter: {self.filter_type.get()}\n"
                    f"Workers: {self.num_workers_var.get()}\n"
                    f"Output format: {self.output_format.get()}\n\n"
                ))
                
                # Start processing
                self.root.after(0, lambda: self.update_status("Processing..."))
                
                start_time = time.time()
                results = self.processor.process_batch(process_func, args=args, kwargs=kwargs, file_format=file_format)
                end_time = time.time()
                
                if self.is_processing:  # Check if we haven't been stopped
                    # Get processing stats
                    stats = self.processor.get_processing_stats(results)
                    
                    # Add stats to results
                    self.root.after(0, lambda: self.add_result_text(
                        f"\nBatch Processing Complete\n"
                        f"Total time: {end_time - start_time:.2f} seconds\n"
                        f"Processed {len(results)} files\n"
                        f"Average processing time: {stats.get('avg_time', 0):.2f} seconds per image\n"
                    ))
                    
                    # Update UI
                    self.root.after(0, lambda: self.update_status("Processing complete"))
                    self.root.after(0, lambda: self.progress_var.set(100))
                    
                # Processing complete
                self.root.after(0, self.processing_complete)
            
            except Exception as e:
                # Handle error
                self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
                self.root.after(0, self.processing_complete)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=process)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start progress update timer
        self.update_progress()
    
    def stop_processing(self):
        """Stop batch processing"""
        if self.is_processing:
            self.is_processing = False
            self.update_status("Stopping...")
            
            # Add to results
            self.add_result_text("\nProcessing stopped by user\n")
            
            # Update UI buttons
            self.stop_button.config(state=tk.DISABLED)
    
    def processing_complete(self):
        """Clean up after processing is complete"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def update_status(self, status):
        """Update status text
        
        Args:
            status: status message to display
        """
        self.status_var.set(status)
    
    def add_result_text(self, text):
        """Add text to results text widget
        
        Args:
            text: text to add
        """
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def update_progress(self):
        """Update progress bar based on processing status"""
        if self.is_processing and self.processor:
            # Calculate progress based on number of processed files
            total_files = len(self.processor.image_files)
            if total_files > 0:
                # Count number of files in output directory that match input filenames
                output_files = os.listdir(self.output_dir)
                processed_count = 0
                
                for filename in self.processor.image_files:
                    base_name = os.path.basename(filename)
                    
                    # Check for different possible output formats
                    name_without_ext = os.path.splitext(base_name)[0]
                    possible_names = [base_name]
                    
                    # Add possibilities for different formats
                    if self.output_format.get() != "keep":
                        possible_names.append(f"{name_without_ext}.{self.output_format.get()}")
                    
                    # Check if any of the possible names exists in output directory
                    if any(name in output_files for name in possible_names):
                        processed_count += 1
                
                # Update progress
                progress = (processed_count / total_files) * 100
                self.progress_var.set(progress)
            
            # Schedule next update
            self.after(500, self.update_progress) 