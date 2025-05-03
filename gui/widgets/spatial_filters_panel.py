import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from filters import spatial_filters
from utils.image_utils import load_kernel_from_csv, create_kernel_preview

class SpatialFiltersPanel(ttk.Frame):
    """Panel for spatial domain filters"""
    
    def __init__(self, parent, app):
        """Initialize panel
        
        Args:
            parent: parent widget
            app: main application instance
        """
        super().__init__(parent, padding=10)
        self.app = app
        self.custom_kernel = None
        self.create_widgets()
    
    def create_widgets(self):
        """Create panel widgets"""
        # Filter type selection
        filter_type_frame = ttk.LabelFrame(self, text="Filter Type", padding=5)
        filter_type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.filter_type = tk.StringVar(value="high_boost")
        
        # Common filters
        ttk.Label(filter_type_frame, text="Common Filters:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(filter_type_frame, text="High Boost", variable=self.filter_type, 
                       value="high_boost").grid(row=1, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Laplacian", variable=self.filter_type, 
                       value="laplacian").grid(row=2, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Sobel", variable=self.filter_type, 
                       value="sobel").grid(row=3, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Canny Edge", variable=self.filter_type, 
                       value="canny").grid(row=4, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Median", variable=self.filter_type, 
                       value="median").grid(row=5, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Bilateral", variable=self.filter_type, 
                       value="bilateral").grid(row=6, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Custom", variable=self.filter_type, 
                       value="custom").grid(row=7, column=0, sticky=tk.W, padx=15)
        
        # Filter parameters
        params_frame = ttk.LabelFrame(self, text="Parameters", padding=5)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # High boost factor
        ttk.Label(params_frame, text="Boost Factor:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.boost_var = tk.DoubleVar(value=1.5)
        boost_scale = ttk.Scale(params_frame, from_=1.0, to=10.0, variable=self.boost_var,
                               orient=tk.HORIZONTAL, length=200, 
                               command=lambda x: self.update_boost_label())
        boost_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        self.boost_label = ttk.Label(params_frame, text="1.5")
        self.boost_label.grid(row=0, column=2, padx=5, pady=2)
        
        # Median kernel size
        ttk.Label(params_frame, text="Median Kernel Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.median_size_var = tk.IntVar(value=5)
        median_size_values = [3, 5, 7, 9, 11, 13, 15]
        median_size_combo = ttk.Combobox(params_frame, textvariable=self.median_size_var, 
                                        values=median_size_values, width=5, state="readonly")
        median_size_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Bilateral filter parameters
        ttk.Label(params_frame, text="Bilateral Diameter:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.bilateral_d_var = tk.IntVar(value=9)
        bilateral_d_scale = ttk.Scale(params_frame, from_=5, to=15, variable=self.bilateral_d_var,
                                     orient=tk.HORIZONTAL, length=200)
        bilateral_d_scale.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        bilateral_d_label = ttk.Label(params_frame, textvariable=self.bilateral_d_var, width=3)
        bilateral_d_label.grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Bilateral Color Sigma:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.bilateral_sigma_color_var = tk.IntVar(value=75)
        bilateral_sigma_color_scale = ttk.Scale(params_frame, from_=10, to=150, 
                                              variable=self.bilateral_sigma_color_var,
                                              orient=tk.HORIZONTAL, length=200)
        bilateral_sigma_color_scale.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)
        bilateral_sigma_color_label = ttk.Label(params_frame, 
                                               textvariable=self.bilateral_sigma_color_var, width=3)
        bilateral_sigma_color_label.grid(row=3, column=2, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Bilateral Space Sigma:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.bilateral_sigma_space_var = tk.IntVar(value=75)
        bilateral_sigma_space_scale = ttk.Scale(params_frame, from_=10, to=150, 
                                              variable=self.bilateral_sigma_space_var,
                                              orient=tk.HORIZONTAL, length=200)
        bilateral_sigma_space_scale.grid(row=4, column=1, padx=5, pady=2, sticky=tk.W)
        bilateral_sigma_space_label = ttk.Label(params_frame, 
                                              textvariable=self.bilateral_sigma_space_var, width=3)
        bilateral_sigma_space_label.grid(row=4, column=2, padx=5, pady=2)
        
        # Canny edge parameters
        ttk.Label(params_frame, text="Canny Low Threshold:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.canny_threshold1_var = tk.IntVar(value=100)
        canny_threshold1_scale = ttk.Scale(params_frame, from_=0, to=255, 
                                           variable=self.canny_threshold1_var,
                                           orient=tk.HORIZONTAL, length=200)
        canny_threshold1_scale.grid(row=5, column=1, padx=5, pady=2, sticky=tk.W)
        canny_threshold1_label = ttk.Label(params_frame, 
                                          textvariable=self.canny_threshold1_var, width=3)
        canny_threshold1_label.grid(row=5, column=2, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Canny High Threshold:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.canny_threshold2_var = tk.IntVar(value=200)
        canny_threshold2_scale = ttk.Scale(params_frame, from_=0, to=255, 
                                          variable=self.canny_threshold2_var,
                                          orient=tk.HORIZONTAL, length=200)
        canny_threshold2_scale.grid(row=6, column=1, padx=5, pady=2, sticky=tk.W)
        canny_threshold2_label = ttk.Label(params_frame, 
                                         textvariable=self.canny_threshold2_var, width=3)
        canny_threshold2_label.grid(row=6, column=2, padx=5, pady=2)
        
        # Sobel direction
        ttk.Label(params_frame, text="Sobel Direction:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.sobel_direction_var = tk.StringVar(value="both")
        sobel_directions = ["both", "x", "y"]
        sobel_direction_combo = ttk.Combobox(params_frame, textvariable=self.sobel_direction_var, 
                                            values=sobel_directions, width=5, state="readonly")
        sobel_direction_combo.grid(row=7, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Custom kernel frame
        custom_frame = ttk.LabelFrame(self, text="Custom Kernel", padding=5)
        custom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Predefined kernels
        ttk.Label(custom_frame, text="Predefined Kernels:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.kernel_type_var = tk.StringVar(value="sharpen")
        kernel_types = list(spatial_filters.KERNELS.keys())
        kernel_type_combo = ttk.Combobox(custom_frame, textvariable=self.kernel_type_var, 
                                        values=kernel_types, width=15, state="readonly")
        kernel_type_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Load custom kernel button
        load_kernel_button = ttk.Button(custom_frame, text="Load from CSV", 
                                       command=self.load_custom_kernel)
        load_kernel_button.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Preview kernel button
        preview_kernel_button = ttk.Button(custom_frame, text="Preview Kernel", 
                                         command=self.preview_kernel)
        preview_kernel_button.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Apply button
        self.apply_button = ttk.Button(self, text="Apply Filter", command=self.apply_filter)
        self.apply_button.pack(pady=10)
        
        # Real-time preview checkbox (connected to app's variable)
        if hasattr(self.app, 'real_time_preview'):
            ttk.Checkbutton(self, text="Real-time Preview", 
                           variable=self.app.real_time_preview).pack(pady=5)
            
            # Add trace to parameters for real-time preview
            self.boost_var.trace_add("write", self.on_param_change)
            self.median_size_var.trace_add("write", self.on_param_change)
            self.bilateral_d_var.trace_add("write", self.on_param_change)
            self.bilateral_sigma_color_var.trace_add("write", self.on_param_change)
            self.bilateral_sigma_space_var.trace_add("write", self.on_param_change)
            self.canny_threshold1_var.trace_add("write", self.on_param_change)
            self.canny_threshold2_var.trace_add("write", self.on_param_change)
            self.sobel_direction_var.trace_add("write", self.on_param_change)
            self.filter_type.trace_add("write", self.on_param_change)
            self.kernel_type_var.trace_add("write", self.on_param_change)
    
    def update_boost_label(self):
        """Update boost factor label with formatted value"""
        self.boost_label.config(text=f"{self.boost_var.get():.1f}")
    
    def on_param_change(self, *args):
        """Handle parameter change for real-time preview"""
        if hasattr(self.app, 'real_time_preview') and self.app.real_time_preview.get():
            # Check if we're already processing
            if not hasattr(self.app, 'is_processing') or not self.app.is_processing:
                self.apply_filter()
    
    def load_custom_kernel(self):
        """Load custom kernel from CSV file"""
        file_path = filedialog.askopenfilename(
            initialdir=self.app.current_dir if hasattr(self.app, 'current_dir') else None,
            title="Select Kernel CSV File",
            filetypes=(
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            try:
                self.custom_kernel = load_kernel_from_csv(file_path)
                messagebox.showinfo("Kernel Loaded", 
                                  f"Loaded kernel of shape {self.custom_kernel.shape}")
                
                # Auto-select custom filter
                self.filter_type.set("custom")
                
                # Trigger preview
                self.preview_kernel()
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load kernel: {str(e)}")
    
    def preview_kernel(self):
        """Show preview of current kernel"""
        kernel = None
        
        if self.filter_type.get() == "custom" and self.custom_kernel is not None:
            kernel = self.custom_kernel
        else:
            kernel_type = self.kernel_type_var.get()
            if kernel_type in spatial_filters.KERNELS:
                kernel = spatial_filters.KERNELS[kernel_type]
        
        if kernel is not None:
            # Create preview image
            preview_img = create_kernel_preview(kernel)
            
            # Create preview window
            preview_window = tk.Toplevel(self)
            preview_window.title("Kernel Preview")
            preview_window.resizable(False, False)
            
            # Convert to PhotoImage for display
            from PIL import Image, ImageTk
            pil_img = Image.fromarray(preview_img)
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # Create canvas to display image
            canvas = tk.Canvas(preview_window, width=tk_img.width(), height=tk_img.height())
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            canvas.pack()
            
            # Keep reference to image to prevent garbage collection
            canvas.image = tk_img
            
            # Add kernel info
            info_frame = ttk.Frame(preview_window, padding=10)
            info_frame.pack(fill=tk.X)
            
            ttk.Label(info_frame, text=f"Shape: {kernel.shape}").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Min: {kernel.min():.2f}").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Max: {kernel.max():.2f}").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Sum: {kernel.sum():.2f}").pack(anchor=tk.W)
        else:
            messagebox.showwarning("Warning", "No valid kernel selected")
    
    def apply_filter(self):
        """Apply the selected spatial filter"""
        # Get filter type
        filter_type = self.filter_type.get()
        
        # Apply filter based on type
        if filter_type == "high_boost":
            boost = self.boost_var.get()
            self.app.apply_filter(spatial_filters.high_boost_filter, boost)
            
        elif filter_type == "laplacian":
            self.app.apply_filter(spatial_filters.laplacian_filter)
            
        elif filter_type == "sobel":
            direction = self.sobel_direction_var.get()
            self.app.apply_filter(spatial_filters.sobel_filter, direction)
            
        elif filter_type == "canny":
            threshold1 = self.canny_threshold1_var.get()
            threshold2 = self.canny_threshold2_var.get()
            self.app.apply_filter(spatial_filters.canny_edge_detector, threshold1, threshold2)
            
        elif filter_type == "median":
            ksize = self.median_size_var.get()
            self.app.apply_filter(spatial_filters.median_filter, ksize)
            
        elif filter_type == "bilateral":
            d = self.bilateral_d_var.get()
            sigma_color = self.bilateral_sigma_color_var.get()
            sigma_space = self.bilateral_sigma_space_var.get()
            self.app.apply_filter(spatial_filters.bilateral_filter, d, sigma_color, sigma_space)
            
        elif filter_type == "custom":
            # Determine which kernel to use
            if self.custom_kernel is not None:
                kernel = self.custom_kernel
            else:
                kernel_type = self.kernel_type_var.get()
                kernel = spatial_filters.KERNELS[kernel_type]
                
            self.app.apply_filter(spatial_filters.apply_custom_kernel, kernel) 