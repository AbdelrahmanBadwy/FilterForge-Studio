import tkinter as tk
from tkinter import ttk
import numpy as np
from filters import segmentation

class SegmentationPanel(ttk.Frame):
    """Panel for image segmentation"""
    
    def __init__(self, parent, app):
        """Initialize panel
        
        Args:
            parent: parent widget
            app: main application instance
        """
        super().__init__(parent, padding=10)
        self.app = app
        self.create_widgets()
    
    def create_widgets(self):
        """Create panel widgets"""
        # Segmentation method selection
        method_frame = ttk.LabelFrame(self, text="Segmentation Method", padding=5)
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.seg_method = tk.StringVar(value="threshold")
        
        ttk.Radiobutton(method_frame, text="Basic Thresholding", variable=self.seg_method, 
                        value="threshold").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(method_frame, text="Adaptive Thresholding", variable=self.seg_method,
                        value="adaptive").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(method_frame, text="Otsu's Method", variable=self.seg_method,
                        value="otsu").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(method_frame, text="Watershed", variable=self.seg_method,
                        value="watershed").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Parameter frames
        
        # Basic thresholding parameters
        self.threshold_frame = ttk.LabelFrame(self, text="Threshold Parameters", padding=5)
        
        ttk.Label(self.threshold_frame, text="Threshold Value:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.threshold_value = tk.IntVar(value=127)
        threshold_scale = ttk.Scale(self.threshold_frame, from_=0, to=255, 
                                   variable=self.threshold_value,
                                   orient=tk.HORIZONTAL, length=200)
        threshold_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        threshold_label = ttk.Label(self.threshold_frame, textvariable=self.threshold_value)
        threshold_label.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(self.threshold_frame, text="Method:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.threshold_method = tk.StringVar(value="binary")
        threshold_methods = ["binary", "binary_inv"]
        threshold_method_combo = ttk.Combobox(self.threshold_frame, textvariable=self.threshold_method, 
                                             values=threshold_methods, width=10, state="readonly")
        threshold_method_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Adaptive thresholding parameters
        self.adaptive_frame = ttk.LabelFrame(self, text="Adaptive Threshold Parameters", padding=5)
        
        ttk.Label(self.adaptive_frame, text="Block Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.adaptive_block_size = tk.IntVar(value=11)
        adaptive_block_values = [3, 5, 7, 9, 11, 13, 15, 21, 25, 31]
        adaptive_block_combo = ttk.Combobox(self.adaptive_frame, textvariable=self.adaptive_block_size, 
                                           values=adaptive_block_values, width=5, state="readonly")
        adaptive_block_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(self.adaptive_frame, text="C Value:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.adaptive_c = tk.IntVar(value=2)
        adaptive_c_scale = ttk.Scale(self.adaptive_frame, from_=-10, to=30, 
                                    variable=self.adaptive_c,
                                    orient=tk.HORIZONTAL, length=200)
        adaptive_c_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        adaptive_c_label = ttk.Label(self.adaptive_frame, textvariable=self.adaptive_c)
        adaptive_c_label.grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(self.adaptive_frame, text="Method:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.adaptive_method = tk.StringVar(value="mean")
        adaptive_methods = ["mean", "gaussian"]
        adaptive_method_combo = ttk.Combobox(self.adaptive_frame, textvariable=self.adaptive_method, 
                                            values=adaptive_methods, width=10, state="readonly")
        adaptive_method_combo.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Watershed parameters
        self.watershed_frame = ttk.LabelFrame(self, text="Watershed Parameters", padding=5)
        ttk.Label(self.watershed_frame, text="Watershed segmentation has no adjustable parameters.").pack(
            padx=5, pady=10)
        
        # Overlay option
        overlay_frame = ttk.LabelFrame(self, text="Output Options", padding=5)
        overlay_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Overlay on Original Image", 
                       variable=self.overlay_var).pack(anchor=tk.W, padx=5, pady=5)
        
        # Show initial frame
        self.threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add trace to method to update visible parameter frame
        self.seg_method.trace_add("write", self.update_params_frame)
        
        # Apply button
        self.apply_button = ttk.Button(self, text="Apply Segmentation", command=self.apply_segmentation)
        self.apply_button.pack(pady=10)
        
        # Real-time preview checkbox (connected to app's variable)
        if hasattr(self.app, 'real_time_preview'):
            ttk.Checkbutton(self, text="Real-time Preview", 
                           variable=self.app.real_time_preview).pack(pady=5)
            
            # Add trace to parameters for real-time preview
            self.threshold_value.trace_add("write", self.on_param_change)
            self.threshold_method.trace_add("write", self.on_param_change)
            self.adaptive_block_size.trace_add("write", self.on_param_change)
            self.adaptive_c.trace_add("write", self.on_param_change)
            self.adaptive_method.trace_add("write", self.on_param_change)
            self.seg_method.trace_add("write", self.on_param_change)
            self.overlay_var.trace_add("write", self.on_param_change)
    
    def update_params_frame(self, *args):
        """Update visible parameters frame based on selected segmentation method"""
        # Hide all frames
        self.threshold_frame.pack_forget()
        self.adaptive_frame.pack_forget()
        self.watershed_frame.pack_forget()
        
        # Show selected frame
        if self.seg_method.get() == "threshold":
            self.threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.seg_method.get() == "adaptive":
            self.adaptive_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.seg_method.get() == "watershed":
            self.watershed_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Trigger preview if real-time is enabled
        self.preview_if_realtime()
    
    def on_param_change(self, *args):
        """Handle parameter change for real-time preview"""
        self.preview_if_realtime()
    
    def preview_if_realtime(self):
        """Trigger preview if real-time is enabled"""
        if hasattr(self.app, 'real_time_preview') and self.app.real_time_preview.get():
            # Check if we're already processing
            if not hasattr(self.app, 'is_processing') or not self.app.is_processing:
                self.apply_segmentation()
    
    def apply_segmentation(self):
        """Apply the selected segmentation method"""
        seg_method = self.seg_method.get()
        overlay = self.overlay_var.get()
        
        if seg_method == "threshold":
            threshold = self.threshold_value.get()
            method = self.threshold_method.get()
            
            # First apply segmentation
            result = self.app.apply_filter(segmentation.threshold_segment, threshold, method)
            
            # Apply overlay if requested
            if overlay and hasattr(self.app, 'original_image') and self.app.original_image is not None:
                self.app.apply_filter(segmentation.overlay_segmentation, self.app.original_image, result)
                
        elif seg_method == "adaptive":
            block_size = self.adaptive_block_size.get()
            c = self.adaptive_c.get()
            method = self.adaptive_method.get()
            
            # Apply adaptive thresholding
            result = self.app.apply_filter(segmentation.adaptive_threshold, block_size, c, method)
            
            # Apply overlay if requested
            if overlay and hasattr(self.app, 'original_image') and self.app.original_image is not None:
                self.app.apply_filter(segmentation.overlay_segmentation, self.app.original_image, result)
                
        elif seg_method == "otsu":
            # Apply Otsu thresholding
            result = self.app.apply_filter(segmentation.threshold_segment, 0, 'otsu')
            
            # Apply overlay if requested
            if overlay and hasattr(self.app, 'original_image') and self.app.original_image is not None:
                self.app.apply_filter(segmentation.overlay_segmentation, self.app.original_image, result)
                
        elif seg_method == "watershed":
            # Apply watershed segmentation
            segmented, markers = self.app.apply_filter(segmentation.watershed_segmentation)
            
            # Apply overlay if requested
            if overlay and hasattr(self.app, 'original_image') and self.app.original_image is not None:
                self.app.apply_filter(segmentation.overlay_segmentation, self.app.original_image, segmented) 