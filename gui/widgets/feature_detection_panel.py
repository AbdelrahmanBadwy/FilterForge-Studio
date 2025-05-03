import tkinter as tk
from tkinter import ttk
import numpy as np
from filters import segmentation

class FeatureDetectionPanel(ttk.Frame):
    """Panel for feature detection"""
    
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
        # Feature detection method selection
        method_frame = ttk.LabelFrame(self, text="Detection Method", padding=5)
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.detection_method = tk.StringVar(value="corners")
        
        ttk.Radiobutton(method_frame, text="Corner Detection", variable=self.detection_method, 
                       value="corners").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(method_frame, text="Blob Detection", variable=self.detection_method,
                       value="blobs").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(method_frame, text="Edge Detection", variable=self.detection_method,
                       value="edges").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Parameter frames
        
        # Corner detection parameters
        self.corner_frame = ttk.LabelFrame(self, text="Corner Detection Parameters", padding=5)
        
        ttk.Label(self.corner_frame, text="Max Corners:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_corners_var = tk.IntVar(value=50)
        max_corners_scale = ttk.Scale(self.corner_frame, from_=10, to=200, 
                                     variable=self.max_corners_var,
                                     orient=tk.HORIZONTAL, length=200)
        max_corners_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        max_corners_label = ttk.Label(self.corner_frame, textvariable=self.max_corners_var)
        max_corners_label.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(self.corner_frame, text="Quality Level:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.quality_level_var = tk.DoubleVar(value=0.01)
        quality_level_scale = ttk.Scale(self.corner_frame, from_=0.001, to=0.1, 
                                      variable=self.quality_level_var,
                                      orient=tk.HORIZONTAL, length=200,
                                      command=lambda x: self.update_quality_label())
        quality_level_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.quality_level_label = ttk.Label(self.corner_frame, text="0.01")
        self.quality_level_label.grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(self.corner_frame, text="Min Distance:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.min_distance_var = tk.IntVar(value=10)
        min_distance_scale = ttk.Scale(self.corner_frame, from_=1, to=50, 
                                      variable=self.min_distance_var,
                                      orient=tk.HORIZONTAL, length=200)
        min_distance_scale.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        min_distance_label = ttk.Label(self.corner_frame, textvariable=self.min_distance_var)
        min_distance_label.grid(row=2, column=2, padx=5, pady=2)
        
        # Blob detection parameters
        self.blob_frame = ttk.LabelFrame(self, text="Blob Detection Parameters", padding=5)
        
        ttk.Label(self.blob_frame, text="Min Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.min_threshold_var = tk.IntVar(value=10)
        min_threshold_scale = ttk.Scale(self.blob_frame, from_=0, to=100, 
                                       variable=self.min_threshold_var,
                                       orient=tk.HORIZONTAL, length=200)
        min_threshold_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        min_threshold_label = ttk.Label(self.blob_frame, textvariable=self.min_threshold_var)
        min_threshold_label.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(self.blob_frame, text="Max Threshold:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.max_threshold_var = tk.IntVar(value=200)
        max_threshold_scale = ttk.Scale(self.blob_frame, from_=100, to=255, 
                                       variable=self.max_threshold_var,
                                       orient=tk.HORIZONTAL, length=200)
        max_threshold_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        max_threshold_label = ttk.Label(self.blob_frame, textvariable=self.max_threshold_var)
        max_threshold_label.grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(self.blob_frame, text="Min Area:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.min_area_var = tk.IntVar(value=100)
        min_area_scale = ttk.Scale(self.blob_frame, from_=10, to=500, 
                                  variable=self.min_area_var,
                                  orient=tk.HORIZONTAL, length=200)
        min_area_scale.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        min_area_label = ttk.Label(self.blob_frame, textvariable=self.min_area_var)
        min_area_label.grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(self.blob_frame, text="Max Area:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.max_area_var = tk.IntVar(value=5000)
        max_area_scale = ttk.Scale(self.blob_frame, from_=1000, to=10000, 
                                  variable=self.max_area_var,
                                  orient=tk.HORIZONTAL, length=200)
        max_area_scale.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)
        max_area_label = ttk.Label(self.blob_frame, textvariable=self.max_area_var)
        max_area_label.grid(row=3, column=2, padx=5, pady=2)
        
        # Edge detection parameters
        self.edge_frame = ttk.LabelFrame(self, text="Edge Detection Parameters", padding=5)
        
        ttk.Label(self.edge_frame, text="Low Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.edge_threshold1_var = tk.IntVar(value=100)
        edge_threshold1_scale = ttk.Scale(self.edge_frame, from_=0, to=255, 
                                         variable=self.edge_threshold1_var,
                                         orient=tk.HORIZONTAL, length=200)
        edge_threshold1_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        edge_threshold1_label = ttk.Label(self.edge_frame, textvariable=self.edge_threshold1_var)
        edge_threshold1_label.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(self.edge_frame, text="High Threshold:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.edge_threshold2_var = tk.IntVar(value=200)
        edge_threshold2_scale = ttk.Scale(self.edge_frame, from_=0, to=255, 
                                         variable=self.edge_threshold2_var,
                                         orient=tk.HORIZONTAL, length=200)
        edge_threshold2_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        edge_threshold2_label = ttk.Label(self.edge_frame, textvariable=self.edge_threshold2_var)
        edge_threshold2_label.grid(row=1, column=2, padx=5, pady=2)
        
        # Show initial frame
        self.corner_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add trace to method to update visible parameter frame
        self.detection_method.trace_add("write", self.update_params_frame)
        
        # Apply button
        self.apply_button = ttk.Button(self, text="Detect Features", command=self.apply_detection)
        self.apply_button.pack(pady=10)
        
        # Real-time preview checkbox (connected to app's variable)
        if hasattr(self.app, 'real_time_preview'):
            ttk.Checkbutton(self, text="Real-time Preview", 
                           variable=self.app.real_time_preview).pack(pady=5)
            
            # Add trace to parameters for real-time preview
            self.max_corners_var.trace_add("write", self.on_param_change)
            self.quality_level_var.trace_add("write", self.on_param_change)
            self.min_distance_var.trace_add("write", self.on_param_change)
            self.min_threshold_var.trace_add("write", self.on_param_change)
            self.max_threshold_var.trace_add("write", self.on_param_change)
            self.min_area_var.trace_add("write", self.on_param_change)
            self.max_area_var.trace_add("write", self.on_param_change)
            self.edge_threshold1_var.trace_add("write", self.on_param_change)
            self.edge_threshold2_var.trace_add("write", self.on_param_change)
            self.detection_method.trace_add("write", self.on_param_change)
    
    def update_quality_label(self):
        """Update quality level label with formatted value"""
        self.quality_level_label.config(text=f"{self.quality_level_var.get():.3f}")
    
    def update_params_frame(self, *args):
        """Update visible parameters frame based on selected detection method"""
        # Hide all frames
        self.corner_frame.pack_forget()
        self.blob_frame.pack_forget()
        self.edge_frame.pack_forget()
        
        # Show selected frame
        if self.detection_method.get() == "corners":
            self.corner_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.detection_method.get() == "blobs":
            self.blob_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.detection_method.get() == "edges":
            self.edge_frame.pack(fill=tk.X, padx=5, pady=5)
        
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
                self.apply_detection()
    
    def apply_detection(self):
        """Apply the selected feature detection method"""
        detection_method = self.detection_method.get()
        
        if detection_method == "corners":
            max_corners = self.max_corners_var.get()
            quality_level = self.quality_level_var.get()
            min_distance = self.min_distance_var.get()
            
            # Detect corners
            corners = self.app.apply_filter(segmentation.detect_corners, max_corners, quality_level, min_distance)
            
            # Create visualization (usually handled in the filter function itself or in the app)
            
        elif detection_method == "blobs":
            min_threshold = self.min_threshold_var.get()
            max_threshold = self.max_threshold_var.get()
            min_area = self.min_area_var.get()
            max_area = self.max_area_var.get()
            
            # Detect blobs
            keypoints = self.app.apply_filter(segmentation.detect_blobs, min_threshold, max_threshold, min_area, max_area)
            
            # Create visualization (usually handled in the filter function itself or in the app)
            
        elif detection_method == "edges":
            threshold1 = self.edge_threshold1_var.get()
            threshold2 = self.edge_threshold2_var.get()
            
            # Detect edges
            edges = self.app.apply_filter(segmentation.detect_edges, threshold1, threshold2) 