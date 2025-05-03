import tkinter as tk
from tkinter import ttk
import numpy as np
from filters import frequency_filters

class FrequencyFiltersPanel(ttk.Frame):
    """Panel for frequency domain filters"""
    
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
        # Filter type selection
        filter_type_frame = ttk.LabelFrame(self, text="Filter Type", padding=5)
        filter_type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.filter_type = tk.StringVar(value="butterworth_low")
        
        # Low-pass filters
        ttk.Label(filter_type_frame, text="Low-pass Filters:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(filter_type_frame, text="Ideal", variable=self.filter_type, 
                       value="ideal_low").grid(row=1, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Butterworth", variable=self.filter_type, 
                       value="butterworth_low").grid(row=2, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Gaussian", variable=self.filter_type, 
                       value="gaussian_low").grid(row=3, column=0, sticky=tk.W, padx=15)
        
        # High-pass filters
        ttk.Label(filter_type_frame, text="High-pass Filters:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(filter_type_frame, text="Ideal", variable=self.filter_type, 
                       value="ideal_high").grid(row=5, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Butterworth", variable=self.filter_type, 
                       value="butterworth_high").grid(row=6, column=0, sticky=tk.W, padx=15)
        ttk.Radiobutton(filter_type_frame, text="Gaussian", variable=self.filter_type, 
                       value="gaussian_high").grid(row=7, column=0, sticky=tk.W, padx=15)
        
        # Filter parameters
        params_frame = ttk.LabelFrame(self, text="Parameters", padding=5)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Cutoff frequency
        ttk.Label(params_frame, text="Cutoff Frequency:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.cutoff_var = tk.IntVar(value=50)
        cutoff_scale = ttk.Scale(params_frame, from_=1, to=150, variable=self.cutoff_var, 
                                orient=tk.HORIZONTAL, length=200)
        cutoff_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        cutoff_label = ttk.Label(params_frame, textvariable=self.cutoff_var, width=3)
        cutoff_label.grid(row=0, column=2, padx=5, pady=2)
        
        # Order (for Butterworth filter)
        ttk.Label(params_frame, text="Order (Butterworth):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.order_var = tk.IntVar(value=2)
        order_scale = ttk.Scale(params_frame, from_=1, to=10, variable=self.order_var,
                               orient=tk.HORIZONTAL, length=200)
        order_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        order_label = ttk.Label(params_frame, textvariable=self.order_var, width=3)
        order_label.grid(row=1, column=2, padx=5, pady=2)
        
        # Boost factor (for high-pass filters)
        ttk.Label(params_frame, text="Boost Factor (High-pass):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.boost_var = tk.DoubleVar(value=1.5)
        boost_scale = ttk.Scale(params_frame, from_=1, to=5, variable=self.boost_var,
                               orient=tk.HORIZONTAL, length=200, 
                               command=lambda x: self.update_boost_label())
        boost_scale.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        self.boost_label = ttk.Label(params_frame, text="1.5")
        self.boost_label.grid(row=2, column=2, padx=5, pady=2)
        
        # Visualization options
        vis_frame = ttk.LabelFrame(self, text="Visualization", padding=5)
        vis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_spectrum_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(vis_frame, text="Show Frequency Spectrum", 
                       variable=self.show_spectrum_var).pack(anchor=tk.W)
        
        # Apply button
        self.apply_button = ttk.Button(self, text="Apply Filter", command=self.apply_filter)
        self.apply_button.pack(pady=10)
        
        # Real-time preview checkbox (connected to app's variable)
        if hasattr(self.app, 'real_time_preview'):
            ttk.Checkbutton(self, text="Real-time Preview", 
                           variable=self.app.real_time_preview).pack(pady=5)
            
            # Add trace to parameters for real-time preview
            self.cutoff_var.trace_add("write", self.on_param_change)
            self.order_var.trace_add("write", self.on_param_change)
            self.boost_var.trace_add("write", self.on_param_change)
            self.filter_type.trace_add("write", self.on_param_change)
    
    def update_boost_label(self):
        """Update boost factor label with formatted value"""
        self.boost_label.config(text=f"{self.boost_var.get():.1f}")
    
    def on_param_change(self, *args):
        """Handle parameter change for real-time preview"""
        if hasattr(self.app, 'real_time_preview') and self.app.real_time_preview.get():
            # Check if we're already processing
            if not hasattr(self.app, 'is_processing') or not self.app.is_processing:
                self.apply_filter()
    
    def apply_filter(self):
        """Apply the selected frequency domain filter"""
        # Get parameters
        filter_type = self.filter_type.get()
        cutoff = self.cutoff_var.get()
        order = self.order_var.get()
        boost = self.boost_var.get() if 'high' in filter_type else None
        
        # Apply filter
        self.app.apply_filter(
            frequency_filters.create_frequency_filters,
            filter_type, cutoff, order, boost
        )
        
        # Show spectrum if requested
        if self.show_spectrum_var.get():
            self.app.show_spectrum() 