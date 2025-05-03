import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from utils.metrics import calculate_histogram, compare_histograms

class HistogramPanel(ttk.Frame):
    """Panel for displaying image histograms"""
    
    def __init__(self, parent, app):
        """Initialize panel
        
        Args:
            parent: parent widget
            app: main application instance
        """
        super().__init__(parent, padding=10)
        self.app = app
        self.original_image = None
        self.processed_image = None
        self.create_widgets()
    
    def create_widgets(self):
        """Create panel widgets"""
        # Control frame
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Display options
        self.display_mode = tk.StringVar(value="compare")
        ttk.Radiobutton(control_frame, text="Compare Histograms", 
                       variable=self.display_mode, value="compare",
                       command=self.update_display).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(control_frame, text="Original Only", 
                       variable=self.display_mode, value="original",
                       command=self.update_display).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(control_frame, text="Processed Only", 
                       variable=self.display_mode, value="processed",
                       command=self.update_display).pack(side=tk.LEFT, padx=10)
        
        # Log scale option
        self.log_scale = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Log Scale", 
                       variable=self.log_scale,
                       command=self.update_display).pack(side=tk.RIGHT, padx=10)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(self, text="Histogram Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(self, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Original image stats
        original_frame = ttk.LabelFrame(stats_frame, text="Original Image", padding=5)
        original_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        self.original_min_var = tk.StringVar(value="Min: -")
        self.original_max_var = tk.StringVar(value="Max: -")
        self.original_mean_var = tk.StringVar(value="Mean: -")
        self.original_std_var = tk.StringVar(value="Std Dev: -")
        
        ttk.Label(original_frame, textvariable=self.original_min_var).pack(anchor=tk.W, pady=2)
        ttk.Label(original_frame, textvariable=self.original_max_var).pack(anchor=tk.W, pady=2)
        ttk.Label(original_frame, textvariable=self.original_mean_var).pack(anchor=tk.W, pady=2)
        ttk.Label(original_frame, textvariable=self.original_std_var).pack(anchor=tk.W, pady=2)
        
        # Processed image stats
        processed_frame = ttk.LabelFrame(stats_frame, text="Processed Image", padding=5)
        processed_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5, pady=5)
        
        self.processed_min_var = tk.StringVar(value="Min: -")
        self.processed_max_var = tk.StringVar(value="Max: -")
        self.processed_mean_var = tk.StringVar(value="Mean: -")
        self.processed_std_var = tk.StringVar(value="Std Dev: -")
        
        ttk.Label(processed_frame, textvariable=self.processed_min_var).pack(anchor=tk.W, pady=2)
        ttk.Label(processed_frame, textvariable=self.processed_max_var).pack(anchor=tk.W, pady=2)
        ttk.Label(processed_frame, textvariable=self.processed_mean_var).pack(anchor=tk.W, pady=2)
        ttk.Label(processed_frame, textvariable=self.processed_std_var).pack(anchor=tk.W, pady=2)
        
        # Set initial state
        self.update_display()
    
    def set_images(self, original, processed=None):
        """Set images for histogram analysis
        
        Args:
            original: original image
            processed: processed image (if None, use original)
        """
        self.original_image = original
        self.processed_image = processed if processed is not None else original
        
        # Update statistics
        self.update_statistics()
        
        # Update display
        self.update_display()
    
    def update_statistics(self):
        """Update image statistics"""
        if self.original_image is not None:
            # Original image statistics
            original_min = np.min(self.original_image)
            original_max = np.max(self.original_image)
            original_mean = np.mean(self.original_image)
            original_std = np.std(self.original_image)
            
            self.original_min_var.set(f"Min: {original_min}")
            self.original_max_var.set(f"Max: {original_max}")
            self.original_mean_var.set(f"Mean: {original_mean:.2f}")
            self.original_std_var.set(f"Std Dev: {original_std:.2f}")
            
            # Processed image statistics
            processed_min = np.min(self.processed_image)
            processed_max = np.max(self.processed_image)
            processed_mean = np.mean(self.processed_image)
            processed_std = np.std(self.processed_image)
            
            self.processed_min_var.set(f"Min: {processed_min}")
            self.processed_max_var.set(f"Max: {processed_max}")
            self.processed_mean_var.set(f"Mean: {processed_mean:.2f}")
            self.processed_std_var.set(f"Std Dev: {processed_std:.2f}")
    
    def update_display(self):
        """Update histogram display based on current settings"""
        # Clear figure
        self.fig.clear()
        
        if self.original_image is None:
            # No image to display
            self.canvas.draw()
            return
        
        display_mode = self.display_mode.get()
        use_log_scale = self.log_scale.get()
        
        if display_mode == "compare":
            # Compare histograms
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            
            self._plot_histogram(ax1, self.original_image, "Original Image", use_log_scale)
            self._plot_histogram(ax2, self.processed_image, "Processed Image", use_log_scale)
            
        elif display_mode == "original":
            # Original histogram only
            ax = self.fig.add_subplot(111)
            self._plot_histogram(ax, self.original_image, "Original Image", use_log_scale)
            
        elif display_mode == "processed":
            # Processed histogram only
            ax = self.fig.add_subplot(111)
            self._plot_histogram(ax, self.processed_image, "Processed Image", use_log_scale)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _plot_histogram(self, ax, image, title, use_log_scale=False):
        """Plot histogram on given axes
        
        Args:
            ax: matplotlib axes to plot on
            image: image to plot histogram for
            title: plot title
            use_log_scale: whether to use logarithmic scale for y-axis
        """
        if image is None:
            return
        
        # Calculate histogram
        hist = calculate_histogram(image)
        
        # Plot histogram
        ax.plot(hist)
        ax.set_title(title)
        ax.set_xlim([0, 256])
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        
        if use_log_scale:
            ax.set_yscale('log')
            # Add note about log scale
            ax.text(0.05, 0.95, "Log Scale", transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)) 