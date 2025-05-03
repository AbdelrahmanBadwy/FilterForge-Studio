import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import threading
import time
from PIL import ImageTk

from filters import frequency_filters, spatial_filters, noise, segmentation
from utils import image_utils, metrics
from gui.widgets.frequency_filters_panel import FrequencyFiltersPanel
from gui.widgets.spatial_filters_panel import SpatialFiltersPanel
from gui.widgets.noise_panel import NoisePanel
from gui.widgets.segmentation_panel import SegmentationPanel
from gui.widgets.feature_detection_panel import FeatureDetectionPanel
from gui.widgets.custom_kernel_panel import CustomKernelPanel
from gui.widgets.batch_processing_panel import BatchProcessingPanel
from gui.widgets.histogram_panel import HistogramPanel

class FilterForgeApp:
    """Main application class for FilterForge Studio"""
    
    def __init__(self, root):
        """Initialize the application
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.geometry("1280x800")
        self.root.minsize(1000, 700)
        self.setup_variables()
        self.create_menu()
        self.create_widgets()
        self.setup_layout()
        
        # Schedule welcome message to show after UI is fully initialized
        self.root.after(100, self.show_welcome_message)
    
    def setup_variables(self):
        """Set up application variables"""
        # Image variables
        self.original_image = None
        self.current_image = None
        self.last_filter_applied = None
        self.processing_history = []
        
        # Current directory for file dialogs
        self.current_dir = os.path.expanduser("~")
        
        # Real-time preview
        self.real_time_preview = tk.BooleanVar(value=True)
        
        # Processing thread
        self.processing_thread = None
        self.is_processing = False
    
    def create_menu(self):
        """Create application menu"""
        # Main menu bar
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_command(label="Save As", command=self.save_image_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo)
        edit_menu.add_command(label="Reset to Original", command=self.reset_to_original)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_checkbutton(label="Real-time Preview", 
                                 variable=self.real_time_preview)
        view_menu.add_separator()
        view_menu.add_command(label="Histogram", command=self.show_histogram)
        view_menu.add_command(label="Frequency Spectrum", command=self.show_spectrum)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Batch Processing", command=self.show_batch_processing)
        tools_menu.add_command(label="Custom Kernel Editor", command=self.show_custom_kernel)
        tools_menu.add_separator()
        tools_menu.add_command(label="Image Metrics", command=self.show_metrics)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_widgets(self):
        """Create application widgets"""
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        
        # Create a notebook for filter panels
        self.filter_notebook = ttk.Notebook(self.main_frame)
        
        # Create filter panels
        self.frequency_panel = FrequencyFiltersPanel(self.filter_notebook, self)
        self.spatial_panel = SpatialFiltersPanel(self.filter_notebook, self)
        self.noise_panel = NoisePanel(self.filter_notebook, self)
        self.segmentation_panel = SegmentationPanel(self.filter_notebook, self)
        self.feature_panel = FeatureDetectionPanel(self.filter_notebook, self)
        
        # Add panels to notebook
        self.filter_notebook.add(self.frequency_panel, text="Frequency Filters")
        self.filter_notebook.add(self.spatial_panel, text="Spatial Filters")
        self.filter_notebook.add(self.noise_panel, text="Noise")
        self.filter_notebook.add(self.segmentation_panel, text="Segmentation")
        self.filter_notebook.add(self.feature_panel, text="Feature Detection")
        
        # Image display frame
        self.display_frame = ttk.Frame(self.main_frame)
        
        # Image canvas with scrollbars
        self.canvas_frame = ttk.Frame(self.display_frame)
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray80")
        
        # Scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, 
                                        command=self.canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, 
                                        command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, 
                             yscrollcommand=self.v_scrollbar.set)
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.display_frame)
        
        # Zoom controls
        self.zoom_label = ttk.Label(self.control_frame, text="Zoom:")
        self.zoom_scale = ttk.Scale(self.control_frame, from_=10, to=200, 
                                   orient=tk.HORIZONTAL, length=200,
                                   command=self.on_zoom_change)
        self.zoom_scale.set(100)  # 100% by default
        self.zoom_value_label = ttk.Label(self.control_frame, text="100%")
        
        # Compare button
        self.compare_button = ttk.Button(self.control_frame, text="Compare with Original",
                                        command=self.toggle_compare)
        self.comparing = False
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
    
    def setup_layout(self):
        """Set up widget layout"""
        # Main layout
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Filter notebook (left panel)
        self.filter_notebook.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Image display (right panel)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas and scrollbars
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Control frame
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # Zoom controls
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        self.zoom_scale.pack(side=tk.LEFT, padx=5)
        self.zoom_value_label.pack(side=tk.LEFT, padx=5)
        
        # Compare button
        self.compare_button.pack(side=tk.RIGHT, padx=10)
        
        # Status bar
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def open_image(self):
        """Open an image file"""
        file_path = filedialog.askopenfilename(
            initialdir=self.current_dir,
            title="Select Image",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            self.current_dir = os.path.dirname(file_path)
            
            try:
                # Load image
                self.original_image = image_utils.load_image(file_path)
                if self.original_image is None:
                    raise ValueError("Could not read image")
                
                # Reset processing history
                self.current_image = self.original_image.copy()
                self.processing_history = []
                self.last_filter_applied = None
                
                # Display image
                self.display_image(self.current_image)
                
                # Update status
                self.status_var.set(f"Loaded image: {os.path.basename(file_path)} " +
                                   f"({self.original_image.shape[1]}x{self.original_image.shape[0]})")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not open image: {str(e)}")
    
    def save_image(self):
        """Save current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        if hasattr(self, 'last_save_path'):
            # Save to last used path
            try:
                success = image_utils.save_image(self.current_image, self.last_save_path)
                if success:
                    self.status_var.set(f"Image saved to {self.last_save_path}")
                else:
                    raise Exception("Failed to save image")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {str(e)}")
        else:
            # No previous save path, use save as
            self.save_image_as()
    
    def save_image_as(self):
        """Save current image with a new filename"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            initialdir=self.current_dir,
            title="Save Image As",
            filetypes=(
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("TIFF files", "*.tiff"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ),
            defaultextension=".jpg"
        )
        
        if file_path:
            self.current_dir = os.path.dirname(file_path)
            self.last_save_path = file_path
            
            try:
                # Determine if JPEG quality option should be used
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                else:
                    params = None
                
                success = image_utils.save_image(self.current_image, file_path, params)
                if success:
                    self.status_var.set(f"Image saved to {file_path}")
                else:
                    raise Exception("Failed to save image")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {str(e)}")
    
    def display_image(self, image):
        """Display image on canvas
        
        Args:
            image: OpenCV image to display
        """
        if image is None:
            return
        
        # Check if image is a list - convert to numpy array if needed
        if isinstance(image, list):
            image = np.array(image)
        
        # Get current zoom level
        zoom = self.zoom_scale.get() / 100.0
        
        # Resize image for display
        display_height = int(image.shape[0] * zoom)
        display_width = int(image.shape[1] * zoom)
        
        if display_width > 0 and display_height > 0:
            display_image = cv2.resize(image, (display_width, display_height), 
                                      interpolation=cv2.INTER_AREA)
        else:
            display_image = image
        
        # Convert to PIL Image for Tkinter
        self.display_pil_image = image_utils.cv2_to_pil(display_image)
        
        # Use ImageTk.PhotoImage
        self.tk_image = ImageTk.PhotoImage(self.display_pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
    
    def on_zoom_change(self, value):
        """Handle zoom slider change
        
        Args:
            value: new zoom value (percentage)
        """
        zoom = float(value)
        if hasattr(self, 'zoom_value_label'):
            self.zoom_value_label.config(text=f"{int(zoom)}%")
        if self.current_image is not None:
            self.display_image(self.current_image)
    
    def toggle_compare(self):
        """Toggle between current image and original image"""
        if self.original_image is None:
            return
        
        self.comparing = not self.comparing
        
        if self.comparing:
            self.display_image(self.original_image)
            self.compare_button.config(text="Show Processed")
        else:
            self.display_image(self.current_image)
            self.compare_button.config(text="Compare with Original")
    
    def apply_filter(self, filter_func, *args, **kwargs):
        """Apply filter to current image
        
        Args:
            filter_func: filter function to apply
            *args: positional arguments for filter_func
            **kwargs: keyword arguments for filter_func
        """
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please open an image first")
            return
        
        # Remember current filter for reapplication
        self.last_filter_applied = (filter_func, args, kwargs)
        
        # Start processing in a separate thread to avoid UI freezing
        self.is_processing = True
        self.status_var.set("Processing...")
        
        # Save current image for undo
        self.processing_history.append(self.current_image.copy())
        
        # Define processing function
        def process():
            try:
                # Apply filter
                result = filter_func(self.current_image, *args, **kwargs)
                
                # Handle case where filter returns multiple values
                if isinstance(result, tuple):
                    processed = result[0]  # Assume first return value is the image
                else:
                    processed = result
                
                # Update UI from main thread
                self.root.after(0, lambda: self.update_after_processing(processed))
            except Exception as error:
                # Show error in main thread
                error_msg = str(error)
                self.root.after(0, lambda: self.show_processing_error(error_msg))
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=process)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def update_after_processing(self, processed_image):
        """Update UI after processing completes
        
        Args:
            processed_image: resulting image from filter
        """
        self.current_image = processed_image
        self.display_image(self.current_image)
        self.is_processing = False
        self.status_var.set("Ready")
    
    def show_processing_error(self, error_message):
        """Show error message after processing fails
        
        Args:
            error_message: error message to display
        """
        self.is_processing = False
        self.status_var.set("Error during processing")
        messagebox.showerror("Processing Error", f"An error occurred: {error_message}")
        
        # Restore previous image
        if self.processing_history:
            self.current_image = self.processing_history.pop()
            self.display_image(self.current_image)
    
    def undo(self):
        """Undo last filter application"""
        if not self.processing_history:
            messagebox.showinfo("Undo", "Nothing to undo")
            return
        
        self.current_image = self.processing_history.pop()
        self.display_image(self.current_image)
        self.status_var.set("Undo completed")
    
    def reset_to_original(self):
        """Reset current image to original image"""
        if self.original_image is None:
            return
        
        self.processing_history.append(self.current_image.copy())
        self.current_image = self.original_image.copy()
        self.display_image(self.current_image)
        self.status_var.set("Reset to original image")
    
    def show_histogram(self):
        """Show histogram of current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first")
            return
        
        histogram_window = tk.Toplevel(self.root)
        histogram_window.title("Image Histogram")
        histogram_window.geometry("800x600")
        
        histogram_panel = HistogramPanel(histogram_window, self)
        histogram_panel.pack(fill=tk.BOTH, expand=True)
        histogram_panel.set_images(self.original_image, self.current_image)
    
    def show_spectrum(self):
        """Show frequency spectrum of current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first")
            return
        
        # Create Fourier transform
        f_transform = np.fft.fftshift(np.fft.fft2(self.current_image))
        magnitude, phase, _ = frequency_filters.visualize_frequency_domain(f_transform)
        
        # Create window
        spectrum_window = tk.Toplevel(self.root)
        spectrum_window.title("Frequency Spectrum")
        spectrum_window.geometry("800x600")
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(10, 5))
        
        # Magnitude spectrum
        ax1 = fig.add_subplot(121)
        ax1.imshow(magnitude, cmap='viridis')
        ax1.set_title('Magnitude Spectrum (log scale)')
        
        # Phase spectrum
        ax2 = fig.add_subplot(122)
        ax2.imshow(phase, cmap='hsv')
        ax2.set_title('Phase Spectrum')
        
        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=spectrum_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_batch_processing(self):
        """Show batch processing panel"""
        batch_window = tk.Toplevel(self.root)
        batch_window.title("Batch Processing")
        batch_window.geometry("800x600")
        
        batch_panel = BatchProcessingPanel(batch_window, self)
        batch_panel.pack(fill=tk.BOTH, expand=True)
    
    def show_custom_kernel(self):
        """Show custom kernel editor"""
        kernel_window = tk.Toplevel(self.root)
        kernel_window.title("Custom Kernel Editor")
        kernel_window.geometry("600x500")
        
        kernel_panel = CustomKernelPanel(kernel_window, self)
        kernel_panel.pack(fill=tk.BOTH, expand=True)
    
    def show_metrics(self):
        """Show image quality metrics"""
        if self.original_image is None or self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first")
            return
        
        if np.array_equal(self.original_image, self.current_image):
            messagebox.showinfo("Metrics", "The current image is identical to the original. "
                               "Please apply some filters first.")
            return
        
        # Calculate metrics
        metrics_dict = metrics.calculate_metrics(self.original_image, self.current_image)
        
        # Create metrics window
        metrics_window = tk.Toplevel(self.root)
        metrics_window.title("Image Quality Metrics")
        metrics_window.geometry("400x300")
        
        # Create frame
        frame = ttk.Frame(metrics_window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add metrics labels
        ttk.Label(frame, text="Image Quality Metrics", font=("Arial", 14, "bold")).pack(pady=10)
        
        ttk.Label(frame, text=f"PSNR: {metrics_dict['PSNR']:.2f} dB", font=("Arial", 12)).pack(pady=5)
        ttk.Label(frame, text=f"SSIM: {metrics_dict['SSIM']:.4f}", font=("Arial", 12)).pack(pady=5)
        ttk.Label(frame, text=f"MSE: {metrics_dict['MSE']:.2f}", font=("Arial", 12)).pack(pady=5)
        
        # Add explanation
        explanation = (
            "PSNR (Peak Signal-to-Noise Ratio): Higher is better. Typical values are 20-40 dB.\n"
            "SSIM (Structural Similarity Index): Ranges from -1 to 1. Higher is better.\n"
            "MSE (Mean Squared Error): Lower is better."
        )
        
        ttk.Label(frame, text=explanation, wraplength=350, justify=tk.LEFT).pack(pady=10)
    
    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
        FilterForge Studio User Guide
        
        Getting Started:
        - Open an image using File > Open
        - Select a filter category from the tabs on the left
        - Adjust filter parameters using sliders and controls
        - Apply the filter by clicking "Apply" button
        - Save results using File > Save or File > Save As
        
        Features:
        - Frequency domain filters (low-pass, high-pass)
        - Spatial filters (edge detection, blurring, etc.)
        - Noise addition and removal
        - Image segmentation
        - Feature detection
        - Custom kernel creation
        - Batch processing
        - Image metrics and analysis
        
        Tips:
        - Use the "Compare with Original" button to view the original image
        - Enable "Real-time Preview" in View menu for instant feedback
        - Use Undo or Reset to Original if you're not satisfied with results
        """
        
        # Create guide window
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("600x500")
        
        # Create text widget
        text = tk.Text(guide_window, wrap=tk.WORD, padx=20, pady=20)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, guide_text)
        text.config(state=tk.DISABLED)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        FilterForge Studio
        
        Version 1.0.0
        
        A comprehensive image processing application with various
        filtering, segmentation, and analysis capabilities.
        
        Built with Python, OpenCV, NumPy, and Tkinter.
        """
        
        messagebox.showinfo("About FilterForge Studio", about_text)

    def show_welcome_message(self):
        """Show welcome message after the UI is fully initialized"""
        try:
            messagebox.showinfo(
                "Welcome to FilterForge Studio",
                "Welcome to FilterForge Studio!\n\n"
                "To get started, open an image using File > Open menu."
            )
        except Exception as e:
            print(f"Error showing welcome message: {e}")
            # Continue with application even if welcome message fails


def run_app():
    """Run the application from CLI"""
    root = tk.Tk()
    root.title("FilterForge Studio")
    app = FilterForgeApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    run_app() 