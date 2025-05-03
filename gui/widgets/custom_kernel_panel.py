import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
from filters import spatial_filters
from utils.image_utils import save_kernel_to_csv, create_kernel_preview

class CustomKernelPanel(ttk.Frame):
    """Panel for custom filter kernel creation and testing"""
    
    def __init__(self, parent, app):
        """Initialize panel
        
        Args:
            parent: parent widget
            app: main application instance
        """
        super().__init__(parent, padding=10)
        self.app = app
        self.kernel_size = 3  # Default size
        self.kernel_entries = []
        self.create_widgets()
    
    def create_widgets(self):
        """Create panel widgets"""
        # Kernel size selection
        size_frame = ttk.Frame(self)
        size_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(size_frame, text="Kernel Size:").pack(side=tk.LEFT, padx=5)
        
        self.size_var = tk.StringVar(value="3x3")
        size_values = ["3x3", "5x5", "7x7", "9x9"]
        size_combo = ttk.Combobox(size_frame, textvariable=self.size_var, 
                                 values=size_values, width=5, state="readonly")
        size_combo.pack(side=tk.LEFT, padx=5)
        
        # Create button to update size
        create_button = ttk.Button(size_frame, text="Create Kernel", command=self.create_kernel_matrix)
        create_button.pack(side=tk.LEFT, padx=10)
        
        # Predefined kernels selection
        predefined_frame = ttk.LabelFrame(self, text="Predefined Kernels", padding=5)
        predefined_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.predefined_var = tk.StringVar(value="sharpen")
        kernel_types = list(spatial_filters.KERNELS.keys())
        predefined_combo = ttk.Combobox(predefined_frame, textvariable=self.predefined_var,
                                      values=kernel_types, width=15, state="readonly")
        predefined_combo.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        load_predefined_button = ttk.Button(predefined_frame, text="Load Predefined",
                                          command=self.load_predefined_kernel)
        load_predefined_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Kernel editor frame
        self.editor_frame = ttk.LabelFrame(self, text="Kernel Editor", padding=5)
        self.editor_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create initial 3x3 kernel
        self.create_kernel_matrix()
        
        # Normalization options
        norm_frame = ttk.Frame(self)
        norm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        normalize_button = ttk.Button(norm_frame, text="Normalize", command=self.normalize_kernel)
        normalize_button.pack(side=tk.LEFT, padx=5)
        
        reset_button = ttk.Button(norm_frame, text="Reset", command=self.reset_kernel)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        preview_button = ttk.Button(norm_frame, text="Preview", command=self.preview_kernel)
        preview_button.pack(side=tk.LEFT, padx=5)
        
        # File operations
        file_frame = ttk.Frame(self)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        save_button = ttk.Button(file_frame, text="Save to CSV", command=self.save_kernel)
        save_button.pack(side=tk.LEFT, padx=5)
        
        load_button = ttk.Button(file_frame, text="Load from CSV", command=self.load_kernel)
        load_button.pack(side=tk.LEFT, padx=5)
        
        # Apply to image
        apply_frame = ttk.Frame(self)
        apply_frame.pack(fill=tk.X, padx=5, pady=10)
        
        apply_button = ttk.Button(apply_frame, text="Apply to Image", command=self.apply_kernel)
        apply_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5, pady=5)
    
    def create_kernel_matrix(self):
        """Create entry widgets for kernel matrix"""
        # Clear previous entries
        for row in self.kernel_entries:
            for entry in row:
                entry.destroy()
        
        # Parse size from combo box
        size_str = self.size_var.get()
        self.kernel_size = int(size_str.split('x')[0])
        
        # Create frame for entries
        self.kernel_entries = []
        
        # Clear editor frame
        for widget in self.editor_frame.winfo_children():
            widget.destroy()
        
        # Create entries for kernel matrix
        for i in range(self.kernel_size):
            row_entries = []
            for j in range(self.kernel_size):
                entry = ttk.Entry(self.editor_frame, width=5)
                entry.grid(row=i, column=j, padx=2, pady=2)
                entry.insert(0, "0.0")
                row_entries.append(entry)
            self.kernel_entries.append(row_entries)
        
        # Set center value to 1.0 by default
        center = self.kernel_size // 2
        self.kernel_entries[center][center].delete(0, tk.END)
        self.kernel_entries[center][center].insert(0, "1.0")
        
        self.status_var.set(f"Created {self.kernel_size}x{self.kernel_size} kernel matrix")
    
    def get_kernel_values(self):
        """Get kernel values from entry widgets
        
        Returns:
            numpy array representing kernel
        """
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                try:
                    value = float(self.kernel_entries[i][j].get())
                    kernel[i, j] = value
                except ValueError:
                    # If entry is not a valid float, use 0.0
                    self.kernel_entries[i][j].delete(0, tk.END)
                    self.kernel_entries[i][j].insert(0, "0.0")
                    kernel[i, j] = 0.0
        
        return kernel
    
    def set_kernel_values(self, kernel):
        """Set entry widgets with kernel values
        
        Args:
            kernel: numpy array representing kernel
        """
        # Ensure kernel size matches
        if kernel.shape[0] != self.kernel_size or kernel.shape[1] != self.kernel_size:
            # Resize the editor
            size_str = f"{kernel.shape[0]}x{kernel.shape[0]}"
            if size_str in ["3x3", "5x5", "7x7", "9x9"]:
                self.size_var.set(size_str)
                self.create_kernel_matrix()
            else:
                messagebox.showwarning("Size Mismatch", 
                                     f"Kernel size {kernel.shape[0]}x{kernel.shape[1]} doesn't match available options. "
                                     f"Resizing to {self.kernel_size}x{self.kernel_size}.")
                # Resize kernel to match current size
                center = kernel.shape[0] // 2
                new_center = self.kernel_size // 2
                offset = new_center - center
                
                new_kernel = np.zeros((self.kernel_size, self.kernel_size))
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        if 0 <= i-offset < kernel.shape[0] and 0 <= j-offset < kernel.shape[1]:
                            new_kernel[i, j] = kernel[i-offset, j-offset]
                kernel = new_kernel
        
        # Update entries
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.kernel_entries[i][j].delete(0, tk.END)
                self.kernel_entries[i][j].insert(0, f"{kernel[i, j]:.2f}")
    
    def normalize_kernel(self):
        """Normalize kernel so its elements sum to 1.0"""
        kernel = self.get_kernel_values()
        kernel_sum = np.sum(kernel)
        
        if abs(kernel_sum) < 1e-10:
            messagebox.showwarning("Normalization Error", "Cannot normalize: sum is too close to zero")
            return
        
        normalized_kernel = kernel / kernel_sum
        self.set_kernel_values(normalized_kernel)
        
        self.status_var.set("Kernel normalized")
    
    def reset_kernel(self):
        """Reset kernel to all zeros with 1.0 in center"""
        self.create_kernel_matrix()
        self.status_var.set("Kernel reset")
    
    def preview_kernel(self):
        """Show preview of current kernel"""
        kernel = self.get_kernel_values()
        
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
    
    def load_predefined_kernel(self):
        """Load a predefined kernel from the KERNELS dictionary"""
        kernel_type = self.predefined_var.get()
        
        if kernel_type in spatial_filters.KERNELS:
            kernel = spatial_filters.KERNELS[kernel_type]
            self.set_kernel_values(kernel)
            self.status_var.set(f"Loaded predefined {kernel_type} kernel")
        else:
            messagebox.showerror("Error", f"Unknown kernel type: {kernel_type}")
    
    def save_kernel(self):
        """Save kernel to CSV file"""
        kernel = self.get_kernel_values()
        
        file_path = filedialog.asksaveasfilename(
            initialdir=self.app.current_dir if hasattr(self.app, 'current_dir') else None,
            title="Save Kernel As CSV",
            defaultextension=".csv",
            filetypes=(
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            try:
                save_kernel_to_csv(kernel, file_path)
                self.status_var.set(f"Kernel saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save kernel: {str(e)}")
    
    def load_kernel(self):
        """Load kernel from CSV file"""
        file_path = filedialog.askopenfilename(
            initialdir=self.app.current_dir if hasattr(self.app, 'current_dir') else None,
            title="Load Kernel CSV",
            filetypes=(
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            )
        )
        
        if file_path:
            try:
                from utils.image_utils import load_kernel_from_csv
                kernel = load_kernel_from_csv(file_path)
                self.set_kernel_values(kernel)
                self.status_var.set(f"Kernel loaded from {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load kernel: {str(e)}")
    
    def apply_kernel(self):
        """Apply current kernel to the image"""
        if not hasattr(self.app, 'current_image') or self.app.current_image is None:
            messagebox.showwarning("Warning", "No image to apply kernel to")
            return
        
        kernel = self.get_kernel_values()
        self.app.apply_filter(spatial_filters.apply_custom_kernel, kernel)
        self.status_var.set("Applied custom kernel to image") 