import tkinter as tk
from tkinter import ttk
import numpy as np
from filters import noise

class NoisePanel(ttk.Frame):
    """Panel for noise generation and removal"""
    
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
        # Create notebook for add/remove noise
        self.noise_notebook = ttk.Notebook(self)
        self.noise_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add Noise tab
        self.add_frame = ttk.Frame(self.noise_notebook, padding=5)
        self.noise_notebook.add(self.add_frame, text="Add Noise")
        
        # Remove Noise tab
        self.remove_frame = ttk.Frame(self.noise_notebook, padding=5)
        self.noise_notebook.add(self.remove_frame, text="Remove Noise")
        
        # Set up Add Noise tab
        self.setup_add_noise_tab()
        
        # Set up Remove Noise tab
        self.setup_remove_noise_tab()
        
        # Apply buttons
        self.add_apply_button = ttk.Button(self, text="Apply", command=self.apply_noise)
        self.add_apply_button.pack(pady=10)
        
        # Real-time preview checkbox (connected to app's variable)
        if hasattr(self.app, 'real_time_preview'):
            ttk.Checkbutton(self, text="Real-time Preview", 
                           variable=self.app.real_time_preview).pack(pady=5)
            
            # Add trace to parameters for real-time preview
            self.noise_notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
    
    def setup_add_noise_tab(self):
        """Set up widgets for adding noise"""
        # Noise type selection
        type_frame = ttk.LabelFrame(self.add_frame, text="Noise Type", padding=5)
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.noise_type = tk.StringVar(value="gaussian")
        
        ttk.Radiobutton(type_frame, text="Gaussian", variable=self.noise_type, 
                        value="gaussian").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(type_frame, text="Salt & Pepper", variable=self.noise_type,
                        value="salt_pepper").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(type_frame, text="Speckle", variable=self.noise_type,
                        value="speckle").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(type_frame, text="Poisson", variable=self.noise_type,
                        value="poisson").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Parameter frames (one for each noise type)
        # Only show the relevant frame based on selected noise type
        
        # Gaussian noise parameters
        self.gaussian_frame = ttk.LabelFrame(self.add_frame, text="Gaussian Noise Parameters", padding=5)
        
        ttk.Label(self.gaussian_frame, text="Mean:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.gaussian_mean_var = tk.DoubleVar(value=0)
        gaussian_mean_scale = ttk.Scale(self.gaussian_frame, from_=-50, to=50, 
                                        variable=self.gaussian_mean_var,
                                        orient=tk.HORIZONTAL, length=200)
        gaussian_mean_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Use lambda to update label
        gaussian_mean_scale.configure(command=lambda val: self.update_label(
            self.gaussian_mean_label, self.gaussian_mean_var.get()))
        
        self.gaussian_mean_label = ttk.Label(self.gaussian_frame, text="0.0")
        self.gaussian_mean_label.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(self.gaussian_frame, text="Sigma:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.gaussian_sigma_var = tk.DoubleVar(value=25)
        gaussian_sigma_scale = ttk.Scale(self.gaussian_frame, from_=1, to=100, 
                                        variable=self.gaussian_sigma_var,
                                        orient=tk.HORIZONTAL, length=200)
        gaussian_sigma_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        gaussian_sigma_scale.configure(command=lambda val: self.update_label(
            self.gaussian_sigma_label, self.gaussian_sigma_var.get()))
        
        self.gaussian_sigma_label = ttk.Label(self.gaussian_frame, text="25.0")
        self.gaussian_sigma_label.grid(row=1, column=2, padx=5, pady=2)
        
        # Salt and Pepper noise parameters
        self.salt_pepper_frame = ttk.LabelFrame(self.add_frame, text="Salt & Pepper Parameters", padding=5)
        
        ttk.Label(self.salt_pepper_frame, text="Salt Probability:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.salt_prob_var = tk.DoubleVar(value=0.02)
        salt_prob_scale = ttk.Scale(self.salt_pepper_frame, from_=0, to=0.2, 
                                   variable=self.salt_prob_var,
                                   orient=tk.HORIZONTAL, length=200)
        salt_prob_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        salt_prob_scale.configure(command=lambda val: self.update_label(
            self.salt_prob_label, self.salt_prob_var.get(), "{:.3f}"))
        
        self.salt_prob_label = ttk.Label(self.salt_pepper_frame, text="0.020")
        self.salt_prob_label.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(self.salt_pepper_frame, text="Pepper Probability:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.pepper_prob_var = tk.DoubleVar(value=0.02)
        pepper_prob_scale = ttk.Scale(self.salt_pepper_frame, from_=0, to=0.2, 
                                     variable=self.pepper_prob_var,
                                     orient=tk.HORIZONTAL, length=200)
        pepper_prob_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        pepper_prob_scale.configure(command=lambda val: self.update_label(
            self.pepper_prob_label, self.pepper_prob_var.get(), "{:.3f}"))
        
        self.pepper_prob_label = ttk.Label(self.salt_pepper_frame, text="0.020")
        self.pepper_prob_label.grid(row=1, column=2, padx=5, pady=2)
        
        # Speckle noise parameters
        self.speckle_frame = ttk.LabelFrame(self.add_frame, text="Speckle Noise Parameters", padding=5)
        
        ttk.Label(self.speckle_frame, text="Intensity:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.speckle_intensity_var = tk.DoubleVar(value=0.1)
        speckle_intensity_scale = ttk.Scale(self.speckle_frame, from_=0.01, to=0.5, 
                                          variable=self.speckle_intensity_var,
                                          orient=tk.HORIZONTAL, length=200)
        speckle_intensity_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        speckle_intensity_scale.configure(command=lambda val: self.update_label(
            self.speckle_intensity_label, self.speckle_intensity_var.get(), "{:.2f}"))
        
        self.speckle_intensity_label = ttk.Label(self.speckle_frame, text="0.10")
        self.speckle_intensity_label.grid(row=0, column=2, padx=5, pady=2)
        
        # Poisson noise parameters (none needed)
        self.poisson_frame = ttk.LabelFrame(self.add_frame, text="Poisson Noise Parameters", padding=5)
        ttk.Label(self.poisson_frame, text="Poisson noise is signal-dependent. No parameters needed.").pack(
            padx=5, pady=10)
        
        # Show initial frame
        self.gaussian_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add trace to noise type to update visible parameter frame
        self.noise_type.trace_add("write", self.update_noise_params_frame)
    
    def setup_remove_noise_tab(self):
        """Set up widgets for removing noise"""
        # Denoising method selection
        method_frame = ttk.LabelFrame(self.remove_frame, text="Denoising Method", padding=5)
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.denoise_method = tk.StringVar(value="median")
        
        ttk.Radiobutton(method_frame, text="Median Filter", variable=self.denoise_method, 
                       value="median").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(method_frame, text="Bilateral Filter", variable=self.denoise_method,
                       value="bilateral").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(method_frame, text="Gaussian Blur", variable=self.denoise_method,
                       value="gaussian").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(method_frame, text="Non-local Means", variable=self.denoise_method,
                       value="nlm").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Parameter frames
        
        # Median filter parameters
        self.median_frame = ttk.LabelFrame(self.remove_frame, text="Median Filter Parameters", padding=5)
        
        ttk.Label(self.median_frame, text="Kernel Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.median_ksize_var = tk.IntVar(value=5)
        median_size_values = [3, 5, 7, 9, 11, 13, 15]
        median_size_combo = ttk.Combobox(self.median_frame, textvariable=self.median_ksize_var, 
                                        values=median_size_values, width=5, state="readonly")
        median_size_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Bilateral filter parameters
        self.bilateral_frame = ttk.LabelFrame(self.remove_frame, text="Bilateral Filter Parameters", padding=5)
        
        ttk.Label(self.bilateral_frame, text="Diameter:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.bilateral_d_var = tk.IntVar(value=9)
        bilateral_d_scale = ttk.Scale(self.bilateral_frame, from_=5, to=15, 
                                     variable=self.bilateral_d_var,
                                     orient=tk.HORIZONTAL, length=200)
        bilateral_d_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        bilateral_d_label = ttk.Label(self.bilateral_frame, textvariable=self.bilateral_d_var)
        bilateral_d_label.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(self.bilateral_frame, text="Sigma Color:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.bilateral_sigma_color_var = tk.IntVar(value=75)
        bilateral_sigma_color_scale = ttk.Scale(self.bilateral_frame, from_=10, to=150, 
                                              variable=self.bilateral_sigma_color_var,
                                              orient=tk.HORIZONTAL, length=200)
        bilateral_sigma_color_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        bilateral_sigma_color_label = ttk.Label(self.bilateral_frame, 
                                             textvariable=self.bilateral_sigma_color_var)
        bilateral_sigma_color_label.grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(self.bilateral_frame, text="Sigma Space:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.bilateral_sigma_space_var = tk.IntVar(value=75)
        bilateral_sigma_space_scale = ttk.Scale(self.bilateral_frame, from_=10, to=150, 
                                              variable=self.bilateral_sigma_space_var,
                                              orient=tk.HORIZONTAL, length=200)
        bilateral_sigma_space_scale.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        bilateral_sigma_space_label = ttk.Label(self.bilateral_frame, 
                                              textvariable=self.bilateral_sigma_space_var)
        bilateral_sigma_space_label.grid(row=2, column=2, padx=5, pady=2)
        
        # Gaussian blur parameters
        self.gaussian_blur_frame = ttk.LabelFrame(self.remove_frame, text="Gaussian Blur Parameters", padding=5)
        
        ttk.Label(self.gaussian_blur_frame, text="Kernel Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.gaussian_blur_ksize_var = tk.IntVar(value=5)
        gaussian_blur_ksize_values = [3, 5, 7, 9, 11, 13, 15]
        gaussian_blur_ksize_combo = ttk.Combobox(self.gaussian_blur_frame, 
                                               textvariable=self.gaussian_blur_ksize_var, 
                                               values=gaussian_blur_ksize_values, width=5, state="readonly")
        gaussian_blur_ksize_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(self.gaussian_blur_frame, text="Sigma:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.gaussian_blur_sigma_var = tk.DoubleVar(value=0)
        gaussian_blur_sigma_scale = ttk.Scale(self.gaussian_blur_frame, from_=0, to=10, 
                                            variable=self.gaussian_blur_sigma_var,
                                            orient=tk.HORIZONTAL, length=200)
        gaussian_blur_sigma_scale.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        gaussian_blur_sigma_scale.configure(command=lambda val: self.update_label(
            self.gaussian_blur_sigma_label, self.gaussian_blur_sigma_var.get(), "{:.1f}"))
        
        self.gaussian_blur_sigma_label = ttk.Label(self.gaussian_blur_frame, text="0.0")
        self.gaussian_blur_sigma_label.grid(row=1, column=2, padx=5, pady=2)
        
        # Non-local means parameters
        self.nlm_frame = ttk.LabelFrame(self.remove_frame, text="Non-local Means Parameters", padding=5)
        
        ttk.Label(self.nlm_frame, text="Filter Strength (h):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.nlm_h_var = tk.IntVar(value=10)
        nlm_h_scale = ttk.Scale(self.nlm_frame, from_=1, to=30, 
                               variable=self.nlm_h_var,
                               orient=tk.HORIZONTAL, length=200)
        nlm_h_scale.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        nlm_h_label = ttk.Label(self.nlm_frame, textvariable=self.nlm_h_var)
        nlm_h_label.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(self.nlm_frame, text="Template Window:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.nlm_template_var = tk.IntVar(value=7)
        nlm_template_values = [3, 5, 7, 9, 11]
        nlm_template_combo = ttk.Combobox(self.nlm_frame, textvariable=self.nlm_template_var, 
                                         values=nlm_template_values, width=5, state="readonly")
        nlm_template_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(self.nlm_frame, text="Search Window:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.nlm_search_var = tk.IntVar(value=21)
        nlm_search_values = [11, 21, 31, 41, 51]
        nlm_search_combo = ttk.Combobox(self.nlm_frame, textvariable=self.nlm_search_var, 
                                      values=nlm_search_values, width=5, state="readonly")
        nlm_search_combo.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Show initial frame
        self.median_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add trace to denoise method to update visible parameter frame
        self.denoise_method.trace_add("write", self.update_denoise_params_frame)
    
    def update_label(self, label, value, format_str="{:.1f}"):
        """Update label with formatted value
        
        Args:
            label: label widget to update
            value: value to display
            format_str: format string to use
        """
        label.config(text=format_str.format(value))
    
    def update_noise_params_frame(self, *args):
        """Update visible noise parameters frame based on selected noise type"""
        # Hide all frames
        self.gaussian_frame.pack_forget()
        self.salt_pepper_frame.pack_forget()
        self.speckle_frame.pack_forget()
        self.poisson_frame.pack_forget()
        
        # Show selected frame
        if self.noise_type.get() == "gaussian":
            self.gaussian_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.noise_type.get() == "salt_pepper":
            self.salt_pepper_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.noise_type.get() == "speckle":
            self.speckle_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.noise_type.get() == "poisson":
            self.poisson_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Trigger preview if real-time is enabled
        self.preview_if_realtime()
    
    def update_denoise_params_frame(self, *args):
        """Update visible denoising parameters frame based on selected method"""
        # Hide all frames
        self.median_frame.pack_forget()
        self.bilateral_frame.pack_forget()
        self.gaussian_blur_frame.pack_forget()
        self.nlm_frame.pack_forget()
        
        # Show selected frame
        if self.denoise_method.get() == "median":
            self.median_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.denoise_method.get() == "bilateral":
            self.bilateral_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.denoise_method.get() == "gaussian":
            self.gaussian_blur_frame.pack(fill=tk.X, padx=5, pady=5)
        elif self.denoise_method.get() == "nlm":
            self.nlm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Trigger preview if real-time is enabled
        self.preview_if_realtime()
    
    def on_tab_change(self, event):
        """Handle tab change for real-time preview"""
        self.preview_if_realtime()
    
    def preview_if_realtime(self):
        """Trigger preview if real-time is enabled"""
        if hasattr(self.app, 'real_time_preview') and self.app.real_time_preview.get():
            # Check if we're already processing
            if not hasattr(self.app, 'is_processing') or not self.app.is_processing:
                self.apply_noise()
    
    def apply_noise(self):
        """Apply the selected noise operation"""
        # Determine which tab is active
        current_tab = self.noise_notebook.index(self.noise_notebook.select())
        
        if current_tab == 0:  # Add Noise tab
            self.add_noise()
        else:  # Remove Noise tab
            self.remove_noise()
    
    def add_noise(self):
        """Add noise to the image"""
        noise_type = self.noise_type.get()
        
        if noise_type == "gaussian":
            mean = self.gaussian_mean_var.get()
            sigma = self.gaussian_sigma_var.get()
            self.app.apply_filter(noise.add_gaussian_noise, mean, sigma)
            
        elif noise_type == "salt_pepper":
            salt_prob = self.salt_prob_var.get()
            pepper_prob = self.pepper_prob_var.get()
            self.app.apply_filter(noise.add_salt_pepper_noise, salt_prob, pepper_prob)
            
        elif noise_type == "speckle":
            intensity = self.speckle_intensity_var.get()
            self.app.apply_filter(noise.add_speckle_noise, intensity)
            
        elif noise_type == "poisson":
            self.app.apply_filter(noise.add_poisson_noise)
    
    def remove_noise(self):
        """Remove noise from the image"""
        denoise_method = self.denoise_method.get()
        
        if denoise_method == "median":
            ksize = self.median_ksize_var.get()
            kwargs = {'ksize': ksize}
            
        elif denoise_method == "bilateral":
            d = self.bilateral_d_var.get()
            sigma_color = self.bilateral_sigma_color_var.get()
            sigma_space = self.bilateral_sigma_space_var.get()
            kwargs = {'d': d, 'sigma_color': sigma_color, 'sigma_space': sigma_space}
            
        elif denoise_method == "gaussian":
            ksize = self.gaussian_blur_ksize_var.get()
            sigma = self.gaussian_blur_sigma_var.get()
            kwargs = {'ksize': (ksize, ksize), 'sigma': sigma}
            
        elif denoise_method == "nlm":
            h = self.nlm_h_var.get()
            template_window_size = self.nlm_template_var.get()
            search_window_size = self.nlm_search_var.get()
            kwargs = {
                'h': h, 
                'template_window_size': template_window_size, 
                'search_window_size': search_window_size
            }
        
        self.app.apply_filter(noise.remove_noise, denoise_method, **kwargs) 