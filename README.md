# FilterForge Studio

A comprehensive GUI-based image processing tool for applying various filters, noise generation/removal, segmentation, and feature detection.

## Features

- **Frequency Domain Filters**: Apply low-pass and high-pass filters (Ideal, Butterworth, Gaussian)
- **Spatial Domain Filters**: Edge detection, high-boost, custom kernels
- **Noise Generation**: Add Gaussian, Salt & Pepper, Speckle, and Poisson noise
- **Noise Removal**: Apply Median, Bilateral, Gaussian, and Non-local Means denoising
- **Image Segmentation**: Thresholding (Otsu's method), watershed algorithm
- **Feature Detection**: Corners (Shi-Tomasi), blobs, edges
- **Visualization**: View frequency spectra, histograms, filter properties
- **Analysis**: Calculate PSNR, SSIM, MSE metrics
- **Batch Processing**: Process multiple images with progress tracking
- **Customization**: Create and apply custom filter kernels
- **Export Options**: Save in various formats (JPEG, PNG, TIFF) with metadata

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/FilterForge-Studio.git
   cd FilterForge-Studio
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### GUI Mode

To launch the GUI application:

```
python Project.py
```

or

```
python Project.py --gui
```

### Command Line Mode

Process a single image with a specific filter:

```
python Project.py --cli -i input.jpg -o output.jpg --filter butterworth_low --cutoff 50 --order 2
```

Add noise to an image:

```
python Project.py --cli -i input.jpg -o output.jpg --add-noise gaussian --noise-params "mean=0,sigma=25"
```

Apply segmentation:

```
python Project.py --cli -i input.jpg -o output.jpg --segment otsu --overlay
```

Batch process multiple images:

```
python Project.py --cli -i input_directory/ -o output_directory/ --batch --filter gaussian_low --cutoff 30
```

Show help for all available options:

```
python Project.py --help
```

## Project Structure

- `filters/`: Filter implementation modules
  - `frequency_filters.py`: Frequency domain filter implementations
  - `spatial_filters.py`: Spatial domain filter implementations
  - `noise.py`: Noise generation and removal
  - `segmentation.py`: Image segmentation algorithms
- `utils/`: Utility modules
  - `metrics.py`: Image quality metrics
  - `batch_processor.py`: Batch processing functionality
  - `image_utils.py`: Image loading/saving utilities
  - `cli.py`: Command-line interface
- `gui/`: GUI implementation
  - `app.py`: Main application window
  - `widgets/`: UI components

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- SciPy
- PIL/Pillow
- tqdm

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original codebase inspired by digital image processing principles
- Filter implementations based on standard algorithms in the field
