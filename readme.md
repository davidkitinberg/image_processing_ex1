# Image Processing – Exercise 1: Image Representations and Point Operations

## Overview

This repository contains the first exercise for an Image Processing course. It implements fundamental digital image processing operations, including:

- Reading and displaying images in grayscale and RGB
- Converting between RGB and YIQ color spaces
- Histogram equalization
- Optimal image quantization
- Interactive gamma correction

All core operations are implemented with vectorized NumPy math for efficiency, using OpenCV for I/O and Matplotlib for visualization.

---

## Repository Structure

```
.
├── ex1_utils.py   # Core image processing functions
├── gamma.py       # Interactive gamma correction GUI
└── readme.md      # This file
```

---

## Functions

### `ex1_utils.py`

| Function | Description |
|---|---|
| `myID()` | Returns the student ID number |
| `imReadAndConvert(filename, representation)` | Reads an image file and returns a normalized `float64` NumPy array in grayscale (`1`) or RGB (`2`) |
| `imDisplay(filename, representation)` | Reads and displays an image using Matplotlib |
| `transformRGB2YIQ(imRGB)` | Converts a normalized RGB image to YIQ color space |
| `transformYIQ2RGB(imYIQ)` | Converts a YIQ image back to RGB color space |
| `histogramEqualize(imOrig)` | Equalizes the histogram of a grayscale or RGB image; returns `(equalizedImage, originalHistogram, equalizedHistogram)` |
| `quantizeImage(imOrig, nQuant, nIter)` | Quantizes an image to `nQuant` colors over `nIter` iterations using optimal segment boundaries; returns `(list of quantized images, list of MSE errors)` |

### `gamma.py`

| Function | Description |
|---|---|
| `gammaDisplay(img_path, rep)` | Opens an OpenCV window with an interactive trackbar (0–200) to apply gamma correction in real time. Gamma = trackbar value / 100, so the midpoint (100) means no change. |

---

## Installation

Install the required third-party libraries with pip:

```bash
pip install numpy opencv-python matplotlib
```

**Requirements summary:**

| Library | Purpose |
|---|---|
| `numpy` | Vectorized array math |
| `opencv-python` | Image I/O and GUI (gamma trackbar) |
| `matplotlib` | Image display and error plots |

---

## Usage Examples

```python
from ex1_utils import imReadAndConvert, imDisplay, histogramEqualize, quantizeImage
from gamma import gammaDisplay

# Read and display an image
img_rgb = imReadAndConvert("image.jpg", 2)   # RGB, normalized [0, 1]
img_gray = imReadAndConvert("image.jpg", 1)  # Grayscale, normalized [0, 1]
imDisplay("image.jpg", 2)

# Histogram equalization
eq_img, hist_orig, hist_eq = histogramEqualize(img_rgb)

# Image quantization (16 colors, up to 20 iterations)
quantized_images, mse_errors = quantizeImage(img_rgb, nQuant=16, nIter=20)

# Interactive gamma correction (grayscale)
gammaDisplay("image.jpg", rep=1)
```

---

## Implementation Notes

- All images are represented as `float64` NumPy arrays with pixel values normalized to **[0, 1]**.
- Color images are read by OpenCV in BGR order and immediately converted to RGB.
- Histogram equalization on RGB images operates only on the **Y (luminance) channel** in YIQ space, preserving color.
- Image quantization likewise operates on the **Y channel** for RGB images.
- Quantization borders (`z`) are initialized so each segment contains an approximately equal number of pixels, which avoids empty-segment edge cases.
- Quantization converges early if the segment borders stop changing between iterations.