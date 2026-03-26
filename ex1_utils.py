import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple

def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 322315300


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: The image np array
    """
    # Read the image using OpenCV. Note: OpenCV loads images in BGR format, not RGB.
    img = cv2.imread(filename)

    # Input validation: check if the image was loaded successfully to prevent crashes
    if img is None:
        raise ValueError("Could not read the image. Please check the file path.")

    # Convert the color space based on the requested representation
    if representation == 1:
        # Convert to grayscale
        out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:
        # Convert from BGR to RGB
        out_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Representation must be 1 (Grayscale) or 2 (RGB)")

    # Convert the array to float64 and normalize the pixel values to the range [0, 1]
    normalized_img = out_img.astype(np.float64) / 255.0

    return normalized_img


def imDisplay(filename: str, representation: int) -> None:
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: None
    """
    # Utilize our previous function to get the correctly formatted image array
    img = imReadAndConvert(filename, representation)

    # Open a new figure window
    plt.figure()

    # Display the image based on the representation
    if representation == 1:
        # Grayscale needs a specific color map, otherwise matplotlib uses a default heatmap
        plt.imshow(img, cmap='gray')
    elif representation == 2:
        # RGB images are handled natively since we already converted them to standard RGB
        plt.imshow(img)

    # Remove the axis ticks and numbers for a cleaner image display
    plt.axis('off')

    # Render and display the window on the screen
    plt.show()


def transformRGB2YIQ(imRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # Define the RGB to YIQ transformation matrix as provided in the instructions
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.275, -0.321],
        [0.212, -0.523, 0.311]
    ])

    # Perform matrix multiplication using the @ operator (dot product)
    # We transpose the matrix (.T) because the image shape is (Height, Width, 3)
    # and we want to multiply the 3 color channels of each pixel by the matrix.
    imYIQ = imRGB @ transform_matrix.T

    return imYIQ


def transformYIQ2RGB(imYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # Define the same base transformation matrix
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.275, -0.321],
        [0.212, -0.523, 0.311]
    ])

    # Calculate the inverse matrix to go backwards from YIQ to RGB
    inverse_matrix = np.linalg.inv(transform_matrix)

    # Perform the matrix multiplication with the inverted matrix
    imRGB = imYIQ @ inverse_matrix.T

    # Clip the values to ensure they stay within the valid [0, 1] RGB range
    # (Floating point math can sometimes produce values like 1.000000001 or -0.0000001)
    imRGB = np.clip(imRGB, 0, 1)

    return imRGB


def histogramEqualize(imOrig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Equalizes the histogram of an image
    :param imOrig: Original image
    :return: (imgEq, histOrg, histEQ)
    """
    # Check if the image is RGB (3 dimensions) or Grayscale (2 dimensions)
    is_rgb = imOrig.ndim == 3

    if is_rgb:
        # Convert to YIQ and extract the Y channel (Luminance)
        im_yiq = transformRGB2YIQ(imOrig)
        channel_to_eq = im_yiq[:, :, 0]
    else:
        # For grayscale, we work directly on the image copy so we don't modify the original
        channel_to_eq = imOrig.copy()

    # Step 1: Normalize values from [0, 1] to [0, 255] integers for histogram calculation
    img_255 = (channel_to_eq * 255).astype(np.uint8)

    # Calculate the original histogram
    # We use range=[0, 256] with 256 bins so each integer [0-255] gets its own exact bin
    histOrg, _ = np.histogram(img_255.flatten(), bins=256, range=[0, 256])

    # Step 2: Calculate the normalized Cumulative Sum (CumSum)
    cumsum = np.cumsum(histOrg)

    # Step 3: Create the LookUpTable (LUT)
    # We divide by the maximum value of cumsum (the total number of pixels) and multiply by 255
    lut = np.round((cumsum / cumsum[-1]) * 255).astype(np.uint8)

    # Step 4: Replace each intensity 'i' with LUT[i] using NumPy's fast array indexing
    img_eq_255 = lut[img_255]

    # Calculate the histogram of the newly equalized image
    histEQ, _ = np.histogram(img_eq_255.flatten(), bins=256, range=[0, 256])

    # Normalize the equalized channel back to the [0, 1] float range
    channel_eq_normalized = img_eq_255.astype(np.float64) / 255.0

    # Reconstruct the final image based on its original format
    if is_rgb:
        # Replace the old Y channel with the equalized one
        im_yiq[:, :, 0] = channel_eq_normalized
        # Convert the modified YIQ image back to RGB
        imgEq = transformYIQ2RGB(im_yiq)
    else:
        imgEq = channel_eq_normalized

    return imgEq, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> Tuple[List[np.ndarray], List[float]]:
    """
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i], List[error_i])
    """
    # Check if the image is RGB or Grayscale
    is_rgb = imOrig.ndim == 3
    if is_rgb:
        # For RGB, operate only on the Y channel (Luminance) [cite: 101]
        im_yiq = transformRGB2YIQ(imOrig)
        channel = im_yiq[:, :, 0]
    else:
        channel = imOrig.copy()

    # Normalize values from [0, 1] to [0, 255] for the histogram [cite: 88]
    img_255 = (channel * 255).astype(np.uint8)
    hist, _ = np.histogram(img_255.flatten(), bins=256, range=[0, 256])

    # Step 1: Initialize z (borders) such that each segment has approximately
    # the same number of pixels to avoid empty segments and crashes.
    cumsum = np.cumsum(hist)
    total_pixels = cumsum[-1]
    pixels_per_segment = total_pixels / nQuant

    z = np.zeros(nQuant + 1, dtype=int)
    z[0] = 0
    z[-1] = 255
    for i in range(1, nQuant):
        # Find the first index where the cumulative sum exceeds the target pixels
        z[i] = np.argmax(cumsum >= i * pixels_per_segment)

    q = np.zeros(nQuant, dtype=float)

    images_list = []
    errors_list = []

    for iter_num in range(nIter):
        # Step 2: Calculate q (the values each segment will map to) [cite: 107]
        for i in range(nQuant):
            start = z[i]
            # Ensure the last segment includes the value 255
            end = z[i + 1] if i < nQuant - 1 else z[i + 1] + 1

            segment_hist = hist[start:end]
            segment_intensities = np.arange(start, end)

            pixels_in_segment = np.sum(segment_hist)
            if pixels_in_segment == 0:
                q[i] = (start + end - 1) / 2  # Fallback just in case
            else:
                # Weighted average of intensities in this segment
                q[i] = np.sum(segment_hist * segment_intensities) / pixels_in_segment

        # Step 3: Calculate new z (borders) [cite: 105, 111, 112]
        new_z = np.zeros(nQuant + 1, dtype=int)
        new_z[0] = 0
        new_z[-1] = 255
        for i in range(1, nQuant):
            new_z[i] = round((q[i - 1] + q[i]) / 2)

        # Step 4: Create the quantized image for the current iteration [cite: 92]
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(nQuant):
            start = z[i]
            end = z[i + 1] if i < nQuant - 1 else z[i + 1] + 1
            lut[start:end] = round(q[i])

        q_img_255 = lut[img_255]
        q_img_float = q_img_255.astype(np.float64) / 255.0

        # Reconstruct the image back to its original format [cite: 102]
        if is_rgb:
            temp_yiq = im_yiq.copy()
            temp_yiq[:, :, 0] = q_img_float
            q_img_final = transformYIQ2RGB(temp_yiq)
        else:
            q_img_final = q_img_float.copy()

        images_list.append(q_img_final)

        # Step 5: Calculate the MSE error [cite: 93, 118]
        error = 0.0
        for i in range(nQuant):
            start = z[i]
            end = z[i + 1] if i < nQuant - 1 else z[i + 1] + 1
            segment_hist = hist[start:end]
            segment_intensities = np.arange(start, end)
            error += np.sum(segment_hist * ((segment_intensities - q[i]) ** 2))

        mse = error / total_pixels
        errors_list.append(mse)

        # Check for convergence: if the borders didn't change, we can stop early
        if np.array_equal(z, new_z):
            break

        # Update z for the next iteration
        z = new_z.copy()

    # Plot the error graph as requested in the assignment [cite: 119]
    plt.figure()
    plt.plot(errors_list)
    plt.title('Quantization MSE Error per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Error')
    plt.show()

    return images_list, errors_list