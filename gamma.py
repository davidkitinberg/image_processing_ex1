import cv2
import numpy as np


def gammaDisplay(img_path: str, rep: int) -> None:
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read the image. Please check the file path.")

    # Handle the representation
    # Note: Since we are using OpenCV's native display (cv2.imshow),
    # we MUST keep color images in BGR format. We only convert if grayscale is requested.
    if rep == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif rep == 2:
        pass  # Keep as BGR
    else:
        raise ValueError("Representation must be 1 (Grayscale) or 2 (RGB/BGR)")

    # Normalize the image to [0, 1] range for the mathematical power operation
    # If we apply power to values up to 255, we will get huge numbers and overflow
    img_float = img.astype(np.float64) / 255.0

    # Define the window name
    window_name = "Gamma Correction"

    # Create the window explicitly before adding the trackbar
    cv2.namedWindow(window_name)

    def on_trackbar(val: int) -> None:
        """
        Callback function that gets executed every time the slider moves.
        """
        # Convert the integer trackbar value (0 to 200) to a float (0.00 to 2.00)
        gamma = val / 100.0

        # Prevent completely zero gamma which might cause math warnings
        if gamma == 0:
            gamma = 0.01

        # Apply the gamma correction formula: V_out = V_in ^ gamma
        corrected_img = np.power(img_float, gamma)

        # Denormalize back to [0, 255] and convert to 8-bit integers for display
        corrected_img_255 = np.clip(corrected_img * 255.0, 0, 255).astype(np.uint8)

        # Display the updated image in the window
        cv2.imshow(window_name, corrected_img_255)

    # Create the trackbar attached to our window.
    # Initial value is 100 (which means gamma = 1.0, meaning no change to the image)
    cv2.createTrackbar("Gamma", window_name, 100, 200, on_trackbar)

    # Call the callback manually once to display the image immediately before the user moves the slider
    on_trackbar(100)

    # Pause the program execution and wait infinitely (0) until the user presses any key
    cv2.waitKey(0)

    # Clean up and close the window after a key is pressed
    cv2.destroyAllWindows()