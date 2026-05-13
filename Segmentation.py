import cv2
import numpy as np
import matplotlib.pyplot as plt


def isolate_signs(image_path):
    """Converts image to HSV and extracts RED, BLUE, and YELLOW objects."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None  # Fails gracefully if image is missing

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Smooth to reduce color spikes
    blurred_bgr = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    # Convert to HSV for better color segmentation
    img_hsv = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)

    # 1. RED Masks
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(img_hsv, lower_red1, upper_red1) + cv2.inRange(img_hsv, lower_red2, upper_red2)

    # 2. BLUE Mask (Catches mandatory signs like "Turn Right")
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # 3. YELLOW Mask (Catches warning signs and some temporary signs)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # Combine Masks!
    combined_mask = mask_red + mask_blue + mask_yellow

    return combined_mask, img_rgb


def clean_mask_and_get_bounding_box(original_img_rgb, binary_mask):
    """Cleans mask and finds the bounding box, forgiving tiny images."""
    if binary_mask is None or original_img_rgb is None:
        return None

    kernel = np.ones((3, 3), np.uint8)
    # Closing operation to fill small holes in the mask
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    # Find outlines of the detected areas
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Just grab the absolute largest color blob, ignoring area limits
        largest_contour = max(contours, key=cv2.contourArea)
        

        # Lowered the strict area requirement from 50 to 10 for tiny GTSRB images
        if cv2.contourArea(largest_contour) > 10:
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Crop out the sign
            cropped_sign = original_img_rgb[y:y + h, x:x + w]
            return cropped_sign
    
    return None