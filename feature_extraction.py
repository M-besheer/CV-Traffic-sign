import cv2
import numpy as np
from Segmentation import isolate_signs, clean_mask_and_get_bounding_box

def preprocess_image(image_path, target_size=(64, 64)):
    """
    Applies segmentation, crops the sign, resizes, and converts to grayscale.
    Falls back to the full image if segmentation fails.
    """
    # 1. Segmentation
    combined_mask, img_rgb = isolate_signs(image_path)
    
    if img_rgb is None:
        return None # Image not found or couldn't be read

    # 2. Get bounding box and crop
    cropped_sign = clean_mask_and_get_bounding_box(img_rgb, combined_mask)
    
    # 3. Fallback if segmentation fails
    if cropped_sign is None:
        cropped_sign = img_rgb
        
    # 4. Resize and convert to grayscale
    resized_sign = cv2.resize(cropped_sign, target_size)
    gray_sign = cv2.cvtColor(resized_sign, cv2.COLOR_RGB2GRAY)
    
    # 5. Apply CLAHE for better contrast
    # Divids into grids and applies histogram equalization to each grid individually
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_img = clahe.apply(gray_sign)
    
    return final_img


# Initialize feature extractors once to save time and reuse the same instance
_sift_extractor = cv2.SIFT_create()


def extract_lbph(img_gray):
    """Calculates Local Binary Pattern (LBP) texture histogram for a single image."""
    out = np.zeros(img_gray.shape, dtype=np.int32)
    padded = np.pad(img_gray, 1, mode='constant')
    center = padded[1:-1, 1:-1]
    out += (padded[0:-2, 0:-2] >= center) * 1
    out += (padded[0:-2, 1:-1] >= center) * 2
    out += (padded[0:-2, 2:] >= center) * 4
    out += (padded[1:-1, 2:] >= center) * 8
    out += (padded[2:, 2:] >= center) * 16
    out += (padded[2:, 1:-1] >= center) * 32
    out += (padded[2:, 0:-2] >= center) * 64
    out += (padded[1:-1, 0:-2] >= center) * 128

    hist, _ = np.histogram(out, bins=256, range=(0, 256))
    hist = hist.astype("float32")
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_sift(img_gray):
    """Extracts SIFT descriptors for a single image."""
    kp, des = _sift_extractor.detectAndCompute(img_gray, None)
    return des