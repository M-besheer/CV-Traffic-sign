import cv2
import numpy as np
import os


# ==========================================
# 1. FEATURE EXTRACTORS
# ==========================================

def get_sift_matches(img1_rgb, img2_rgb):
    """Extracts SIFT features and returns the number of good matches."""
    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0

    # L2 Norm for SIFT descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)


def get_color_histogram(img_rgb):
    """Extracts a normalized 3D RGB Color Histogram."""
    # Compute histogram across all 3 channels (RGB)
    hist = cv2.calcHist([img_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)  # Normalize so it can be compared
    return hist.flatten()


def get_lbp_histogram(img_rgb):
    """Calculates Local Binary Pattern (LBP) texture histogram using pure Numpy."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # ---------------------------------------------------------
    # FIXED: Explicitly define out as int32 to prevent casting errors
    out = np.zeros(gray.shape, dtype=np.int32)
    # ---------------------------------------------------------

    padded = np.pad(gray, 1, mode='constant')

    # Extract the center pixel and compare to 8 neighbors
    center = padded[1:-1, 1:-1]
    out += (padded[0:-2, 0:-2] >= center) * 1
    out += (padded[0:-2, 1:-1] >= center) * 2
    out += (padded[0:-2, 2:] >= center) * 4
    out += (padded[1:-1, 2:] >= center) * 8
    out += (padded[2:, 2:] >= center) * 16
    out += (padded[2:, 1:-1] >= center) * 32
    out += (padded[2:, 0:-2] >= center) * 64
    out += (padded[1:-1, 0:-2] >= center) * 128

    # Create histogram of the 256 possible texture patterns
    hist, _ = np.histogram(out, bins=256, range=(0, 256))
    hist = hist.astype("float32")
    cv2.normalize(hist, hist)  # Normalize
    return hist


# ==========================================
# 2. ENSEMBLE VOTING LOGIC
# ==========================================

def compare_to_template(test_img_rgb, template_img_rgb):
    """Compares the test image to a single template and returns a weighted score."""

    # 1. SIFT Evaluation (Weight: 45%)
    sift_matches = get_sift_matches(test_img_rgb, template_img_rgb)
    # Cap at 40 matches for 100% confidence
    sift_conf = min(sift_matches / 40.0, 1.0)

    # 2. LBP Texture Evaluation (Weight: 30%)
    lbp_test = get_lbp_histogram(test_img_rgb)
    lbp_template = get_lbp_histogram(template_img_rgb)
    # Bhattacharyya distance: 0 is perfect match, 1 is total mismatch
    lbp_dist = cv2.compareHist(lbp_test, lbp_template, cv2.HISTCMP_BHATTACHARYYA)
    lbp_conf = 1.0 - lbp_dist

    # 3. Color Evaluation (Weight: 25%)
    color_test = get_color_histogram(test_img_rgb)
    color_template = get_color_histogram(template_img_rgb)
    color_dist = cv2.compareHist(color_test, color_template, cv2.HISTCMP_BHATTACHARYYA)
    color_conf = 1.0 - color_dist

    # Apply Weights!
    final_score = (sift_conf * 0.45) + (lbp_conf * 0.30) + (color_conf * 0.25)

    return final_score, sift_conf, lbp_conf, color_conf


def ensemble_predict(cropped_test_img_bgr, template_folder):
    """Loops through all templates and finds the one with the highest soft-vote score."""
    test_img_rgb = cv2.cvtColor(cropped_test_img_bgr, cv2.COLOR_BGR2RGB)

    best_class = None
    highest_score = -1.0
    detailed_breakdown = ""

    print("Evaluating Templates...")
    for template_name in os.listdir(template_folder):
        template_path = os.path.join(template_folder, template_name)
        template_bgr = cv2.imread(template_path)

        if template_bgr is None:
            continue

        template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

        # Get the ensemble score
        score, sift_c, lbp_c, col_c = compare_to_template(test_img_rgb, template_rgb)

        # Keep track of the highest scoring template
        if score > highest_score:
            highest_score = score
            best_class = template_name
            detailed_breakdown = f"SIFT: {sift_c * 100:.1f}% | LBP: {lbp_c * 100:.1f}% | Color: {col_c * 100:.1f}%"

    print("\n" + "=" * 40)
    print(f"FINAL PREDICTION: {best_class}")
    print(f"Confidence Score: {highest_score * 100:.2f}%")
    print(f"Winning Vote Breakdown -> {detailed_breakdown}")
    print("=" * 40)

    return best_class


if __name__ == "__main__":
    # --- TEST THE PIPELINE ---
    # 1. Provide a path to a cropped test sign
    test_image_path = "test_image.png"  # Replace with a cropped sign from your earlier code

    # 2. Provide the path to the folder containing your 43 perfect templates
    template_bank_dir = "testttest"

    # Run the ensemble
    test_bgr = cv2.imread(test_image_path)
    if test_bgr is not None:
        ensemble_predict(test_bgr, template_bank_dir)
    else:
        print("Please provide a valid test image path.")