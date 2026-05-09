import cv2
import os
import pandas as pd
import numpy as np
from Segmentation import isolate_red_signs, clean_mask_and_get_bounding_box
from ensemble import get_lbp_histogram, get_color_histogram


# ==========================================
# NEW: ENHANCEMENT TRICKS FROM SLIDES
# ==========================================

def crop_center(img_rgb, margin=0.15):
    """Crops the outer border of the image to remove background noise (trees/sky)."""
    if img_rgb is None or img_rgb.shape[0] == 0 or img_rgb.shape[1] == 0:
        return img_rgb

    h, w = img_rgb.shape[:2]
    x1, y1 = int(w * margin), int(h * margin)
    x2, y2 = int(w * (1 - margin)), int(h * (1 - margin))

    # Safety check to prevent cropping the image out of existence
    if x2 <= x1 or y2 <= y1:
        return img_rgb

    return img_rgb[y1:y2, x1:x2]


def extract_sift_descriptors(img_rgb):
    """Extracts SIFT descriptors with Histogram Equalization (Slide 02)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # NEW: Histogram Equalization to fix dark/washed out signs
    gray = cv2.equalizeHist(gray)

    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(gray, None)
    return des


# ==========================================

def get_sift_match_count(des1, des2):
    """Compares two sets of SIFT descriptors."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)


def build_training_cache(train_folder_path):
    """Loops through all remaining images in the Train folder and caches their features."""
    print("--- PHASE 1: Caching Training Features ---")
    training_cache = []

    for class_id_str in os.listdir(train_folder_path):
        class_folder = os.path.join(train_folder_path, class_id_str)
        if not os.path.isdir(class_folder):
            continue

        class_id = int(class_id_str)

        for img_name in os.listdir(class_folder):
            if not img_name.endswith('.png'):
                continue

            img_path = os.path.join(class_folder, img_name)
            img_bgr = cv2.imread(img_path)

            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # --- APPLY CENTER CROP TO TRAINING TEMPLATES ---
            core_img_rgb = crop_center(img_rgb, margin=0.15)

            # Extract and store features
            training_cache.append({
                'class_id': class_id,
                'filename': img_name,
                # SIFT uses the whole image to find corners
                'sift_des': extract_sift_descriptors(img_rgb),
                # LBP and Color use the cropped core to avoid background noise
                'lbp_hist': get_lbp_histogram(core_img_rgb),
                'color_hist': get_color_histogram(core_img_rgb)
            })

    print(f"✅ Successfully cached features for {len(training_cache)} training templates.\n")
    return training_cache


def brute_force_evaluate(gtsrb_base_path, training_cache):
    """Evaluates the test set by comparing against all cached training images."""
    print("--- PHASE 2: Evaluating Test Set ---")
    csv_path = os.path.join(gtsrb_base_path, 'Test.csv')
    test_data = pd.read_csv(csv_path)

    correct_predictions = 0
    segmentation_failures = 0
    missing_images = 0

    # Using head(500) for testing. Remove to run the whole test set!
    images_to_test = test_data.head(500)

    for index, row in images_to_test.iterrows():
        true_class_id = int(row['ClassId'])
        image_path = os.path.join(gtsrb_base_path, row['Path'])

        mask, img_rgb = isolate_red_signs(image_path)

        if mask is None or img_rgb is None:
            missing_images += 1
            continue

        cropped_rgb = clean_mask_and_get_bounding_box(img_rgb, mask)

        if cropped_rgb is None:
            segmentation_failures += 1
            continue

        # --- APPLY CENTER CROP TO TEST IMAGE ---
        core_test_rgb = crop_center(cropped_rgb, margin=0.15)

        # 2. Extract Test Features
        test_sift = extract_sift_descriptors(cropped_rgb)
        test_lbp = get_lbp_histogram(core_test_rgb)
        test_color = get_color_histogram(core_test_rgb)

        best_class = None
        highest_score = -1.0

        for train_img in training_cache:
            # SIFT Score (Adjusted for tiny GTSRB images)
            sift_matches = get_sift_match_count(test_sift, train_img['sift_des'])
            sift_conf = min(sift_matches / 10.0, 1.0)

            # LBP Score
            lbp_dist = cv2.compareHist(test_lbp, train_img['lbp_hist'], cv2.HISTCMP_BHATTACHARYYA)
            lbp_conf = 1.0 - lbp_dist

            # Color Score
            color_dist = cv2.compareHist(test_color, train_img['color_hist'], cv2.HISTCMP_BHATTACHARYYA)
            color_conf = 1.0 - color_dist

            # Weighted Score
            total_score = (sift_conf * 0.45) + (lbp_conf * 0.30) + (color_conf * 0.25)

            if total_score > highest_score:
                highest_score = total_score
                best_class = train_img['class_id']

        if best_class == true_class_id:
            correct_predictions += 1

        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1} test rows... Correct so far: {correct_predictions}")

    # --- FINAL METRICS ---
    actual_tested = len(images_to_test) - missing_images

    if actual_tested == 0:
        print("\n❌ Error: No images were found! Check your dataset path.")
        return

    accuracy = (correct_predictions / actual_tested) * 100
    segmentation_fail_rate = (segmentation_failures / actual_tested) * 100

    print("\n" + "=" * 40)
    print("ULTIMATE BRUTE FORCE EVALUATION COMPLETE")
    print("=" * 40)
    print(f"Total Rows Checked:     {len(images_to_test)}")
    print(f"Missing Files Skipped:  {missing_images}")
    print(f"Actual Images Tested:   {actual_tested}")
    print(f"Correct Predictions:    {correct_predictions}")
    print(f"Segmentation Failures:  {segmentation_failures}")
    print(f"System Accuracy:        {accuracy:.2f}%")
    print(f"Segmentation Miss Rate: {segmentation_fail_rate:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    # UPDATE THESE PATHS TO YOUR LOCAL MACHINE
    GTSRB_BASE_PATH = r"D:\Desktop\CV-Traffic-sign\dataset"
    TRAIN_FOLDER_PATH = os.path.join(GTSRB_BASE_PATH, "Train")

    # Run the pipeline
    cached_train_features = build_training_cache(TRAIN_FOLDER_PATH)
    brute_force_evaluate(GTSRB_BASE_PATH, cached_train_features)