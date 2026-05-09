import cv2
import numpy as np
import matplotlib.pyplot as plt


def isolate_red_signs(image_path):
    """Converts image to HSV and extracts RED and BLUE objects."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None  # Fails gracefully if image is missing

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    blurred_bgr = cv2.GaussianBlur(img_bgr, (5, 5), 0)
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

    # Combine Masks!
    combined_mask = mask_red + mask_blue

    return combined_mask, img_rgb


def clean_mask_and_get_bounding_box(original_img_rgb, binary_mask):
    """Cleans mask and finds the bounding box, forgiving tiny images."""
    if binary_mask is None or original_img_rgb is None:
        return None

    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
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


def extract_features(cropped_rgb_img):
    """Uses SIFT to find keypoints in the cropped image and draws them."""

    if cropped_rgb_img is None:
        print("No image to process!")
        return None, None

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(cropped_rgb_img, cv2.COLOR_RGB2GRAY)

    # 2. Initialize the SIFT detector (Replaced ORB)
    sift = cv2.SIFT_create()

    # 3. Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if keypoints is None or len(keypoints) == 0:
        print("No features found in this image.")
        return None, None

    # 4. Draw the keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(
        cropped_rgb_img,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 5. Display the result
    plt.figure(figsize=(6, 6))
    plt.imshow(img_with_keypoints)
    plt.title(f"SIFT Keypoints Detected: {len(keypoints)}")
    plt.axis('off')
    plt.show()

    return keypoints, descriptors


def match_features(cropped_rgb, kp1, des1, template_path):
    """Matches features from the cropped image against a perfect template using SIFT."""

    if des1 is None or len(des1) == 0:
        print("No descriptors to match!")
        return None

    # 1. Load the perfect template image
    template_bgr = cv2.imread(template_path)
    if template_bgr is None:
        print(f"Template image not found at {template_path}")
        return None

    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Extract SIFT features from the template (Replaced ORB)
    sift = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(template_gray, None)

    if des2 is None or len(des2) == 0:
        print("No features found in the template.")
        return None

    # 3. Initialize the Brute Force Matcher with L2 Norm for SIFT (Replaced NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # 4. Match descriptors
    matches = bf.match(des1, des2)

    # 5. Sort matches by distance (lowest distance = best match)
    matches = sorted(matches, key=lambda x: x.distance)

    # 6. Draw the top matches
    matched_img = cv2.drawMatches(
        cropped_rgb, kp1,
        template_rgb, kp2,
        matches[:15], None,
        flags=2
    )

    # 7. Display the result
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img)
    plt.title(f"Top {min(15, len(matches))} SIFT Feature Matches")
    plt.axis('off')
    plt.show()

    return matches


if __name__ == "__main__":
    # 1. Point this to the 120 km/h sign you just uploaded
    sample_path = "image.png"

    # Run the segmentation pipeline
    mask, img_rgb = isolate_red_signs(sample_path)
    cropped_rgb = clean_mask_and_get_bounding_box(img_rgb, mask)

    if cropped_rgb is not None:
        # --- NEW CODE: SAVE THE CROPPED IMAGE ---
        # Convert RGB back to BGR so OpenCV saves the colors correctly
        cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)

        # Save it as test_image.png for your Ensemble script to read
        output_filename = "test_image.png"
        cv2.imwrite(output_filename, cropped_bgr)
        print(f"✅ Successfully saved cropped sign to: {output_filename}")
        # -----------------------------------------

        # Continue with your SIFT testing
        kp, des = extract_features(cropped_rgb)
        matches = match_features(cropped_rgb, kp, des, sample_path)
        if matches:
            print(f"Total successful matches found: {len(matches)}")
    else:
        print("❌ Failed to detect and crop the sign.")