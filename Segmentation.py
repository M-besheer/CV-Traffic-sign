import cv2
import numpy as np
import matplotlib.pyplot as plt


def isolate_red_signs(image_path):
    """Converts image to HSV, applies blur, and extracts red objects with robust bounds."""

    # 1. Load the image and convert for Matplotlib viewing
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Image not found!")
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Apply a slight Gaussian Blur to smooth out pixelation and noise
    blurred_bgr = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    # 2. Convert blurred image from BGR to HSV
    img_hsv = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)

    # 3. Define the HSV range for the color RED (Widened for dark/noisy images)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 4. Create and combine the binary masks
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 5. Apply the mask back to the original image
    segmented_result = cv2.bitwise_and(img_rgb, img_rgb, mask=red_mask)

    # 6. Display the process side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original Image")
    axes[0].axis('off')

    axes[1].imshow(red_mask, cmap='gray')
    axes[1].set_title("2. Robust Binary Mask")
    axes[1].axis('off')

    axes[2].imshow(segmented_result)
    axes[2].set_title("3. Segmented Result")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return red_mask, img_rgb


def clean_mask_and_get_bounding_box(original_img_rgb, binary_mask):
    """Cleans the binary mask and finds the bounding box for small GTSRB images."""

    if binary_mask is None:
        return None

    # Reduced kernel to 3x3 so we don't crush tiny 32x32 pixel images
    kernel = np.ones((3, 3), np.uint8)

    # Apply Morphological Closing
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Find Contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = original_img_rgb.copy()

    if contours:
        # Assuming the largest red blob in the image is our sign
        largest_contour = max(contours, key=cv2.contourArea)

        # Dropped area threshold from 500 to 50 to catch small GTSRB signs
        if cv2.contourArea(largest_contour) > 50:
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Draw a bright green rectangle
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop out the sign
            cropped_sign = original_img_rgb[y:y + h, x:x + w]

            # Display the results
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(clean_mask, cmap='gray')
            axes[0].set_title("1. Cleaned Mask")
            axes[0].axis('off')

            axes[1].imshow(output_img)
            axes[1].set_title("2. Bounding Box Detected")
            axes[1].axis('off')

            axes[2].imshow(cropped_sign)
            axes[2].set_title("3. Cropped ROI")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

            return cropped_sign

    print("No valid contours found even with lowered thresholds!")
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
    sample_path = "image.png"  # Replace with your actual image path
    # Run the pipeline
    mask, img_rgb = isolate_red_signs(sample_path)
    cropped = clean_mask_and_get_bounding_box(img_rgb, mask)
    kp, des = extract_features(cropped)
    # Match against template
    matches = match_features(cropped, kp, des, sample_path)
    if matches:
        print(f"Total successful matches found: {len(matches)}")