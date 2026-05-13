import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

from feature_extraction import extract_surf, extract_lbph
from matcher import get_good_matches, get_lbph_score, normalize_scores
from Segmentation import isolate_red_signs, clean_mask_and_get_bounding_box

# ----------- CACHED SETUP -----------
@st.cache_resource
def load_training_features():
    """Loads and caches training features to quickly classify the uploaded image."""
    train_dir = os.path.join('ds', 'DATA')
    train_features = []
    
    if not os.path.exists(train_dir):
        return []

    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    for class_name in train_classes:
        class_path = os.path.join(train_dir, class_name)
        images = os.listdir(class_path)
        for img_name in images: # Load a subset or all depending on dataset size
            img_path = os.path.join(class_path, img_name)
            
            # Inline preprocessing for cache
            combined_mask, img_rgb = isolate_red_signs(img_path)
            if img_rgb is None: continue
            cropped_sign = clean_mask_and_get_bounding_box(img_rgb, combined_mask)
            if cropped_sign is None: cropped_sign = img_rgb
            resized_sign = cv2.resize(cropped_sign, (64, 64))
            gray_sign = cv2.cvtColor(resized_sign, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            final_img = clahe.apply(gray_sign)
            
            surf_des = extract_surf(final_img)
            lbph_hist = extract_lbph(final_img)
            
            train_features.append({
                'class': class_name,
                'surf': surf_des,
                'lbph': lbph_hist
            })
    return train_features

# ----------- MAIN APP -----------
st.set_page_config(page_title="Explainable Sign Classifier", layout="wide")
st.title("Traffic Sign Classification Pipeline")
st.write("Upload an image of a traffic sign to see exactly **how** the computer vision pipeline breaks it down, analyzes it, and classifies it.")

st.sidebar.header("Dataset Status")
with st.spinner('Loading training dataset features...'):
    train_features = load_training_features()
st.sidebar.success(f"Loaded {len(train_features)} training images into memory.")

uploaded_file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Placeholder for the final prediction result at the top
    result_placeholder = st.empty()
    
    # Show pipeline
    st.header("Step-by-Step Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Original Image")
        st.image(original_rgb, use_container_width=True)
        st.info("**Reason:** This is the raw input before any processing.")
        
    # --- STEP 2: SEGMENTATION ---
    blurred_bgr = cv2.GaussianBlur(original_bgr, (5, 5), 0)
    img_hsv = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)
    
    # Replicate logic inline to show the mask
    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([180, 255, 255])
    mask_red = cv2.inRange(img_hsv, lower_red1, upper_red1) + cv2.inRange(img_hsv, lower_red2, upper_red2)
    
    lower_blue, upper_blue = np.array([100, 50, 50]), np.array([140, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    
    lower_yellow, upper_yellow = np.array([15, 100, 100]), np.array([35, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    
    combined_mask = mask_red + mask_blue + mask_yellow

    with col2:
        st.subheader("2. Color Segmentation (HSV)")
        st.image(combined_mask, use_container_width=True)
        st.info("**Reason:** We convert the image to HSV color space to filter out Red, Blue, and Yellow pixels, highlighting where the sign likely is.")
        
    # --- STEP 3: CROPPING ---
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_sign = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 10:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_sign = original_rgb[y:y + h, x:x + w]
            
            # Draw bounding box for visualization
            bbox_img = original_rgb.copy()
            cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("3. Bounding Box & Crop")
        if cropped_sign is not None:
            st.image(bbox_img, use_container_width=True)
            st.image(cropped_sign, caption="Cropped Result", width=150)
            st.success("**Reason:** Contours are extracted from the mask to find the largest color blob. We draw a box around it and crop out the background noise.")
        else:
            cropped_sign = original_rgb
            st.warning("Segmentation failed. Falling back to using the whole image.")
            st.image(cropped_sign, use_container_width=True)

    # --- STEP 4: PREPROCESSING & FEATURE EXTRACTION ---
    resized_sign = cv2.resize(cropped_sign, (64, 64))
    gray_sign = cv2.cvtColor(resized_sign, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_img = clahe.apply(gray_sign)
    
    with col4:
        st.subheader("4. Preprocessing & CLAHE")
        st.image(final_img, width=150)
        st.success("**Reason:** The sign is resized to standard 64x64 and converted to grayscale. CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied so contours and edges pop out regardless of bad lighting/shadows.")
        
    
    # Features
    st.markdown("---")
    st.subheader("5. Feature Extraction (SURF & LBPH)")
    
    surf_des = extract_surf(final_img)
    lbph_hist = extract_lbph(final_img)
    
    try:
        _surf_extractor = cv2.xfeatures2d.SURF_create()
    except (AttributeError, cv2.error):
        _surf_extractor = cv2.SIFT_create()
        
    kp, _ = _surf_extractor.detectAndCompute(final_img, None)
    img_with_keypoints = cv2.drawKeypoints(final_img, kp, None, (255, 0, 0), 4)
    
    col5, col6 = st.columns(2)
    with col5:
        st.image(img_with_keypoints, width=200)
        st.write(f"**SURF/SIFT Keypoints found:** {len(kp) if kp else 0}")
        st.info("**Reason:** We identify local scale-invariant 'keypoints' (corners, blobs) across the sign to match against our dataset.")
        
    with col6:
        st.line_chart(lbph_hist)
        st.write("**Local Binary Pattern Histogram (LBPH)**")
        st.info("**Reason:** LBPH extracts the *texture* by comparing each pixel to its neighbors. The histogram summarizes these textures.")
        

    # --- STEP 6: MATCHING ---
    st.markdown("---")
    st.subheader("6. Classification (Matching algorithm)")
    
    if surf_des is None or len(surf_des) == 0:
        st.error("No SURF features found to match against!")
    elif len(train_features) == 0:
        st.error("Training features didn't load (check if your 'ds/DATA' folder exists).")
    else:
        with st.spinner("Matching against the dataset..."):
            surf_matcher = cv2.BFMatcher(cv2.NORM_L2)
            raw_surf = []
            raw_lbph = []
            for tf in train_features:
                raw_surf.append(get_good_matches(surf_matcher, surf_des, tf['surf']))
                raw_lbph.append(get_lbph_score(lbph_hist, tf['lbph']))
                
            norm_surf = normalize_scores(raw_surf)
            norm_lbph = 1.0 - normalize_scores(raw_lbph)
            
            class_confidences = {}
            class_counts = {}
            
            for i, tf in enumerate(train_features):
                c = tf['class']
                conf = norm_surf[i] # 1.0 weight for surf, 0 for lbph as per matcher.py
                if c not in class_confidences:
                    class_confidences[c] = 0.0
                    class_counts[c] = 0
                class_confidences[c] += conf
                class_counts[c] += 1
                
            best_class = None
            best_conf = -1.0
            
            for c in class_confidences:
                avg = class_confidences[c] / class_counts[c] if class_counts[c] > 0 else 0
                if avg > best_conf:
                    best_conf = avg
                    best_class = c
                    
            # Load Class Names
            class_names = {}
            labels_path = os.path.join('ds', 'labels.csv')
            if os.path.exists(labels_path):
                import csv
                with open(labels_path, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader) # skip header
                    for row in reader:
                        if len(row) >= 2:
                            class_names[str(row[0])] = row[1]
                            
            predicted_name = class_names.get(str(best_class), "Unknown Sign")
            
            # Show the final result at the top by injecting into the placeholder
            with result_placeholder.container():
                st.success(f"## 🎉 Predicted Class: {best_class} ({predicted_name})")
                
            st.write("**Reason:** The test image features were compared against all preprocessed training images using K-Nearest Neighbors matching (with Ratio Test) for SURF keypoints and Chi-Square distance for LBPH. After normalization, we averaged the match confidence for each class category to output the highest-scoring class.")

# --- METRICS SECTION ---
st.markdown("---")
st.header("📊 Model Metrics")
st.write("Place your metric images (e.g., confusion matrix, accuracy plots) in a folder named `metrics/` in your project directory. They will appear here!")

metrics_dir = "metrics"
if os.path.exists(metrics_dir) and os.path.isdir(metrics_dir):
    metric_images = [f for f in os.listdir(metrics_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(metric_images) > 0:
        metric_cols = st.columns(2)
        for idx, img_file in enumerate(metric_images):
            img_path = os.path.join(metrics_dir, img_file)
            col_idx = idx % 2
            with metric_cols[col_idx]:
                st.image(img_path, caption=img_file, use_container_width=True)
    else:
        st.info("No metric images found in the `metrics/` folder yet.")
else:
    st.info("Create a `metrics` folder in your project directory and add your metric images there to view them.")