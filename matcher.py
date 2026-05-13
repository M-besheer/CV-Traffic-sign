import os
import cv2
import numpy as np
from feature_extraction import preprocess_image, extract_surf,  extract_lbph

def get_good_matches(matcher, des1, des2):
    """
    Finds the number of good matches using the ratio test.
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
        good = 0
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good += 1
        return good
    except cv2.error:
        return 0

def get_lbph_score(hist1, hist2):
    """
    Computes distance for LBPH using Chi-Square (lower distance = better match).
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

def normalize_scores(scores):
    """
    Applies Min-Max Normalization to a list of scores.
    """
    scores = np.array(scores, dtype=np.float32)
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val > min_val:
        return (scores - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(scores)

def main():
    train_dir = os.path.join('ds', 'DATA')
    test_dir = os.path.join('ds', 'TEST')
    
    surf_matcher = cv2.BFMatcher(cv2.NORM_L2)

    
    print("Precomputing training features... (This might take some time)")
    train_features = [] # List of dicts
    
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for class_name in train_classes:
        class_path = os.path.join(train_dir, class_name)
        images = os.listdir(class_path)
        print(f"Loading features for class {class_name} ({len(images)} images)...")
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = preprocess_image(img_path)
            if img is None:
                continue
                
            surf_des = extract_surf(img)

            lbph_hist = extract_lbph(img)
            
            train_features.append({
                'class': class_name,
                'surf': surf_des,
                'lbph': lbph_hist
            })

    print(f"Finished loading {len(train_features)} training images.\n")
    
    print("Evaluating testing images...")
    correct_predictions = 0
    total_predictions = 0
    
    test_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    for class_name in test_classes:
        class_path = os.path.join(test_dir, class_name)
        images = os.listdir(class_path)
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = preprocess_image(img_path)
            if img is None:
                continue
                
            # Calculate features for test image
            test_surf = extract_surf(img)
            test_lbph = extract_lbph(img)
            
            raw_surf_scores = []
            raw_lbph_scores = []
            
            # Compare test image against all training images
            for tf in train_features:
                surf_score = get_good_matches(surf_matcher, test_surf, tf['surf'])
                lbph_score = get_lbph_score(test_lbph, tf['lbph'])
                
                raw_surf_scores.append(surf_score)
                raw_lbph_scores.append(lbph_score)
                
            # Min-Max Normalization over all train comparisons for this test image
            norm_surf = normalize_scores(raw_surf_scores)
            
            # For Chi-Square distance, a lower value is better.
            # We invert the normalized score so that 1.0 means the best match (lowest distance)
            # and 0.0 means the worst match, aligning it with SURF
            norm_lbph = normalize_scores(raw_lbph_scores)
            norm_lbph = 1.0 - norm_lbph
            
            class_confidences = {}
            class_counts = {}
            
            # Compute average confidence per class
            for i, tf in enumerate(train_features):
                c = tf['class']
                # Average confidence for this specific train image
                conf = (1 * norm_surf[i] + 0 * norm_lbph[i])
                
                if c not in class_confidences:
                    class_confidences[c] = 0.0
                    class_counts[c] = 0
                    
                class_confidences[c] += conf
                class_counts[c] += 1
                
            best_class = None
            best_conf = -1.0
            
            # Select class with the highest average confidence
            for c in class_confidences:
                avg_conf = class_confidences[c] / class_counts[c]
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_class = c
                    
            total_predictions += 1
            if best_class == class_name:
                correct_predictions += 1
                
            print(f"Test image {class_name}/{img_name} -> Pred: {best_class} | True: {class_name} | Acc: {correct_predictions}/{total_predictions} ({(correct_predictions/total_predictions)*100:.2f}%)")

    if total_predictions > 0:
        final_accuracy = correct_predictions / total_predictions
        print(f"\nFinal Accuracy: {final_accuracy * 100:.2f}%")
    else:
        print("\nNo testing images found.")

if __name__ == '__main__':
    main()
