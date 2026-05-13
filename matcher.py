import os
import cv2
import numpy as np
from feature_extraction import preprocess_image, extract_surf,  extract_lbph
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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


def save_detailed_evaluation(y_true, y_pred):

    """
    Generates and saves a visual Confusion Matrix and a Performance Metrics table.
    """
    
    # Get unique class names
    labels = sorted(list(set(y_true)))
    
    # 1. HEATMAP 
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(20, 15)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('Actual Labels', fontsize=14)

    plt.title('Traffic Sign Confusion Matrix', fontsize=18)
    plt.savefig('1_confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # 2. PREPARE DATA
    report = classification_report(y_true, y_pred, output_dict=True)
    df_full = pd.DataFrame(report).transpose()
    
    # Split the data: Performance Metrics vs Summaries
    df_signs = df_full.iloc[:-3, :].sort_values(by='f1-score', ascending=False)
    df_summary = df_full.iloc[-3:, :] # Accuracy, Macro Avg, Weighted Avg

    #  3. Performance Metrics 
    fig1, ax1 = plt.subplots(figsize=(12, len(df_signs) * 0.4)) 
    ax1.axis('off')
    ax1.table(cellText=df_signs.round(3).values, 
              colLabels=df_signs.columns, 
              rowLabels=df_signs.index, 
              cellLoc='center', loc='center')
    plt.title(" Performance Metrics", fontsize=16, pad=20)
    plt.savefig('Performance_Metrics_able.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 4. SUMMARY TABLE 
    
    fig2, ax2 = plt.subplots(figsize=(8, 3)) 
    ax2.axis('off')
    ax2.table(cellText=df_summary.round(3).values, 
              colLabels=df_summary.columns, 
              rowLabels=df_summary.index, 
              cellLoc='center', loc='center',
              colColours=["#f2f2f2"] * len(df_summary.columns)) # Light grey header
    
    plt.title("Overall Project Summary Metrics", fontsize=16, pad=20)
    plt.savefig('3_overall_summary_table.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("Success! Generated:")
    print("- 1_confusion_matrix.png")
    print("- 2_signs_metrics_table.png")
    print("- 3_overall_summary_table.png")

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

    all_true_labels = []
    all_pred_labels = []
    
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

            all_true_labels.append(class_name)
            all_pred_labels.append(best_class)

            if best_class == class_name:
                correct_predictions += 1
                
            print(f"Test image {class_name}/{img_name} -> Pred: {best_class} | True: {class_name} | Acc: {correct_predictions}/{total_predictions} ({(correct_predictions/total_predictions)*100:.2f}%)")

    if total_predictions > 0:

        class_map = {
        '0': 'Speed limit (5km/h)', '1': 'Speed limit (15km/h)', '2': 'Speed limit (30km/h)', 
        '3': 'Speed limit (40km/h)', '4': 'Speed limit (50km/h)', '5': 'Speed limit (60km/h)', 
        '6': 'Speed limit (70km/h)', '7': 'speed limit (80km/h)', '8': 'Dont Go straight or left', 
        '9': 'Chinese Yield', '10': 'Dont Go straight', '11': 'Dont Go Left', 
        '12': 'Dont Go Left or Right', '13': 'Dont Go Right', '14': 'Dont overtake from Left', 
        '15': 'No Uturn', '16': 'No Car', '17': 'No horn', '18': 'No entry', '19': 'No stopping', 
        '20': 'Go straight or right', '21': 'Go straight', '22': 'Go Left', '23': 'Go Left or right', 
        '24': 'Go Right', '25': 'keep Left', '26': 'keep Right', '27': 'Roundabout mandatory', 
        '28': 'watch out for cars', '29': 'Horn', '30': 'Bicycles crossing', '31': 'Uturn', 
        '32': 'Road Divider', '33': 'Chinese Stop', '34': 'Danger Ahead', '35': 'Zebra Crossing', 
        '36': 'Bicycles crossing', '37': 'Children crossing', '38': 'Dangerous curve to the left', 
        '39': 'Dangerous curve to the right', '40': 'Steep Downhill', '41': 'Steep Uphill', 
        '42': 'Chinese Slow', '43': 'Go right or straight', '44': 'Go left or straight', 
        '45': 'Village', '46': 'ZigZag Curve', '47': 'Train Crossing', '48': 'Under Construction', 
        '49': 'High Voltage', '50': 'Fences', '51': 'Heavy Vehicle Accidents'
        }
        # Map the lists to the readable names
        all_true_labels = [class_map.get(str(x), x) for x in all_true_labels]
        all_pred_labels = [class_map.get(str(x), x) for x in all_pred_labels]
        save_detailed_evaluation(all_true_labels, all_pred_labels)

    else:
        print("\nNo testing images found.")

if __name__ == '__main__':
    main()
