# CV Traffic Sign Recognition

A classical computer-vision pipeline for traffic sign detection and classification using color segmentation, preprocessing, and feature matching.
---
# Dataset

https://www.kaggle.com/datasets/tuanai/traffic-signs-dataset

---

## 📌 Pipeline Overview

1. **Color Segmentation (HSV)**
   - Convert image to HSV and apply red/blue/yellow thresholds.
   - Combine masks to isolate sign-colored regions.

2. **Mask Cleanup & Sign Isolation**
   - Morphological closing to fill gaps.
   - Find the largest contour and crop its bounding box.

3. **Preprocessing**
   - Resize to 64×64.
   - Convert to grayscale.
   - Apply **CLAHE** for contrast enhancement.

4. **Feature Extraction**
   - **SIFT/SURF** keypoints & descriptors.
   - **LBP** histogram for texture.

5. **Matching & Classification**
   - BFMatcher (L2) for SIFT/SURF.
   - Chi‑Square distance for LBP.
   - Normalize scores → pick best class.

---

## 🧭 Pipeline Diagram

```mermaid
flowchart TB
  %% Row 1 (Left to Right)
  subgraph R1 [Input Processing]
    direction LR
    A[Input Image] --> B[HSV Conversion] --> C[Gaussian Blur] --> D[Color Masks]
  end

  %% Row 2 (Left to Right)
  subgraph R2 [Mask Processing]
    direction LR
    E[Combine Masks] --> F[Morphological Closing] --> G[Find Largest Contour] --> H[Crop Bounding Box]
  end

  %% Row 3 (Right to Left)
  subgraph R3 [Feature Extraction]
    direction RL
    L[SIFT/SURF] --> K[CLAHE] --> J[Grayscale] --> I[Resize 64x64]
  end

  %% Row 4 (Left to Right)
  subgraph R4 [Classification]
    direction LR
    M[LBP Histogram] --> N[Chi-Square Distance] --> O[Normalize Scores] --> P[Select Best Class]
  end

  %% Connecting the rows to create the snake effect
  D --> E
  H --> L
  I --> M
  P --> Q[Prediction]
  
  %% Optional styling to make it cleaner
  style R1 fill:transparent,stroke:#555,stroke-width:2px,stroke-dasharray: 5 5
  style R2 fill:transparent,stroke:#555,stroke-width:2px,stroke-dasharray: 5 5
  style R3 fill:transparent,stroke:#555,stroke-width:2px,stroke-dasharray: 5 5
  style R4 fill:transparent,stroke:#555,stroke-width:2px,stroke-dasharray: 5 5
```

---

## 📊 Metrics

![Metrics](metrics/3_overall_summary_table.png)

---

## 🚀 Run the Streamlit Interface

```bash
streamlit run app.py
```

This launches a simple UI where you can upload a sign image and see the full classification pipeline.

---

## ✅ How to Run (CLI)

```bash
python matcher.py
```

---

## 🛠️ Stack

- Python  
- OpenCV  
- NumPy  
- Matplotlib  
- Streamlit  
