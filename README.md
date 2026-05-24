# Ultrasound-Image-Classification-and-Segmentation
An end-to-end pipeline for **Breast Ultrasound Image (BUSI)** analysis covering EDA, preprocessing, multi-task classification + segmentation, and explainable AI. The goal is to classify breast lesions (**Normal / Benign / Malignant**) and simultaneously segment the lesion region, with Grad-CAM visualizations to interpret model decisions.

**Dataset:** [BUSI with GT (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) — 780 ultrasound images across three classes.

---

## Installation

```bash
pip install -q timm albumentations scikit-learn matplotlib seaborn \
                opencv-python-headless scikit-image segmentation-models-pytorch
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 1.10 (CUDA recommended).

---

## Pipeline

```
Dataset_BUSI_with_GT.zip
        │
        ▼
   EDA (class distribution, intensities, lesion area)
        │
        ▼
   Preprocessing  →  Resize 224×224 → CLAHE → ImageNet Normalize
        │
        ▼
   Stratified Split (70 / 15 / 15)
        │
        ▼
   Augmentation (medical-image-safe: Flip, Rotate, CLAHE, Brightness, GaussNoise)
        │
        ▼
   Multi-Task Training  →  Classification + Segmentation
   (BCE + FocalTversky · AdamW · CosineAnnealing · AMP · 30 epochs)
        │
        ▼
   Evaluation with TTA  (Acc, F1, Dice, IoU, ROC, Confusion Matrices)
        │
        ▼
   Grad-CAM Explanations (ViT + U-Net)
```

---

## Results

### 📊 Test Set Performance

| Model | Accuracy ↑ | F1 ↑ | Dice ↑ | IoU ↑ | Parameters |
|---|:---:|:---:|:---:|:---:|:---:|
| **ViT**    | **0.8889** | **0.8889** | 0.7490 | 0.6470 | 87,090,884 |
| **U-Net**  | 0.8205 | 0.8203 | **0.7904** | **0.7203** | 33,702,292 |
| **CNN**    | 0.7436 | 0.7342 | 0.5773 | 0.4695 | **15,545,604** |
| **ResNet** | 0.7436 | 0.7453 | 0.5947 | 0.4909 | 33,434,500 |

---

## 🔍 Explainable AI (Grad-CAM)

Grad-CAM is applied to interpret which regions drove each prediction:

- **ViT Grad-CAM** — hooks the last transformer block to combine patch activations and gradients into a spatial heatmap.
- **U-Net Grad-CAM** — hooks `layer4` of the ResNet-50 encoder; the heatmap is upsampled and overlaid on the original image.
