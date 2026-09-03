# 3D Volumetric Brain Tumor Segmentation (BraTS 2020)

An end-to-end deep learning pipeline for multi-modal 3D medical image segmentation using a custom 3D U-Net architecture built with TensorFlow/Keras and trained on the BraTS 2020 dataset.

---

## 🧠 Project Overview

Accurate localization and segmentation of gliomas from multi-modal MRI scans are critical for surgical planning and radiation therapy. This repository provides a complete implementation for volumetric brain tumor segmentation, processing multi-channel MRI volumes to predict precise tumor sub-regions.

* **Domain:** Medical Imaging, Biomedical Engineering, Computer Vision
* **Dataset:** MICCAI BraTS 2020 (Brain Tumor Segmentation)
* **Framework:** TensorFlow / Keras 3 (`tf.keras`)

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Medical Imaging I/O:** `nibabel` (NIfTI format handling)
* **Image Processing & Math:** NumPy, SciPy, Scikit-image (`skimage` for Marching Cubes 3D surface extraction)
* **Visualization:** Matplotlib, Ipywidgets (for interactive diagnostic dashboards)

---

## 📐 Model Architecture & Pipeline

1. **Multi-Modal Data Fusion:** Ingests 4 distinct MRI modalities per patient volume stacked into a 4D tensor:
   * **T1** (Native structural imaging)
   * **T1c** (Contrast-enhanced T1, highlighting active tumor boundaries)
   * **T2** (Pathology and edema characterization)
   * **FLAIR** (Fluid-attenuated inversion recovery, highlighting peritumoral edema)
2. **Preprocessing & Standardization:** Z-score normalization applied per-slice to handle scanner intensity variations, coupled with spatial cropping to a uniform $128 \times 128 \times 128$ volumetric grid.
3. **3D U-Net Architecture:** Features a symmetric encoder-decoder path with 3D Convolutional blocks (`Conv3D`), Batch Normalization, ReLU activations, Max Pooling, Dropout for regularization, and 3D Transpose Convolutions with skip-connections.
4. **Loss Function:** Optimized directly using the **Sorensen-Dice Loss** to mitigate severe background-to-tumor class imbalance.

---

## 🚀 Quick Start & Usage

### 1. Environment Setup
Ensure you have the required dependencies installed in your Python environment (preferably running inside a GPU-accelerated runtime like Kaggle or Google Colab):
```bash
pip install tensorflow nibabel numpy matplotlib scikit-image ipywidgets
