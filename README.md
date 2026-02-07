# Multiclass-Brain-Tumor-Detection
**(Machine Learning Vision Project)**

This project implements a computer vision pipeline to classify brain tumor MRI images into multiple tumor categories using classical machine learning algorithms such as SVM and Random Forest.

The system performs image preprocessing, feature extraction, model training, evaluation, and prediction on new MRI images.

**Project Overview**

The goal of this project is to build an automated system that can classify MRI brain images into tumor classes.
The pipeline includes:

- Image loading from dataset folders

- Image resizing and preprocessing

- Feature extraction (flattened pixel vectors)

- Model training (SVM & Random Forest)

- Model evaluation using classification metrics

- Model saving/loading

- Prediction on new MRI images

This project demonstrates an end-to-end vision ML workflow from raw images to deployment-ready prediction.

**Technologies Used**

Python
OpenCV | scikit-learn | NumPy | Matplotlib  | Seaborn | scikit-image | joblib

**Future Improvements**

- Use CNN/Deep Learning models (ResNet, VGG)

- Add data augmentation

- Improve accuracy with feature extraction

- Deploy as web app using Streamlit/Flask

- Convert to real-time diagnostic tool
