# 😷 Face Mask Detection & Tracking System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green)

## 📌 Project Overview
This project is an end-to-end **Computer Vision system** designed to automate the monitoring of face mask compliance in real-time. Unlike standard detection models, this system integrates **Object Tracking** logic to assign unique IDs to individuals. This ensures accurate counting for statistical analysis and prevents duplicate entries in the dashboard.

The solution is lightweight, efficient, and deployed as an interactive web application using **Streamlit**.

## 🚀 Key Features
* **Real-time Detection:** Instantly detects faces and classifies them as "With Mask" or "Without Mask".
* **Smart Tracking System:** Implements tracking algorithms to maintain unique IDs for subjects across video frames, ensuring accurate "People Counting".
* **Analytics Dashboard:** A dynamic Streamlit interface displaying live stats and counters.
* **High Efficiency:** Uses **MobileNetV2** for fast inference, making it suitable for edge devices and surveillance systems.

## 📊 Dataset
The model was trained on a balanced dataset of approximately **12,000 images**:
* **~6,000 With Mask:** Scraped from Google Search.
* **~6,000 Without Mask:** Preprocessed from the CelebFace dataset.
* **Preprocessing:** Images were resized to `224x224`, normalized, and augmented (Rotation, Zoom, Horizontal Flip) to prevent overfitting.

## 🧠 Model Performance
We experimented with three Transfer Learning architectures. **MobileNetV2** was selected as the final model due to its superior balance between accuracy and speed (FPS).

| Model Architecture | Validation Accuracy | Validation Loss | Status |
| :--- | :---: | :---: | :--- |
| **VGG16** | 99.50% | 0.0179 | High Accuracy / Slow Inference |
| **ResNet50** | 77.38% | 0.5294 | Underperformed |
| **MobileNetV2** | **99.25%** | **0.0191** | **Selected (Fastest & Accurate)** |

> **Result:** The MobileNetV2 model achieved **99.25% accuracy** with minimal loss, making it ideal for real-time video processing.

## 💻 Tech Stack
* **Deep Learning:** TensorFlow, Keras.
* **Computer Vision:** OpenCV.
* **Web Framework:** Streamlit.
* **Language:** Python.

## 📸 Demo
<img width="1919" height="1023" alt="Screenshot 2025" src="https://github.com/user-attachments/assets/59e00ea4-4ce9-4a72-b535-4a40c8f5f288" />


## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/omarDlgaber/Face_Mask_Detection.git
    cd Face_Mask_Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 👤 Author
**Omar Adel**
* **Project:** Final Project | Data Science & AI Diploma
* **Date:** December 2025

---
⭐ *If you find this project useful, please consider giving it a star on GitHub!*
