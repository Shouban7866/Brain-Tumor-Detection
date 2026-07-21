# 🧠 Brain Tumor Detection using Deep Learning

An AI-powered Brain Tumor Detection System built using **TensorFlow, ResNet50, FastAPI, and Streamlit**. The application classifies MRI brain scans into four categories and provides prediction confidence through an interactive web interface.

---

## 🚀 Features

- Transfer Learning with ResNet50
- Fine-Tuning of pretrained layers
- FastAPI REST API backend
- Streamlit interactive frontend
- MRI image upload
- Real-time predictions
- Prediction confidence score
- Probability distribution for all classes
- EarlyStopping & ReduceLROnPlateau callbacks

---

## 📂 Project Structure

```
Brain-Tumor-Detection/
│
├── model/
│   └── cnn_model.h5
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── streamlit_app.py
│
├── notebook/
│   └── training.ipynb
│
├── requirements.txt
│
├── README.md
│
└── dataset/
```

---

## 🧠 Classes

The model predicts one of the following classes:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- ResNet50
- FastAPI
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Seaborn

---
