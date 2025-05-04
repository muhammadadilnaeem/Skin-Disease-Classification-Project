import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set Streamlit page config
st.set_page_config(page_title="Skin Tumor Classifier", layout="wide", page_icon="🧬")

# ---------- PAGE 1: INTRODUCTION ---------- #
def show_introduction():
    st.markdown("""
        <style>
        .main-title {
            font-size:48px;
            color:#FF6F61;
            text-align:center;
        }
        .section-title {
            color:#4CAF50;
            font-size:28px;
        }
        .content-text {
            font-size:18px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">🧬 Skin Tumor Classification Project</h1>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">🎯 Project Objectives</h2>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">This project aims to develop a deep learning model to classify images of skin tumors into 10 different categories based on their visual characteristics.</p>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">📦 Dataset Information</h2>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">The dataset contains images of various skin tumors. It is preprocessed and augmented to improve generalization. It includes classes like Basal Cell Carcinoma, Melanoma, and more.</p>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">🧰 Tech Stack</h2>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">TensorFlow, OpenCV, NumPy, Pandas, PIL, Matplotlib, Streamlit</p>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">📈 Model Training & Evaluation</h2>', unsafe_allow_html=True)
    st.markdown('<p class="content-text">The model is trained using TensorFlow. Below are the accuracy and loss plots for training and validation:</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/training_accuracy.png", caption="Training vs Validation Accuracy")
    with col2:
        st.image("images/training_loss.png", caption="Training vs Validation Loss")

# ---------- PAGE 2: DEMO ---------- #
def load_model():
    model = tf.keras.models.load_model("model/skin_tumor_model.h5")
    return model

def preprocess_image(image):
    image = image.resize((180, 180))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def show_demo():
    st.markdown('<h1 class="main-title">🧪 Skin Tumor Prediction Demo</h1>', unsafe_allow_html=True)

    st.markdown('<p class="content-text">Upload a skin image below to get the classification result. The model will predict which type of tumor the image most likely represents.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("\n")
        model = load_model()
        processed = preprocess_image(image)

        prediction = model.predict(processed)
        class_names = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion', 'Tinea Ringworm Candidiasis']
        pred_class = class_names[np.argmax(prediction)]
        confidence = 100 * np.max(prediction)

        st.success(f"🔍 Prediction: {pred_class} with {confidence:.2f}% confidence")

# ---------- MAIN APP ---------- #
pages = {
    "📘 Introduction": show_introduction,
    "🚀 Try the Demo": show_demo
}

st.sidebar.title("🔎 Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
