import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #764ba2;
        text-align: center;
        margin-bottom: 3rem;
    }
    .project-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Load models
@st.cache_resource
def load_mnist_model():
    try:
        model = keras.models.load_model('models/mnist_model.h5')
        return model
    except:
        return None


@st.cache_resource
def load_cats_dogs_model():
    try:
        model = keras.models.load_model('models/cats_vs_dogs_model.h5')
        return model
    except:
        return None


# Header
st.markdown('<h1 class="main-header">🤖 Image Classification Web App</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deep Learning Projects - MNIST & Cats vs Dogs</p>', unsafe_allow_html=True)

# Tabs for different projects
tab1, tab2 = st.tabs(["🔢 MNIST Digit Recognition", "🐱🐶 Cats vs Dogs Classifier"])

# TAB 1: MNIST
with tab1:
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.markdown("### MNIST Handwritten Digit Recognition")
    st.markdown("Upload an image of a handwritten digit (0-9) and let the AI recognize it!")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        mnist_file = st.file_uploader("Upload a digit image", type=['png', 'jpg', 'jpeg'], key='mnist')

        if mnist_file is not None:
            # Display uploaded image
            image = Image.open(mnist_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Preprocess
            img = image.convert('L')  # Convert to grayscale
            img = img.resize((28, 28))
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            # Load model and predict
            mnist_model = load_mnist_model()

            if mnist_model is not None:
                if st.button('🔍 Recognize Digit', key='mnist_predict'):
                    with st.spinner('Analyzing...'):
                        predictions = mnist_model.predict(img_array, verbose=0)
                        predicted_digit = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_digit] * 100

                        with col2:
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.markdown("### 📊 Result")
                            st.markdown(f"# {predicted_digit}")
                            st.markdown(f"**Confidence:** {confidence:.2f}%")

                            # Show all predictions
                            st.markdown("#### All Predictions:")
                            for i in range(10):
                                prob = predictions[0][i] * 100
                                st.progress(int(prob))
                                st.text(f"Digit {i}: {prob:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("MNIST model not found! Please train the model first.")

    with col2:
        if mnist_file is None:
            st.info("👆 Upload an image to get started!")
            st.markdown("""
            **Tips for best results:**
            - Use clear images of handwritten digits
            - White digit on black background works best
            - Make sure the digit fills most of the image
            """)

# TAB 2: Cats vs Dogs
with tab2:
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.markdown("### Cats vs Dogs Classifier")
    st.markdown("Upload an image of a cat or dog and let the AI identify it!")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        animal_file = st.file_uploader("Upload a cat or dog image", type=['png', 'jpg', 'jpeg'], key='animal')

        if animal_file is not None:
            # Display uploaded image
            image = Image.open(animal_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Preprocess
            img = image.resize((150, 150))
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Load model and predict
            cats_dogs_model = load_cats_dogs_model()

            if cats_dogs_model is not None:
                if st.button('🔍 Identify Animal', key='animal_predict'):
                    with st.spinner('Analyzing...'):
                        prediction = cats_dogs_model.predict(img_array, verbose=0)[0][0]

                        if prediction > 0.5:
                            animal = "Dog 🐶"
                            confidence = prediction * 100
                            cat_confidence = (1 - prediction) * 100
                        else:
                            animal = "Cat 🐱"
                            confidence = (1 - prediction) * 100
                            cat_confidence = prediction * 100

                        with col2:
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.markdown("### 📊 Result")
                            st.markdown(f"# {animal}")
                            st.markdown(f"**Confidence:** {confidence:.2f}%")

                            # Show probabilities
                            st.markdown("#### Probabilities:")
                            st.progress(int(confidence) if "Cat" in animal else int(cat_confidence))
                            st.text(f"Cat 🐱: {100 - cat_confidence if prediction > 0.5 else confidence:.1f}%")

                            st.progress(int(cat_confidence) if "Cat" in animal else int(confidence))
                            st.text(f"Dog 🐶: {cat_confidence if prediction > 0.5 else 100 - confidence:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Cats vs Dogs model not found! Please train the model first.")

    with col2:
        if animal_file is None:
            st.info("👆 Upload an image to get started!")
            st.markdown("""
            **Tips for best results:**
            - Use clear images of cats or dogs
            - Single animal per image works best
            - Front-facing images give better accuracy
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with TensorFlow & Streamlit | Deep Learning Portfolio 2025</p>
    <p>Models trained on MNIST and Cats vs Dogs datasets</p>
</div>
""", unsafe_allow_html=True)