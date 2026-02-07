import streamlit as st
import tensorflow as tf
import numpy as np
import base64
import os

st.set_page_config(page_title="Smart Spray X", page_icon="🌿", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
    .main {background-color: #f5fff6;}
    h1, h2, h3 {color: #1f6f3d;}
    .stButton>button {
        background-color: #1f6f3d;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        transition: background-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #145a2c;
        color: white;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15);
    }
    .uploadedFile {border: 2px dashed #1f6f3d; border-radius: 10px; padding: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    if not os.path.exists('trained_model.keras'):
        st.error("Model file 'trained_model.keras' not found in the current directory.")
        return None
    return tf.keras.models.load_model('trained_model.keras')

model = load_model()

#! Tensorflow Model Prediction


def model_prediction(test_image):
    if model is None:
        return None, None

    image = tf.keras.preprocessing.image.load_img(
        test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert the single image to the batch

    prediction = model.predict(input_arr)
    result_idx = np.argmax(prediction)
    return result_idx, prediction


#! Creating the UI
# * Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Plant Disease Recognition"])

# * Home Page

if app_mode == "Home":
    st.header("SMART-SPRAY-X ~ AI-Driven Pesticide Optimization System")
    image_path = "Background_image.jpeg"
    st.image(image_path, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.info("🎯 Fast AI-based disease detection")
    with col2:
        st.info("💧 Optimized pesticide usage")
    st.subheader("🌱 Intelligent Pesticide Sprinkling System Based on Infection Level")

    st.markdown("""
                Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
                """)


# * About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images
                """)
    # TODO : here the detials about out team is needed to be inserted when it is ready for its final build .


# * Recognition Page
elif (app_mode == "Plant Disease Recognition"):
    st.header("🌿 Plant Disease Recognition")
    st.markdown("Upload a clear image of a plant leaf to get an instant AI diagnosis.")
    test_image = st.file_uploader("Choose an Image")

    # * Define Class
    #TODO ~ we need to make the class names more appropriate
    class_name = ['Apple___Apple_scab',
                  'Apple___Black_rot',
                  'Apple Cedar Rust',
                  'Apple___healthy',
                  'Blueberry___healthy',
                  'Cherry_(including_sour)___Powdery_mildew',
                  'Cherry_(including_sour)___healthy',
                  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                  'Corn_(maize)___Common_rust_',
                  'Corn_(maize)___Northern_Leaf_Blight',
                  'Corn_(maize)___healthy',
                  'Grape___Black_rot',
                  'Grape___Esca_(Black_Measles)',
                  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                  'Grape___healthy',
                  'Orange___Haunglongbing_(Citrus_greening)',
                  'Peach___Bacterial_spot',
                  'Peach___healthy',
                  'Pepper,_bell___Bacterial_spot',
                  'Pepper,_bell___healthy',
                  'Potato___Early_blight',
                  'Potato___Late_blight',
                  'Potato___healthy',
                  'Raspberry___healthy',
                  'Soybean___healthy',
                  'Squash___Powdery_mildew',
                  'Strawberry___Leaf_scorch',
                  'Strawberry___healthy',
                  'Tomato___Bacterial_spot',
                  'Tomato___Early_blight',
                  'Tomato___Late_blight',
                  'Tomato___Leaf_Mold',
                  'Tomato___Septoria_leaf_spot',
                  'Tomato___Spider_mites Two-spotted_spider_mite',
                  'Tomato___Target_Spot',
                  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                  'Tomato___Tomato_mosaic_virus',
                  'Tomato___healthy']

    if (st.button("Show Image")):
        if test_image is not None:
            st.image(test_image, use_container_width=True, caption="Uploaded Leaf Image")
        else:
            st.warning("Please upload an image first.")

    # * Predcition button
    if (st.button("Predict")):
        if test_image is None:
            st.error("Please upload an image first!")
        else:
            with st.spinner("🔍 Analyzing the leaf..."):
                result_idx, prediction = model_prediction(test_image)

            if result_idx is None:
                st.error("Prediction failed because the model could not be loaded.")
            else:
                confidence = float(prediction[0][result_idx])
                st.success(f"🌿 Diagnosis: {class_name[result_idx]}")
                st.write(f"Confidence: {confidence*100:.2f}%")
                st.progress(min(confidence, 1.0))

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Smart-Spray-X")