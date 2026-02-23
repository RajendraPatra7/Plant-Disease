import streamlit as st
import tensorflow as tf
import numpy as np
import base64
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Smart Spray X", page_icon="🌿", layout="wide")

# Custom Styling - Premium Dark + Green Theme
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Orbitron:wght@700;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Make text selection IMPOSSIBLE TO MISS */
    ::selection {
        background: #ffc107 !important;   /* bright yellow */
        color: #000000 !important;        /* pure black text */
    }
    ::-moz-selection {
        background: #ffc107 !important;
        color: #000000 !important;
    }

    /* Make text cursor (caret) highly visible everywhere */
    .main, p, div, span, h1, h2, h3, h4, li {
        caret-color: #52b788 !important; /* bright green caret */
    }
    
    /* Main Background with Gradient */
    .main {
        background: linear-gradient(135deg, #0a1f0f 0%, #0d1b2a 50%, #0a1f0f 100%);
        color: #e8f5e9;
    }
    
    /* Header glow animation */
    @keyframes headerGlow {
        0%, 100% { box-shadow: 0 4px 20px rgba(64, 145, 108, 0.4), 0 0 30px rgba(82, 183, 136, 0.1); }
        50% { box-shadow: 0 4px 30px rgba(64, 145, 108, 0.6), 0 0 50px rgba(82, 183, 136, 0.2); }
    }

    @keyframes textShimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    /* Top Header Bar */
    .top-header {
        background: linear-gradient(135deg, #0d2818 0%, #1a4d2e 30%, #2d6a4f 60%, #1a4d2e 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border: 2px solid #52b788;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        animation: headerGlow 3s ease-in-out 0.3s infinite;
        position: relative;
        overflow: hidden;
    }

    .top-header::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; right: 0; bottom: 0;
        width: 300%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
        animation: textShimmer 5s linear infinite;
    }
    
    .top-header h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0 0 0.4rem 0;
        letter-spacing: 6px;
        text-transform: uppercase;
        background: linear-gradient(90deg, #d8f3dc, #ffffff, #95d5b2, #ffffff, #d8f3dc);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: textShimmer 4s linear infinite;
        position: relative;
    }

    .top-header .header-tagline {
        color: #95d5b2;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 2px;
        margin: 0;
        position: relative;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a4d2e 0%, #0d1b2a 100%);
        border-right: 2px solid #40916c;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    .sidebar-logo {
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 2rem;
        background: rgba(64, 145, 108, 0.1);
        border-radius: 12px;
        border: 2px solid #40916c;
    }
    
    .sidebar-logo h2 {
        color: #d8f3dc;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: #d8f3dc !important;
        font-weight: 600;
        font-size: 1.1rem;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        cursor: pointer !important;
        caret-color: transparent !important;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
        cursor: pointer !important;
        caret-color: transparent !important;
    }

    [data-testid="stSidebar"] .stSelectbox input {
        caret-color: transparent !important;
    }
    
    /* Headings */
    h1, h2, h3, h4 {
        color: #1b4332 !important;
        font-weight: 700;
    }

    h1 {
        font-size: 2.8rem !important;
        color: #1b4332 !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 2rem !important;
        color: #2d6a4f !important;
    }

    h3 {
        font-size: 1.5rem !important;
        color: #2d6a4f !important;
    }
    h4 { color: #2d6a4f !important; }
    
    /* Subtext */
    .subtitle {
        font-size: 1.2rem;
        color: #40916c;
        margin-bottom: 3rem;
        font-weight: 400;
        text-align: center;
    }
    
    /* Float animation keyframe */
    @keyframes floatBob {
        0%   { transform: translateY(0px) scale(1.05); }
        50%  { transform: translateY(-8px) scale(1.05); }
        100% { transform: translateY(0px) scale(1.05); }
    }

    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(26, 77, 46, 0.8) 0%, rgba(29, 53, 87, 0.6) 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid #40916c;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 12px 35px rgba(64, 145, 108, 0.5), 0 0 20px rgba(82, 183, 136, 0.2);
        border-color: #52b788;
        animation: floatBob 2s ease-in-out 0.3s infinite;
    }
    
    .feature-card h3 {
        color: #e8f5e9 !important;
        font-size: 1.3rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    .feature-card p {
        color: #f5fff6;
        font-size: 1rem;
        line-height: 1.6;
        font-weight: 500;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Info Cards (replacing st.info) */
    .info-card {
        background: linear-gradient(135deg, rgba(26, 77, 46, 0.6) 0%, rgba(13, 27, 42, 0.6) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #52b788;
        margin: 1rem 0;
        color: #f5fff6;
        font-size: 1.15rem;
        font-weight: 500;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(82, 183, 136, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2d6a4f 0%, #1b4332 100%);
        color: #d8f3dc;
        border-radius: 10px;
        border: 2px solid #40916c;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #40916c 0%, #2d6a4f 100%);
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(64, 145, 108, 0.5);
        border-color: #52b788;
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(26, 77, 46, 0.3);
        border: 2px dashed #40916c;
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #52b788;
        background: rgba(26, 77, 46, 0.5);
        box-shadow: 0 4px 15px rgba(64, 145, 108, 0.2);
    }
    
    [data-testid="stFileUploader"] label {
        color: #d8f3dc !important;
        font-size: 1.1rem !important;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #52b788 0%, #2d6a4f 100%);
    }
    
    /* Success/Error/Warning Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(82, 183, 136, 0.2) 0%, rgba(45, 106, 79, 0.2) 100%);
        border-left: 4px solid #52b788;
        color: #d8f3dc !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.2) 0%, rgba(139, 0, 0, 0.2) 100%);
        border-left: 4px solid #dc3545;
        color: #ffcccc !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 152, 0, 0.2) 100%);
        border-left: 4px solid #ffc107;
        color: #fff3cd !important;
    }
    
    /* Markdown Content */
    .markdown-text-container {
        color: #1b4332;
        line-height: 1.8;
        font-size: 1.05rem;
        caret-color: #52b788 !important;
    }

    /* Make all normal Streamlit markdown text visible on light backgrounds */
    [data-testid="stMarkdown"] p,
    [data-testid="stMarkdown"] li {
        color: #1b4332 !important; /* dark readable green */
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #52b788 !important;
    }
    
    /* Caption/Footer */
    .caption-text {
        text-align: center;
        color: #1b4332;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #40916c;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 1rem 1rem;
        margin-bottom: 0.5rem;
    }

    /* Hide components iframe gap */
    iframe[title="st.iframe"],
    iframe[title="streamlit_component"] {
        display: none !important;
    }
    
    /* Center Container */
    .center-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top Header Bar
st.markdown(
    """
    <div class="top-header">
        <h1>🌿 Smart Spray X</h1>
        <p class="header-tagline">AI-Driven Pesticide Optimization &amp; Plant Disease Detection</p>
    </div>
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
# * Sidebar with Logo
st.sidebar.markdown(
    """
    <div class="sidebar-logo">
        <h2>🌿 Smart Spray X</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Navigation")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Plant Disease Recognition"])

# Scroll to top when page changes
if 'last_page' not in st.session_state:
    st.session_state.last_page = app_mode

if st.session_state.last_page != app_mode:
    st.session_state.last_page = app_mode
    # Brute-force scroll to top - reset ALL scrollable elements
    components.html(
        """
        <script>
        function forceScrollTop() {
            // Reset every scrollable element in the parent document
            const doc = window.parent.document;
            const allElements = doc.querySelectorAll('*');
            allElements.forEach(el => {
                if (el.scrollTop > 0) {
                    el.scrollTop = 0;
                }
            });
            // Also try scrollIntoView on the very first element
            const firstHeader = doc.querySelector('[data-testid="stHeader"]');
            if (firstHeader) firstHeader.scrollIntoView({behavior: 'instant', block: 'start'});
            window.parent.scrollTo(0, 0);
        }
        forceScrollTop();
        let c = 0;
        const iv = setInterval(() => { forceScrollTop(); if (++c >= 15) clearInterval(iv); }, 100);
        </script>
        """,
        height=0,
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="padding: 1rem; color: #95d5b2; font-size: 0.9rem;">
        <p><strong>Powered by:</strong></p>
        <p>🤖 TensorFlow AI</p>
        <p>🎯 Smart Detection</p>
        <p>💧 Optimized Spray</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# * Home Page

if app_mode == "Home":
    # Hero Section
    st.markdown(
        """
        <div class="hero-section">
            <h1>AI-Driven Pesticide Optimization System</h1>
            <p class="subtitle">Intelligent disease detection and optimized pesticide application for healthier crops</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Background Image
    image_path = "Background_image.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    
    # Feature Cards in 3 columns
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3>Fast AI Detection</h3>
                <p>Instant disease identification using state-of-the-art deep learning models</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💧</div>
                <h3>Optimized Spray</h3>
                <p>Precise pesticide application based on infection severity levels</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌱</div>
                <h3>Eco-Friendly</h3>
                <p>Reduce chemical waste and promote sustainable farming practices</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Welcome Section
    st.markdown("## Welcome to Smart Spray X 🌿🔍")
    st.markdown(
        """
        <div class="markdown-text-container">
        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, 
        and our system will analyze it to detect any signs of diseases. Together, let's protect our crops 
        and ensure a healthier harvest!
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How It Works
    st.markdown("### How It Works")
    st.markdown(
        """
        <div class="info-card">
            <strong>1. Upload Image</strong><br>
            Go to the <strong>Disease Recognition</strong> page and upload an image of a plant with suspected diseases.
        </div>
        <div class="info-card">
            <strong>2. AI Analysis</strong><br>
            Our system will process the image using advanced algorithms to identify potential diseases.
        </div>
        <div class="info-card">
            <strong>3. Get Results</strong><br>
            View the results and recommendations for further action within seconds.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Why Choose Us
    st.markdown("### Why Choose Us?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="info-card">
                ✅ <strong>Accuracy:</strong> State-of-the-art ML techniques for precise disease detection
            </div>
            <div class="info-card">
                ✅ <strong>User-Friendly:</strong> Simple and intuitive interface for seamless experience
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
                ✅ <strong>Fast & Efficient:</strong> Receive results in seconds for quick decision-making
            </div>
            <div class="info-card">
                ✅ <strong>Sustainable:</strong> Optimize pesticide use and reduce environmental impact
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Get Started CTA
    st.markdown("### Get Started")
    st.markdown(
        """
        <div class="markdown-text-container">
        Click on the <strong>Plant Disease Recognition</strong> page in the sidebar to upload an image 
        and experience the power of our AI-driven system!
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("### 📊 By The Numbers")
    col1, col2, col3, col4 = st.columns(4)
    
    stats = [
        ("🌿", "38", "Disease Classes"),
        ("📸", "87K+", "Training Images"),
        ("🎯", "95%+", "Accuracy"),
        ("⚡", "<2s", "Detection Speed"),
    ]
    
    for col, (icon, number, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(
                f"""
                <div class="feature-card" style="padding: 1.5rem 1rem;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #52b788; margin-bottom: 0.3rem;">{number}</div>
                    <p style="font-size: 0.95rem; margin: 0;">{label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tech Stack Section
    st.markdown("### 🛠️ Powered By")
    col1, col2, col3, col4 = st.columns(4)
    
    techs = [
        ("🧠", "TensorFlow", "Deep Learning Engine"),
        ("🖥️", "Streamlit", "Interactive Web UI"),
        ("🐍", "Python", "Core Language"),
        ("📷", "OpenCV", "Image Processing"),
    ]
    
    for col, (icon, name, desc) in zip([col1, col2, col3, col4], techs):
        with col:
            st.markdown(
                f"""
                <div class="feature-card" style="padding: 1.5rem 1rem;">
                    <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <h3 style="font-size: 1.1rem !important; margin-bottom: 0.3rem !important;">{name}</h3>
                    <p style="font-size: 0.85rem; margin: 0; opacity: 0.85;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# * About Page
elif app_mode == "About":
    st.markdown("<h1 style='text-align: center;'>About Smart Spray X</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("## About the Dataset")
    st.markdown(
        """
        <div class="info-card">
            This dataset is recreated using offline augmentation from the original dataset. 
            It consists of about <strong>87K RGB images</strong> of healthy and diseased crop leaves 
            categorized into <strong>38 different classes</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### Dataset Content")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📁</div>
                <h3>Training Set</h3>
                <p>70,295 images</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧪</div>
                <h3>Test Set</h3>
                <p>33 images</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">✅</div>
                <h3>Validation Set</h3>
                <p>17,572 images</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="markdown-text-container">
        The total dataset is divided into an <strong>80/20 ratio</strong> of training and validation sets, 
        preserving the directory structure. A new directory containing test images was created for prediction purposes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    # TODO : here the detials about out team is needed to be inserted when it is ready for its final build .


# * Recognition Page
elif (app_mode == "Plant Disease Recognition"):
    st.markdown("# 🌿 Plant Disease Recognition")
    st.markdown(
        """
        <p class="subtitle">Upload a clear image of a plant leaf to get an instant AI diagnosis</p>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File Uploader Section
    test_image = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

    # * Define Class - Human-readable display names
    class_name = ['Apple Leaf - Apple Scab',
                  'Apple Leaf - Black Rot',
                  'Apple Leaf - Cedar Rust',
                  'Healthy Apple Leaf',
                  'Healthy Blueberry Leaf',
                  'Cherry Leaf - Powdery Mildew',
                  'Healthy Cherry Leaf',
                  'Corn Leaf - Cercospora (Gray Leaf Spot)',
                  'Corn Leaf - Common Rust',
                  'Corn Leaf - Northern Leaf Blight',
                  'Healthy Corn Leaf',
                  'Grape Leaf - Black Rot',
                  'Grape Leaf - Esca (Black Measles)',
                  'Grape Leaf - Leaf Blight (Isariopsis)',
                  'Healthy Grape Leaf',
                  'Orange Leaf - Huanglongbing (Citrus Greening)',
                  'Peach Leaf - Bacterial Spot',
                  'Healthy Peach Leaf',
                  'Bell Pepper Leaf - Bacterial Spot',
                  'Healthy Bell Pepper Leaf',
                  'Potato Leaf - Early Blight',
                  'Potato Leaf - Late Blight',
                  'Healthy Potato Leaf',
                  'Healthy Raspberry Leaf',
                  'Healthy Soybean Leaf',
                  'Squash Leaf - Powdery Mildew',
                  'Strawberry Leaf - Leaf Scorch',
                  'Healthy Strawberry Leaf',
                  'Tomato Leaf - Bacterial Spot',
                  'Tomato Leaf - Early Blight',
                  'Tomato Leaf - Late Blight',
                  'Tomato Leaf - Leaf Mold',
                  'Tomato Leaf - Septoria Leaf Spot',
                  'Tomato Leaf - Spider Mites',
                  'Tomato Leaf - Target Spot',
                  'Tomato Leaf - Yellow Leaf Curl Virus',
                  'Tomato Leaf - Mosaic Virus',
                  'Healthy Tomato Leaf']
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        show_image_btn = st.button("🖼️ Show Image", use_container_width=True)
    with col2:
        predict_btn = st.button("🔍 Predict Disease", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    if show_image_btn:
        if test_image is not None:
            st.image(test_image, use_container_width=True, caption="Uploaded Leaf Image")
        else:
            st.warning("⚠️ Please upload an image first.")

    # * Predcition button
    if predict_btn:
        if test_image is None:
            st.error("❌ Please upload an image first!")
        else:
            with st.spinner("🔍 Analyzing the leaf..."):
                result_idx, prediction = model_prediction(test_image)

            if result_idx is None:
                st.error("❌ Prediction failed because the model could not be loaded.")
            else:
                confidence = float(prediction[0][result_idx])
                
                # Display Results in a nice card
                st.markdown("<br>", unsafe_allow_html=True)
                st.success(f"✅ Diagnosis: **{class_name[result_idx]}**")
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <h3>Confidence Score</h3>
                        <p style="font-size: 2rem; font-weight: 700; color: #52b788;">{confidence*100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(confidence, 1.0))

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="caption-text">
        Built with ❤️ using Streamlit | Smart Spray X © 2026
    </div>
    """,
    unsafe_allow_html=True,
)