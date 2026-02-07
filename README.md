# 🌱 SMART-SPRAY-X: AI-Driven Pesticide Optimization System

> **A Solution for Smart India Hackathon 2025**  
> *Empowering Farmers with Intelligent Disease Detection*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📖 Overview
**SMART-SPRAY-X** is an AI-driven initiative for **Precision Agriculture**, specifically targeted for the **Smart India Hackathon 2025**. It addresses the critical issue of indiscriminate pesticide use by farmers.

Unlike traditional methods where pesticides are sprayed uniformly, **SMART-SPRAY-X** uses intelligent disease detection to determine:
1. **Whether** a plant is infected.
2. **What** disease it has.
3. **(Future Scope)** The *severity* of infection to recommend the *exact dosage* of pesticide needed.

This **"Spot-Spray"** approach drastically reduces chemical usage, lowers costs for farmers, and protects the environment from toxic residue.

## 💡 Solution Concept
*Optimization of Pesticide Sprinkling based on Infection Level*

1. **Input**: Farmer captures an image of the crop leaf via the app.
2. **Diagnosis**: The CNN model identifies the specific disease (or confirms the plant is healthy).
3. **Action**: The system (in its full deployment) would interface with a smart sprayer to release pesticide *only* on infected areas and in *optimal* quantities.


## 🚀 Key Features
- **🌿 Multi-Class Detection**: Capable of identifying **38 different plant diseases** across various crop species (Apple, Corn, Grape, Potato, Tomato, etc.).
- **⚡ High Accuracy**: Powered by a robust **Convolutional Neural Network (CNN)** achieving **~96% accuracy** on validation data.
- **📊 Interactive Dashboard**: A user-friendly interface built with **Streamlit** for seamless image uploading and real-time prediction.
- **📈 Data-Driven**: Trained on a comprehensive dataset of over **87,000 images**.

## 🛠️ Technology Stack
- **Deep Learning**: TensorFlow, Keras
- **Web Framework**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Image Processing**: Librosa (for specific feature extraction), PIL

## 📂 Dataset Details
The model is trained on a large-scale dataset consisting of 87K RGB images of healthy and diseased crop leaves.
- **Total Images**: ~87,000
- **Classes**: 38 (Including Apple Scab, Black Rot, Late Blight, etc.)
- **Split**: 80% Training / 20% Validation

## ⚙️ Installation & Usage

Follow these steps to set up the project locally:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/smart-spray-x.git
   cd "Plant Disease"
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run main.py
   ```

5. **Access the App**
   The application will open automatically in your browser at `http://localhost:8501`.

## 📸 How It Works
1. **Upload**: Navigate to the **"Plant Disease Recognition"** page and upload an image of a suspected plant leaf.
2. **Analyze**: The system processes the image using the pre-trained CNN model.
3. **Result**: The predicted disease class is displayed along with confidence metrics.

## 🧠 Model Architecture
The core is a trainable **Convolutional Neural Network (CNN)** optimized for image classification:
- **Input Layer**: Accepts 128x128 RGB images.
- **Feature Extraction**: Successive Conv2D + MaxPooling layers with ReLU activation to capture textures and patterns (spots, lesions).
- **Classification**: Dense layers with Dropout (to prevent overfitting) leading to a Softmax output layer for 38 classes.
- **Optimizer**: Adam (Adaptive Moment Estimation) for fast convergence.

## 🔮 Future Roadmap
- [ ] **Severity Estimation**: quantifying the infection level (Mild/Moderate/Severe) to calculate precise pesticide dosage.
- [ ] **Clean Class Names**: Mapping raw dataset labels (e.g., `Corn_(maize)___healthy`) to user-friendly names (e.g., `Corn - Healthy`).
- [ ] **Hardware Integration**: Connecting the detection module with nozzle control systems for automated drones or sprayers.
- [ ] **Multilingual Support**: Adding regional languages for wider accessibility to farmers across India.

## 👥 Team
**Smart India Hackathon 2025 Participant**  
*Developing innovative solutions for sustainable agriculture.*

---
*Made with ❤️ for SIH 2025*
