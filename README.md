# 🌱 SMART-SPRAY-X: AI-Driven Pesticide Optimization System

> **A Solution for Smart India Hackathon 2025**  
> *Empowering Farmers with Intelligent Disease Detection*

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Live-success?style=for-the-badge)

## 🌐 Live Demo

| Service | URL |
|---|---|
| 🖥️ **Frontend** (React + Vite) | [smartspray-x.vercel.app](https://smartspray-x.vercel.app) |
| ⚙️ **Backend API** (FastAPI) | [plant-disease-9v9x.onrender.com](https://plant-disease-9v9x.onrender.com) |
| 📄 **API Docs** (Swagger) | [plant-disease-9v9x.onrender.com/docs](https://plant-disease-9v9x.onrender.com/docs) |
| 💚 **Health Check** | [/api/v1/health](https://plant-disease-9v9x.onrender.com/api/v1/health) |

> ⚠️ **Note**: Backend is hosted on Render free tier — first request after inactivity may take ~30-45 seconds to wake up.

## 📖 Overview
**SMART-SPRAY-X** is an AI-driven initiative for **Precision Agriculture**, specifically targeted for the **Smart India Hackathon 2025**. It addresses the critical issue of indiscriminate pesticide use by farmers.

Unlike traditional methods where pesticides are sprayed uniformly, **SMART-SPRAY-X** uses intelligent disease detection to determine:
1. **Whether** a plant is infected.
2. **What** disease it has.
3. **(Future Scope)** The *severity* of infection to recommend the *exact dosage* of pesticide needed.

This **"Spot-Spray"** approach drastically reduces chemical usage, lowers costs for farmers, and protects the environment from toxic residue.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                          │
│                     (Vercel - React + Vite)                   │
│             https://smartspray-x.vercel.app                  │
└─────────────────────────┬───────────────────────────────────┘
                          │  REST API (HTTPS)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                           │
│                 (Render - Docker Container)                   │
│        https://plant-disease-9v9x.onrender.com               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Leaf          │  │ CNN Model    │  │ Pesticide         │   │
│  │ Validator     │  │ (38 classes) │  │ Recommendations   │   │
│  │ (HSV + OOD)   │  │ TF/Keras     │  │ Spot-Spray Engine │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🛡️ Key Features
- **🌿 Multi-Class Detection**: Identifies **38 different plant diseases** across crops (Apple, Corn, Grape, Potato, Tomato, etc.)
- **⚡ High Accuracy**: CNN achieving **~96% accuracy** on validation data
- **🛑 Non-Leaf Rejection**: Pure NumPy HSV foliage analysis + confidence thresholding rejects non-leaf images (cars, text, etc.)
- **💊 Pesticide Recommendations**: Spot-spray optimization for each disease class
- **📊 Interactive UI**: Glassmorphism dark emerald React frontend with drag & drop upload
- **📈 Data-Driven**: Trained on **87,000+ images**

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, Vite, Lucide Icons |
| **Backend** | FastAPI, Uvicorn, Python 3.10 |
| **AI/ML** | TensorFlow/Keras (CNN), NumPy |
| **Deployment** | Vercel (Frontend), Render + Docker (Backend) |
| **Image Processing** | PIL, Pure NumPy HSV |

## 📂 Dataset Details
- **Total Images**: ~87,000 RGB
- **Classes**: 38 (Apple Scab, Black Rot, Late Blight, etc.)
- **Input Size**: 128×128×3
- **Split**: 80% Training / 20% Validation

## ⚙️ Local Development

### Backend
```bash
cd "Plant Disease"
conda activate tensorflow
python start_backend.py
# → http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check + model status |
| `POST` | `/api/v1/predict` | Upload leaf image → disease prediction |
| `GET` | `/api/v1/classes` | List all 38 disease classes |
| `GET` | `/docs` | Swagger API documentation |

## 🧠 Model Architecture
- **Input Layer**: 128×128 RGB images
- **Feature Extraction**: Conv2D + MaxPooling with ReLU activation
- **Classification**: Dense layers with Dropout → Softmax (38 classes)
- **Optimizer**: Adam
- **Model Size**: 13MB (`.keras` format)

## 📸 How It Works
1. **Upload**: Drag & drop a leaf image on the React frontend
2. **Validate**: HSV foliage analysis rejects non-leaf images (< 12% foliage coverage)
3. **Predict**: CNN classifies the disease with confidence scoring
4. **Recommend**: Spot-spray pesticide recommendation displayed

## 🔮 Future Roadmap
- [ ] **Severity Estimation**: Quantifying infection level (Mild/Moderate/Severe) for precise pesticide dosage
- [ ] **Hardware Integration**: Connecting with nozzle control systems for automated drones/sprayers
- [ ] **Multilingual Support**: Adding regional languages for wider accessibility across India
- [ ] **Offline Mode**: PWA support for areas with limited connectivity

## 👥 Team
**Smart India Hackathon 2025 Participant**  
*Developing innovative solutions for sustainable agriculture.*

---
*Made with ❤️ for SIH 2025*
