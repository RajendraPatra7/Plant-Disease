import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ImageUploader from './components/ImageUploader';
import ResultCard from './components/ResultCard';
import { Target, Droplets, Leaf, Shield, Cpu, Zap, CheckCircle2, FileText, Database } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setErrorMsg(null);
    setPredictionResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Prediction request failed');
      }

      const data = await response.json();
      if (data.is_valid_leaf === false) {
        setErrorMsg(data.error_message);
        setPredictionResult(null);
      } else {
        setPredictionResult(data);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || 'Failed to connect to backend server. Make sure FastAPI server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main style={{ flex: 1 }}>
        {/* HOME TAB */}
        {activeTab === 'home' && (
          <div>
            <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
              <h2 style={{ fontSize: '2.4rem', color: '#d8f3dc', marginBottom: '0.8rem' }}>
                AI-Driven Pesticide Optimization System
              </h2>
              <p style={{ color: '#95d5b2', fontSize: '1.15rem', maxWidth: '750px', margin: '0 auto' }}>
                Intelligent disease detection and spot-spray pesticide optimization for sustainable, healthy agriculture.
              </p>
            </div>

            {/* Feature Cards Grid */}
            <div className="card-grid">
              <div className="feature-card">
                <div className="feature-icon">🎯</div>
                <h3>Fast AI Detection</h3>
                <p>Instant plant leaf disease identification using CNN deep learning model</p>
              </div>

              <div className="feature-card">
                <div className="feature-icon">💧</div>
                <h3>Optimized Spray</h3>
                <p>Precise pesticide dosage recommendations based on infection diagnosis</p>
              </div>

              <div className="feature-card">
                <div className="feature-icon">🌱</div>
                <h3>Eco-Friendly</h3>
                <p>Reduce chemical runoff and promote sustainable farming for SIH 2025</p>
              </div>
            </div>

            {/* Call to Action CTA */}
            <div style={{ textAlign: 'center', margin: '3rem 0' }}>
              <button
                className="btn-primary"
                onClick={() => setActiveTab('recognition')}
                style={{ fontSize: '1.2rem', padding: '1rem 2.8rem' }}
              >
                🌿 Start Plant Disease Recognition
              </button>
            </div>

            {/* Stats Counter */}
            <div style={{ margin: '3.5rem 0' }}>
              <h3 style={{ color: '#d8f3dc', fontSize: '1.6rem', textAlign: 'center', marginBottom: '1.5rem' }}>
                📊 System Performance By The Numbers
              </h3>

              <div className="card-grid">
                <div className="feature-card" style={{ padding: '1.4rem' }}>
                  <div style={{ fontSize: '2.2rem', fontWeight: '900', color: '#52b788' }}>38</div>
                  <p style={{ color: '#d8f3dc', fontWeight: '600' }}>Disease Classes</p>
                </div>
                <div className="feature-card" style={{ padding: '1.4rem' }}>
                  <div style={{ fontSize: '2.2rem', fontWeight: '900', color: '#52b788' }}>87K+</div>
                  <p style={{ color: '#d8f3dc', fontWeight: '600' }}>RGB Leaf Images</p>
                </div>
                <div className="feature-card" style={{ padding: '1.4rem' }}>
                  <div style={{ fontSize: '2.2rem', fontWeight: '900', color: '#52b788' }}>95%+</div>
                  <p style={{ color: '#d8f3dc', fontWeight: '600' }}>Model Accuracy</p>
                </div>
                <div className="feature-card" style={{ padding: '1.4rem' }}>
                  <div style={{ fontSize: '2.2rem', fontWeight: '900', color: '#52b788' }}>&lt; 2s</div>
                  <p style={{ color: '#d8f3dc', fontWeight: '600' }}>Detection Speed</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* RECOGNITION TAB */}
        {activeTab === 'recognition' && (
          <div>
            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
              <h2 style={{ fontSize: '2.2rem', color: '#d8f3dc', marginBottom: '0.4rem' }}>
                🌿 Plant Disease Recognition
              </h2>
              <p style={{ color: '#95d5b2' }}>
                Upload a clear leaf photo to execute real-time AI diagnosis & spot-spraying advice
              </p>
            </div>

            <ImageUploader
              selectedFile={selectedFile}
              setSelectedFile={setSelectedFile}
              previewUrl={previewUrl}
              setPreviewUrl={setPreviewUrl}
              onPredict={handlePredict}
              loading={loading}
            />

            {errorMsg && (
              <div style={{ maxWidth: '800px', margin: '1.5rem auto', padding: '1rem', background: 'rgba(220, 53, 69, 0.2)', borderLeft: '4px solid #ff6b6b', borderRadius: '8px', color: '#ffcccc' }}>
                ❌ <strong>Error:</strong> {errorMsg}
              </div>
            )}

            <ResultCard result={predictionResult} />
          </div>
        )}

        {/* ABOUT TAB */}
        {activeTab === 'about' && (
          <div style={{ maxWidth: '900px', margin: '0 auto' }}>
            <h2 style={{ fontSize: '2.2rem', color: '#d8f3dc', textAlign: 'center', marginBottom: '1.5rem' }}>
              About Smart Spray X System
            </h2>

            <div style={{ background: 'rgba(26, 77, 46, 0.4)', padding: '2rem', borderRadius: '16px', border: '1px solid var(--border-emerald)', marginBottom: '2rem' }}>
              <h3 style={{ color: '#52b788', fontSize: '1.4rem', marginBottom: '0.8rem' }}>
                📖 Project Overview
              </h3>
              <p style={{ color: '#e8f5e9', fontSize: '1.05rem', lineHeight: '1.8' }}>
                Smart Spray X is an AI solution developed for <strong>Smart India Hackathon (SIH) 2025</strong> aimed at Precision Agriculture.
                By detecting plant leaf infections early and estimating targeted treatment actions, the system helps farmers minimize chemical application,
                reduce operational expenses, and preserve soil health.
              </p>
            </div>

            <h3 style={{ color: '#d8f3dc', fontSize: '1.4rem', marginBottom: '1rem' }}>
              📂 Dataset & Split Distribution
            </h3>

            <div className="card-grid">
              <div className="feature-card">
                <FileText size={32} color="#52b788" style={{ margin: '0 auto 0.6rem auto' }} />
                <h3>Training Set</h3>
                <p style={{ fontSize: '1.4rem', fontWeight: '800', color: '#d8f3dc' }}>70,295</p>
                <p>RGB Images</p>
              </div>

              <div className="feature-card">
                <Database size={32} color="#52b788" style={{ margin: '0 auto 0.6rem auto' }} />
                <h3>Validation Set</h3>
                <p style={{ fontSize: '1.4rem', fontWeight: '800', color: '#d8f3dc' }}>17,572</p>
                <p>RGB Images</p>
              </div>

              <div className="feature-card">
                <Zap size={32} color="#52b788" style={{ margin: '0 auto 0.6rem auto' }} />
                <h3>Test Set</h3>
                <p style={{ fontSize: '1.4rem', fontWeight: '800', color: '#d8f3dc' }}>33</p>
                <p>Sample Images</p>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Smart Spray X — Decoupled FastAPI + React Architecture © 2026 | Built with ❤️ for SIH 2025</p>
      </footer>
    </div>
  );
}
