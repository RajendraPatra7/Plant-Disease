import React, { useEffect, useState } from 'react';
import { Leaf, Activity, CheckCircle, AlertTriangle } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Navbar({ activeTab, setActiveTab }) {
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    const checkApi = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/health`);
        if (res.ok) {
          const data = await res.json();
          setApiStatus(data.model_loaded ? 'online' : 'no-model');
        } else {
          setApiStatus('offline');
        }
      } catch (err) {
        setApiStatus('offline');
      }
    };
    checkApi();
    const interval = setInterval(checkApi, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header>
      <div className="top-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.8rem' }}>
          <Leaf size={38} color="#52b788" />
          <h1>Smart Spray X</h1>
        </div>
        <p className="subtitle">AI-Driven Pesticide Optimization & Plant Disease Recognition</p>

        {/* Backend API Status Indicator Pill */}
        <div style={{ position: 'absolute', top: '1rem', right: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(0,0,0,0.4)', padding: '0.4rem 0.9rem', borderRadius: '20px', fontSize: '0.82rem', border: '1px solid var(--border-emerald)' }}>
          <Activity size={14} className={apiStatus === 'online' ? 'pulse' : ''} color={apiStatus === 'online' ? '#52b788' : '#ff6b6b'} />
          <span>API Backend:</span>
          {apiStatus === 'online' && <span style={{ color: '#52b788', fontWeight: 'bold' }}>Connected</span>}
          {apiStatus === 'no-model' && <span style={{ color: '#ffc107', fontWeight: 'bold' }}>Model Missing</span>}
          {apiStatus === 'offline' && <span style={{ color: '#ff6b6b', fontWeight: 'bold' }}>Offline</span>}
          {apiStatus === 'checking' && <span style={{ color: '#95d5b2' }}>Connecting...</span>}
        </div>
      </div>

      <nav className="nav-tabs">
        <button
          className={`nav-tab ${activeTab === 'home' ? 'active' : ''}`}
          onClick={() => setActiveTab('home')}
        >
          🏠 Home
        </button>
        <button
          className={`nav-tab ${activeTab === 'recognition' ? 'active' : ''}`}
          onClick={() => setActiveTab('recognition')}
        >
          🌿 Disease Recognition
        </button>
        <button
          className={`nav-tab ${activeTab === 'about' ? 'active' : ''}`}
          onClick={() => setActiveTab('about')}
        >
          ℹ️ About System
        </button>
      </nav>
    </header>
  );
}
