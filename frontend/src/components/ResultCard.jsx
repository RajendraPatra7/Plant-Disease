import React from 'react';
import { CheckCircle2, AlertTriangle, Droplets, ShieldCheck, Info } from 'lucide-react';

export default function ResultCard({ result }) {
  if (!result) return null;

  const {
    full_class_name,
    crop,
    disease,
    status,
    confidence_percentage,
    recommendations
  } = result;

  const isHealthy = status === 'Healthy';

  return (
    <div className="result-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.5rem' }}>
        <div>
          <span className={`badge ${isHealthy ? 'badge-healthy' : 'badge-diseased'}`}>
            {isHealthy ? '✅ Healthy Plant' : '⚠️ Disease Detected'}
          </span>
          <h2 style={{ fontSize: '1.8rem', color: '#d8f3dc', marginTop: '0.6rem' }}>
            {full_class_name}
          </h2>
        </div>

        <div style={{ textAlignment: 'right', background: 'rgba(26, 77, 46, 0.5)', padding: '0.8rem 1.4rem', borderRadius: '12px', border: '1px solid var(--border-emerald)' }}>
          <div style={{ fontSize: '0.85rem', color: '#95d5b2' }}>Model Confidence</div>
          <div style={{ fontSize: '1.8rem', fontWeight: '800', color: '#52b788' }}>
            {confidence_percentage}%
          </div>
        </div>
      </div>

      {/* Confidence Bar */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: '#95d5b2' }}>
          <span>AI Diagnostic Certainty</span>
          <span>{confidence_percentage}%</span>
        </div>
        <div className="progress-bar-bg">
          <div
            className="progress-bar-fill"
            style={{ width: `${Math.min(confidence_percentage, 100)}%` }}
          />
        </div>
      </div>

      {/* Spot-Spraying & Pesticide Recommendation Box */}
      {recommendations && (
        <div className="rec-box">
          <h3 style={{ color: '#d8f3dc', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.8rem' }}>
            <Droplets color="#52b788" size={22} /> Spot-Spray Optimization Plan
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
            <div style={{ background: 'rgba(13, 27, 42, 0.5)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(82,183,136,0.2)' }}>
              <div style={{ fontSize: '0.8rem', color: '#95d5b2', marginBottom: '0.2rem' }}>Recommended Spray Action</div>
              <div style={{ color: '#e8f5e9', fontWeight: '600' }}>{recommendations.spray_action}</div>
            </div>

            <div style={{ background: 'rgba(13, 27, 42, 0.5)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(82,183,136,0.2)' }}>
              <div style={{ fontSize: '0.8rem', color: '#95d5b2', marginBottom: '0.2rem' }}>Pesticide / Chemical Type</div>
              <div style={{ color: '#52b788', fontWeight: '700' }}>{recommendations.pesticide_type}</div>
            </div>

            <div style={{ background: 'rgba(13, 27, 42, 0.5)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(82,183,136,0.2)' }}>
              <div style={{ fontSize: '0.8rem', color: '#95d5b2', marginBottom: '0.2rem' }}>Optimal Mixture Dosage</div>
              <div style={{ color: '#d8f3dc', fontWeight: '600' }}>{recommendations.dosage}</div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.6rem', marginTop: '1.2rem', padding: '0.8rem', background: 'rgba(82, 183, 136, 0.1)', borderRadius: '8px', fontSize: '0.9rem', color: '#d8f3dc' }}>
            <ShieldCheck color="#52b788" size={20} style={{ shrink: 0, marginTop: '2px' }} />
            <div>
              <strong>Eco-Friendly Spot Spraying Tip:</strong> {recommendations.eco_tip}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
