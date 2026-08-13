import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';

export default function ImageUploader({ selectedFile, setSelectedFile, previewUrl, setPreviewUrl, onPredict, loading }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
        accept="image/png, image/jpeg, image/jpg"
        style={{ display: 'none' }}
      />

      <div
        className={`uploader-box ${isDragOver ? 'drag-over' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <UploadCloud className="upload-icon" />
        <h3 style={{ color: '#d8f3dc', fontSize: '1.4rem', marginBottom: '0.5rem' }}>
          {selectedFile ? 'Change Leaf Image' : 'Upload Plant Leaf Image'}
        </h3>
        <p style={{ color: '#95d5b2' }}>Drag & drop your leaf photo here, or click to browse</p>
        <p style={{ fontSize: '0.82rem', color: 'rgba(149, 213, 178, 0.7)', marginTop: '0.6rem' }}>
          Supports JPG, JPEG, PNG (Preprocessed to 128×128 RGB array)
        </p>
      </div>

      {previewUrl && (
        <div className="preview-container">
          <div style={{ position: 'relative' }}>
            <img src={previewUrl} alt="Leaf Preview" className="preview-img" />
            <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.7)', padding: '0.3rem 0.8rem', borderRadius: '12px', fontSize: '0.8rem', color: '#52b788', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <ImageIcon size={14} /> Preview Ready
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1rem' }}>
            <button
              className="btn-primary"
              onClick={onPredict}
              disabled={loading}
            >
              {loading ? (
                <>
                  <RefreshCw className="spin" size={20} /> Analyzing Leaf...
                </>
              ) : (
                <>
                  🔍 Predict Leaf Disease
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
