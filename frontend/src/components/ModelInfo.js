import React from 'react';

const ModelInfo = ({ modelInfo }) => {
  if (!modelInfo) {
    return (
      <div className="card">
        <h2>📊 Model Information</h2>
        <p>Loading model information...</p>
      </div>
    );
  }

  const { model_type, n_features, metrics } = modelInfo;

  return (
    <div className="card">
      <h2>📊 Model Information</h2>
      
      <div className="model-info">
        <div className="info-item">
          <span className="info-label">Model Type:</span>
          <span className="info-value">{model_type}</span>
        </div>
        
        <div className="info-item">
          <span className="info-label">Number of Features:</span>
          <span className="info-value">{n_features}</span>
        </div>
      </div>

      {metrics && Object.keys(metrics).length > 0 && (
        <>
          <h3 style={{ marginTop: '2rem', marginBottom: '1rem', color: '#667eea' }}>
            Model Performance
          </h3>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Accuracy</div>
              <div className="metric-value">
                {(metrics.accuracy * 100).toFixed(2)}%
              </div>
            </div>
            
            <div className="metric-card">
              <div className="metric-label">Precision</div>
              <div className="metric-value">
                {(metrics.precision * 100).toFixed(2)}%
              </div>
            </div>
            
            <div className="metric-card">
              <div className="metric-label">Recall</div>
              <div className="metric-value">
                {(metrics.recall * 100).toFixed(2)}%
              </div>
            </div>
            
            <div className="metric-card">
              <div className="metric-label">F1 Score</div>
              <div className="metric-value">
                {(metrics.f1_score * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ModelInfo;
