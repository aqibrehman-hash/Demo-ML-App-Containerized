import React from 'react';

const PredictionForm = ({ 
  modelInfo, 
  features, 
  onFeatureChange, 
  onPredict, 
  onRandomSample, 
  onReset,
  loading 
}) => {
  if (!modelInfo) {
    return (
      <div className="card">
        <h2>🔮 Make Predictions</h2>
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>🔮 Make Predictions</h2>
      
      <form onSubmit={onPredict}>
        <div className="form-group">
          <label>Enter Feature Values:</label>
          <div className="features-grid">
            {features.map((value, index) => (
              <div key={index} className="feature-input-group">
                <label>{modelInfo.feature_names[index]}</label>
                <input
                  type="number"
                  step="0.01"
                  value={value}
                  onChange={(e) => onFeatureChange(index, e.target.value)}
                  placeholder="0.00"
                />
              </div>
            ))}
          </div>
        </div>

        <div className="button-group">
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="loading-spinner"></span> Predicting...
              </>
            ) : (
              '🚀 Predict'
            )}
          </button>
          
          <button 
            type="button" 
            className="btn btn-outline"
            onClick={onRandomSample}
            disabled={loading}
          >
            🎲 Random Sample
          </button>
          
          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={onReset}
            disabled={loading}
          >
            🔄 Reset
          </button>
        </div>
      </form>
    </div>
  );
};

export default PredictionForm;
