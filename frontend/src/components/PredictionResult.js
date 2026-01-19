import React from 'react';

const PredictionResult = ({ prediction }) => {
  if (!prediction) return null;

  const { prediction_label, probability, confidence } = prediction;

  return (
    <div className="prediction-result">
      <h3>✨ Prediction Result</h3>
      
      <div className="prediction-main">
        <div className="prediction-class">{prediction_label}</div>
        <div className="confidence-badge">
          Confidence: {(confidence * 100).toFixed(2)}%
        </div>
      </div>

      <div className="probability-bars">
        <div className="probability-item">
          <div className="probability-label">
            <span>Class 0</span>
            <span>{(probability.class_0 * 100).toFixed(2)}%</span>
          </div>
          <div className="probability-bar">
            <div 
              className="probability-fill" 
              style={{ width: `${probability.class_0 * 100}%` }}
            ></div>
          </div>
        </div>

        <div className="probability-item">
          <div className="probability-label">
            <span>Class 1</span>
            <span>{(probability.class_1 * 100).toFixed(2)}%</span>
          </div>
          <div className="probability-bar">
            <div 
              className="probability-fill" 
              style={{ width: `${probability.class_1 * 100}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
