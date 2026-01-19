import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ModelInfo from './components/ModelInfo';
import PredictionForm from './components/PredictionForm';
import PredictionResult from './components/PredictionResult';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [features, setFeatures] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/model-info`);
      if (response.data.status === 'success') {
        setModelInfo(response.data);
        // Initialize features array with zeros
        setFeatures(new Array(response.data.n_features).fill(0));
      }
    } catch (err) {
      setError('Failed to load model information. Make sure the Flask backend is running.');
      console.error('Error fetching model info:', err);
    }
  };

  const handleFeatureChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = parseFloat(value) || 0;
    setFeatures(newFeatures);
  };

  const handleRandomSample = async () => {
    try {
      const response = await axios.get(`${API_URL}/random-sample`);
      if (response.data.status === 'success') {
        setFeatures(response.data.features);
        setPrediction(null); // Clear previous prediction
      }
    } catch (err) {
      setError('Failed to generate random sample');
      console.error('Error generating random sample:', err);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        features: features
      });
      
      if (response.data.status === 'success') {
        setPrediction(response.data);
      } else {
        setError(response.data.message || 'Prediction failed');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to make prediction');
      console.error('Error making prediction:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFeatures(new Array(modelInfo.n_features).fill(0));
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 ML Prediction App</h1>
        <p>End-to-End Machine Learning Pipeline with Flask & React</p>
      </header>

      <div className="container">
        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        <div className="main-content">
          <div className="left-section">
            <ModelInfo modelInfo={modelInfo} />
          </div>

          <div className="right-section">
            <PredictionForm
              modelInfo={modelInfo}
              features={features}
              onFeatureChange={handleFeatureChange}
              onPredict={handlePredict}
              onRandomSample={handleRandomSample}
              onReset={handleReset}
              loading={loading}
            />

            {prediction && (
              <PredictionResult prediction={prediction} />
            )}
          </div>
        </div>
      </div>

      <footer className="App-footer">
        <p>Built for Docker Testing and Learning</p>
      </footer>
    </div>
  );
}

export default App;
