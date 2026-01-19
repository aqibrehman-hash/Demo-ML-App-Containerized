"""
Flask API for ML Model Predictions
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables to store model artifacts
model = None
scaler = None
feature_names = None
metrics = None

def load_model_artifacts():
    """Load model, scaler, and metadata"""
    global model, scaler, feature_names, metrics
    
    # Auto-detect model directory
    import os
    model_dir = 'backend/models' if os.path.exists('backend/models') else 'models'
    
    try:
        # Load model
        model = joblib.load(f'{model_dir}/model.pkl')
        print("Model loaded successfully")
        
        # Load scaler
        scaler = joblib.load(f'{model_dir}/scaler.pkl')
        print("Scaler loaded successfully")
        
        # Load feature names
        with open(f'{model_dir}/feature_names.json', 'r') as f:
            feature_data = json.load(f)
            feature_names = feature_data['features']
        print(f"Feature names loaded: {len(feature_names)} features")
        
        # Load metrics
        if os.path.exists(f'{model_dir}/metrics.json'):
            with open(f'{model_dir}/metrics.json', 'r') as f:
                metrics = json.load(f)
            print("Metrics loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'ML Model API is running',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get data from request
        data = request.get_json()
        
        # Validate input
        if 'features' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing "features" in request body'
            }), 400
        
        features = data['features']
        
        # Validate feature count
        if len(features) != len(feature_names):
            return jsonify({
                'status': 'error',
                'message': f'Expected {len(feature_names)} features, got {len(features)}'
            }), 400
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = int(model.predict(X_scaled)[0])
        prediction_proba = model.predict_proba(X_scaled)[0].tolist()
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'prediction_label': f'Class {prediction}',
            'probability': {
                'class_0': round(prediction_proba[0], 4),
                'class_1': round(prediction_proba[1], 4)
            },
            'confidence': round(max(prediction_proba), 4)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information and metrics"""
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        info = {
            'status': 'success',
            'model_type': 'Random Forest Classifier',
            'n_features': len(feature_names) if feature_names else 0,
            'feature_names': feature_names,
            'metrics': metrics if metrics else {}
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/random-sample', methods=['GET'])
def random_sample():
    """Generate a random sample for testing"""
    try:
        if feature_names is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        # Generate random features
        n_features = len(feature_names)
        random_features = np.random.randn(n_features).tolist()
        
        return jsonify({
            'status': 'success',
            'features': random_features,
            'feature_names': feature_names
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("="*50)
    print("Starting Flask ML API Server")
    print("="*50)
    
    # Load model artifacts
    if load_model_artifacts():
        print("\nAll model artifacts loaded successfully!")
        print("Starting server on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nError: Could not load model artifacts.")
        print("Please run the ML pipeline first:")
        print("  python backend/data/generate_data.py")
        print("  python backend/ml_pipeline/train_model.py")
