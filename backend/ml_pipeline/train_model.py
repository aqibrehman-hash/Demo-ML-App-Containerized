"""
ML Pipeline - Train and evaluate model
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

class MLPipeline:
    def __init__(self, data_path=None):
        # Auto-detect path based on directory structure
        import os
        if data_path is None:
            if os.path.exists('backend/data/dataset.csv'):
                data_path = 'backend/data/dataset.csv'
            else:
                data_path = 'data/dataset.csv'
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_data(self):
        """Load dataset from CSV"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2):
        """Split and scale the data"""
        print("Preprocessing data...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("Training model...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred))
        }
        
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n=== Model Performance ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")
        
        return metrics
    
    def save_model(self, model_dir=None):
        """Save trained model and scaler"""
        print("\nSaving model artifacts...")
        
        # Auto-detect model directory
        if model_dir is None:
            model_dir = 'backend/models' if os.path.exists('backend') else 'models'
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f'{model_dir}/model.pkl')
        print(f"Model saved to {model_dir}/model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        print(f"Scaler saved to {model_dir}/scaler.pkl")
        
        # Save feature names
        with open(f'{model_dir}/feature_names.json', 'w') as f:
            json.dump({'features': self.feature_names}, f)
        print(f"Feature names saved to {model_dir}/feature_names.json")
        
        return model_dir
    
    def save_metrics(self, metrics, metrics_path=None):
        """Save evaluation metrics"""
        if metrics_path is None:
            metrics_path = 'backend/models/metrics.json' if os.path.exists('backend') else 'models/metrics.json'
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
    
    def run_pipeline(self):
        """Execute complete ML pipeline"""
        print("="*50)
        print("Starting ML Pipeline")
        print("="*50)
        
        # Load data
        X, y = self.load_data()
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Preprocess
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set:  {X_test.shape[0]} samples")
        
        # Train
        self.train_model(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(X_test, y_test)
        
        # Save everything
        self.save_model()
        self.save_metrics(metrics)
        
        print("\n" + "="*50)
        print("Pipeline completed successfully!")
        print("="*50)
        
        return metrics

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.run_pipeline()
