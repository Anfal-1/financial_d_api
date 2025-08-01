from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
import os
from werkzeug.exceptions import BadRequest
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store loaded model and scaler
model = None
scaler = None
FRAUD_THRESHOLD = 0.01  # MSE threshold for fraud detection

def load_model_and_scaler():
    """Load the pre-trained model and scaler on startup"""
    global model, scaler
    
    try:
        # Load the autoencoder model
        model_path = os.path.join('model', 'autoencoder_model.h5')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Autoencoder model loaded successfully from {model_path}")
            logger.info(f"Model input shape: {model.input_shape}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            # Don't raise error to allow API to start without model for testing
        
        # Load the scaler
        scaler_path = os.path.join('model', 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"Scaler loaded successfully from {scaler_path}")
        else:
            logger.warning(f"Scaler file not found at {scaler_path}")
            
    except Exception as e:
        logger.error(f"Error loading model or scaler: {str(e)}")
        # Don't raise to allow API to start

def validate_input(data):
    """Validate the input data"""
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")
    
    if 'features' not in data:
        raise ValueError("Missing 'features' field in input")
    
    features = data['features']
    if not isinstance(features, list):
        raise ValueError("'features' must be a list")
    
    if len(features) == 0:
        raise ValueError("'features' list cannot be empty")
    
    # Check if all features are numeric
    for i, feature in enumerate(features):
        if not isinstance(feature, (int, float)) and feature is not None:
            try:
                # Try to convert to float
                features[i] = float(feature)
            except (ValueError, TypeError):
                raise ValueError(f"Feature at index {i} must be numeric, got: {type(feature).__name__}")
    
    # Check for None values
    if any(f is None for f in features):
        raise ValueError("Features cannot contain None values")
    
    return features

def compute_reconstruction_error(features):
    """Compute reconstruction error using the autoencoder"""
    try:
        # Convert to numpy array and reshape
        input_data = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Validate input dimensions
        expected_features = model.input_shape[1] if model.input_shape[1] else len(features)
        if input_data.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {input_data.shape[1]}")
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Get reconstruction from autoencoder
        reconstruction = model.predict(scaled_data, verbose=0)
        
        # Compute Mean Squared Error
        mse = np.mean(np.square(scaled_data - reconstruction))
        
        return float(mse)
        
    except Exception as e:
        logger.error(f"Error computing reconstruction error: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model and scaler are loaded
        if model is None:
            return jsonify({
                'error': 'Autoencoder model not loaded. Please check server configuration.',
                'details': 'Model file (autoencoder_model.h5) not found or failed to load'
            }), 500
            
        if scaler is None:
            return jsonify({
                'error': 'Scaler not loaded. Please check server configuration.',
                'details': 'Scaler file (scaler.pkl) not found or failed to load'
            }), 500
        
        # Check content type
        if not request.is_json:
            return jsonify({
                'error': 'Request must contain JSON data',
                'details': 'Content-Type must be application/json'
            }), 400
        
        # Get JSON data from request
        try:
            data = request.get_json()
        except Exception as e:
            return jsonify({
                'error': 'Invalid JSON format',
                'details': str(e)
            }), 400
        
        if data is None:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
        
        # Validate input
        features = validate_input(data)
        
        # Compute reconstruction error
        reconstruction_error = compute_reconstruction_error(features)
        
        # Determine if transaction is fraudulent
        is_fraud = reconstruction_error > FRAUD_THRESHOLD
        
        # Return response
        response = {
            'is_fraud': bool(is_fraud),
            'reconstruction_error': float(reconstruction_error)
        }
        
        logger.info(f"Prediction completed - Features: {len(features)}, MSE: {reconstruction_error:.6f}, Fraud: {is_fraud}")
        return jsonify(response), 200
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({
            'error': 'Invalid input data',
            'details': str(e)
        }), 400
    
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error occurred during prediction',
            'details': 'Please check server logs for more information'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'fraud_threshold': FRAUD_THRESHOLD,
            'tensorflow_version': tf.__version__
        }
        
        if model is not None:
            status['model_input_shape'] = model.input_shape
            status['model_output_shape'] = model.output_shape
            
        return jsonify(status), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    info = {
        'api_name': 'Fraud Detection API',
        'version': '1.0.0',
        'description': 'Autoencoder-based fraud detection using reconstruction error',
        'endpoints': {
            '/predict': {
                'method': 'POST',
                'description': 'Predict if transaction is fraudulent',
                'input': {
                    'features': 'List of numeric features'
                },
                'output': {
                    'is_fraud': 'Boolean indicating fraud',
                    'reconstruction_error': 'MSE reconstruction error'
                }
            },
            '/health': {
                'method': 'GET',
                'description': 'Health check endpoint'
            },
            '/info': {
                'method': 'GET',
                'description': 'API information'
            }
        },
        'fraud_threshold': FRAUD_THRESHOLD
    }
    return jsonify(info), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/predict', '/health', '/info']
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'details': 'Check the API documentation for allowed methods'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'details': 'Please check server logs'
    }), 500

# Initialize model and scaler on startup
try:
    load_model_and_scaler()
except Exception as e:
    logger.error(f"Failed to initialize: {str(e)}")

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
