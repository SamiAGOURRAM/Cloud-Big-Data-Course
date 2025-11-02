"""
Flask Web Application for Iris Flower Classification
Predicts the species of Iris flower based on sepal and petal measurements
"""
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
import os

app = Flask(__name__)

# Load the trained model, scaler, and metadata
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html', 
                         feature_names=metadata['feature_names'],
                         model_name=metadata['best_model'],
                         accuracy=metadata['accuracy'])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get prediction details
        predicted_species = metadata['target_names'][prediction]
        confidence = prediction_proba[prediction] * 100
        
        # Create probability dictionary
        probabilities = {
            species: round(prob * 100, 2) 
            for species, prob in zip(metadata['target_names'], prediction_proba)
        }
        
        return render_template('result.html',
                             prediction=predicted_species,
                             confidence=round(confidence, 2),
                             probabilities=probabilities,
                             input_features={
                                 'Sepal Length': sepal_length,
                                 'Sepal Width': sepal_width,
                                 'Petal Length': petal_length,
                                 'Petal Width': petal_width
                             })
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (returns JSON)"""
    try:
        data = request.get_json()
        
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        predicted_species = metadata['target_names'][prediction]
        
        probabilities = {
            species: round(prob * 100, 2) 
            for species, prob in zip(metadata['target_names'], prediction_proba)
        }
        
        return jsonify({
            'prediction': predicted_species,
            'confidence': round(prediction_proba[prediction] * 100, 2),
            'probabilities': probabilities,
            'model': metadata['best_model']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html', metadata=metadata)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
