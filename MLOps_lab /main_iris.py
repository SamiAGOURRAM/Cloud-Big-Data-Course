"""
FastAPI Application for Iris Flower Classification
Uses the trained model from Flask_Lab to predict Iris species
"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pickle
import numpy as np
import json
import sys
import os

# Add Flask_Lab to path to access the model
sys.path.append('../Flask_Lab')

# Initialize FastAPI app
app = FastAPI(
    title="Iris Flower Classification API",
    description="Predict Iris flower species using ML model",
    version="1.0.0"
)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates_iris")

# Load the trained model from Flask_Lab
model_path = '../Flask_Lab/model.pkl'
scaler_path = '../Flask_Lab/scaler.pkl'
metadata_path = '../Flask_Lab/model_metadata.json'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
except FileNotFoundError:
    print("Warning: Model files not found. Please ensure Flask_Lab models are available.")
    model = None
    scaler = None
    metadata = {
        'feature_names': ['sepal length (cm)', 'sepal width (cm)', 
                         'petal length (cm)', 'petal width (cm)'],
        'target_names': ['setosa', 'versicolor', 'virginica'],
        'best_model': 'Logistic Regression',
        'accuracy': 1.0
    }


# Pydantic models
class IrisInput(BaseModel):
    """Input schema for Iris flower classification"""
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0, le=10)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0, le=10)
    petal_length: float = Field(..., description="Petal length in cm", ge=0, le=10)
    petal_width: float = Field(..., description="Petal width in cm", ge=0, le=10)

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class IrisOutput(BaseModel):
    """Output schema for prediction result"""
    prediction: str = Field(..., description="Predicted species")
    confidence: float = Field(..., description="Confidence score in percentage")
    probabilities: dict = Field(..., description="Probability for each species")
    model_name: str = Field(..., description="Model used")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page with prediction form"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "feature_names": metadata['feature_names'],
        "model_name": metadata['best_model'],
        "accuracy": metadata['accuracy']
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    """Handle form submission and return prediction result page"""
    try:
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
        
        result = {
            "prediction": predicted_species,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "result": result
        })
    
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })


@app.post("/api/predict", response_model=IrisOutput)
async def predict_api(iris: IrisInput):
    """API endpoint for Iris classification (returns JSON)"""
    try:
        # Create feature array
        features = np.array([[
            iris.sepal_length,
            iris.sepal_width,
            iris.petal_length,
            iris.petal_width
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        predicted_species = metadata['target_names'][prediction]
        
        probabilities = {
            species: round(prob * 100, 2) 
            for species, prob in zip(metadata['target_names'], prediction_proba)
        }
        
        return IrisOutput(
            prediction=predicted_species,
            confidence=round(prediction_proba[prediction] * 100, 2),
            probabilities=probabilities,
            model_name=metadata['best_model']
        )
    
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": metadata.get('best_model', 'Unknown')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
