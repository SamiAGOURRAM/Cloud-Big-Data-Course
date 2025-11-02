"""
FastAPI Application for Furniture Price Prediction
Uses Decision Tree Regressor to predict furniture prices based on features
"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pickle
import numpy as np
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Furniture Price Prediction API",
    description="Predict furniture prices using ML model",
    version="1.0.0"
)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Category mappings (from the notebook)
CATEGORY_MAPPING = {
    'Bar furniture': 0, 'Beds': 1, 'Bookcases & shelving units': 2,
    'Cabinets & cupboards': 3, 'Café furniture': 4, 'Chairs': 5,
    'Chests of drawers & drawer units': 6, "Children's furniture": 7,
    'Nursery furniture': 8, 'Outdoor furniture': 9, 'Room dividers': 10,
    'Sideboards, buffets & console tables': 11, 'Sofas & armchairs': 12,
    'TV & media furniture': 13, 'Tables & desks': 14, 'Trolleys': 15,
    'Wardrobes': 16
}

SELLABLE_ONLINE_MAPPING = {'No': 0, 'Yes': 1}
OTHER_COLORS_MAPPING = {'No': 0, 'Yes': 1}


# Pydantic models for request/response
class FurnitureInput(BaseModel):
    """Input schema for furniture price prediction"""
    category: str = Field(..., description="Furniture category")
    sellable_online: str = Field(..., description="Is it sellable online? (Yes/No)")
    other_colors: str = Field(..., description="Are other colors available? (Yes/No)")
    depth: float = Field(..., description="Depth in cm", ge=0)
    height: float = Field(..., description="Height in cm", ge=0)
    width: float = Field(..., description="Width in cm", ge=0)

    class Config:
        schema_extra = {
            "example": {
                "category": "Chairs",
                "sellable_online": "Yes",
                "other_colors": "No",
                "depth": 50.0,
                "height": 100.0,
                "width": 60.0
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction result"""
    predicted_price: float = Field(..., description="Predicted price in SAR")
    input_features: dict = Field(..., description="Input features used for prediction")
    model_name: str = Field(default="Decision Tree Regressor", description="Model used")


# Helper function to prepare features
def prepare_features(category: str, sellable_online: str, other_colors: str, 
                     depth: float, height: float, width: float) -> np.ndarray:
    """Convert input features to model-ready format"""
    category_encoded = CATEGORY_MAPPING.get(category, 0)
    sellable_encoded = SELLABLE_ONLINE_MAPPING.get(sellable_online, 0)
    colors_encoded = OTHER_COLORS_MAPPING.get(other_colors, 0)
    
    features = np.array([[category_encoded, sellable_encoded, colors_encoded, 
                         depth, height, width]])
    return features


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page with prediction form"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "categories": list(CATEGORY_MAPPING.keys()),
        "title": "Furniture Price Predictor"
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    category: str = Form(...),
    sellable_online: str = Form(...),
    other_colors: str = Form(...),
    depth: float = Form(...),
    height: float = Form(...),
    width: float = Form(...)
):
    """Handle form submission and return prediction result page"""
    try:
        # Prepare features
        features = prepare_features(category, sellable_online, other_colors, 
                                    depth, height, width)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Prepare response
        result = {
            "predicted_price": round(prediction, 2),
            "category": category,
            "sellable_online": sellable_online,
            "other_colors": other_colors,
            "depth": depth,
            "height": height,
            "width": width
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


@app.post("/api/predict", response_model=PredictionOutput)
async def predict_api(furniture: FurnitureInput):
    """API endpoint for price prediction (returns JSON)"""
    try:
        # Prepare features
        features = prepare_features(
            furniture.category, 
            furniture.sellable_online,
            furniture.other_colors,
            furniture.depth,
            furniture.height,
            furniture.width
        )
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return PredictionOutput(
            predicted_price=round(prediction, 2),
            input_features={
                "category": furniture.category,
                "sellable_online": furniture.sellable_online,
                "other_colors": furniture.other_colors,
                "depth": furniture.depth,
                "height": furniture.height,
                "width": furniture.width
            }
        )
    
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/categories")
async def get_categories():
    """Get list of available categories"""
    return {"categories": list(CATEGORY_MAPPING.keys())}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
