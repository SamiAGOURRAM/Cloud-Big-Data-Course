# 🚀 MLOps Lab - FastAPI ML Applications

This project contains two FastAPI applications demonstrating ML model deployment with Jinja2 templates.

## 📋 Projects Overview

### 1. 🛋️ Furniture Price Prediction API
Predicts furniture prices based on category, dimensions, and availability features using a Decision Tree Regressor.

### 2. 🌸 Iris Flower Classification API  
Classifies Iris flowers into species (Setosa, Versicolor, Virginica) using a Logistic Regression model.

## 🛠 Technologies

- **FastAPI** 0.104.1 - Modern Python web framework
- **Uvicorn** 0.24.0 - ASGI server
- **Jinja2** 3.1.2 - Template engine
- **Scikit-learn** 1.3.2 - Machine learning
- **Pydantic** 2.5.0 - Data validation

## 📁 Project Structure

```
MLOps_lab/
├── main.py                          # Furniture price prediction API
├── main_iris.py                     # Iris classification API
├── model.pkl                        # Trained furniture model
├── furniture.csv                    # Dataset
├── requirements.txt                 # Python dependencies
├── templates/                       # Furniture app templates
│   ├── index.html                  # Home page with form
│   ├── result.html                 # Prediction results
│   └── error.html                  # Error page
├── templates_iris/                  # Iris app templates
│   ├── index.html                  # Home page with form
│   ├── result.html                 # Classification results
│   └── error.html                  # Error page
└── Furniture prediction notebook.ipynb  # Model training notebook
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to MLOps_lab
cd "MLOps_lab "

# Install dependencies
pip install -r requirements.txt
```

### Running Applications

#### Furniture Price Prediction (Port 8000)

```bash
# Start the server
python main.py

# Or with uvicorn
uvicorn main:app --reload --port 8000
```

Access at: **http://localhost:8000**

#### Iris Flower Classification (Port 8001)

```bash
# Start the server
python main_iris.py

# Or with uvicorn
uvicorn main_iris:app --reload --port 8001
```

Access at: **http://localhost:8001**

## 💻 Usage

### Web Interface

1. **Open the app** in your browser
2. **Fill in the form** with required features
3. **Click "Predict"** button
4. **View results** with detailed prediction information

### API Endpoints

#### Furniture Price API

**Web Form**: `GET /` - Interactive HTML form

**Prediction**: `POST /predict` - Form submission

**JSON API**: `POST /api/predict`

Example curl request:
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Chairs",
    "sellable_online": "Yes",
    "other_colors": "No",
    "depth": 50.0,
    "height": 100.0,
    "width": 60.0
  }'
```

Response:
```json
{
  "predicted_price": 450.75,
  "input_features": {
    "category": "Chairs",
    "sellable_online": "Yes",
    "other_colors": "No",
    "depth": 50.0,
    "height": 100.0,
    "width": 60.0
  },
  "model_name": "Decision Tree Regressor"
}
```

#### Iris Classification API

**Web Form**: `GET /` - Interactive HTML form

**Prediction**: `POST /predict` - Form submission

**JSON API**: `POST /api/predict`

Example curl request:
```bash
curl -X POST "http://localhost:8001/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

Response:
```json
{
  "prediction": "setosa",
  "confidence": 99.87,
  "probabilities": {
    "setosa": 99.87,
    "versicolor": 0.13,
    "virginica": 0.00
  },
  "model_name": "Logistic Regression"
}
```

### Interactive API Documentation

FastAPI automatically generates interactive documentation:

- **Swagger UI**: http://localhost:8000/docs 
- **ReDoc**: http://localhost:8000/redoc 

## 📊 Model Information

### Furniture Price Prediction

- **Algorithm**: Decision Tree Regressor
- **Features**: 6 (category, sellable_online, other_colors, depth, height, width)
- **Train Score**: 86.81%
- **Test Score**: 62.73%
- **Target**: Price in SAR (Saudi Riyal)

**Categories**:
- Bar furniture, Beds, Bookcases & shelving units
- Cabinets & cupboards, Café furniture, Chairs
- Chests of drawers, Children's furniture, Nursery furniture
- Outdoor furniture, Room dividers, Sideboards
- Sofas & armchairs, TV & media furniture
- Tables & desks, Trolleys, Wardrobes

### Iris Classification

- **Algorithm**: Logistic Regression
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Test Accuracy**: 100%
- **Cross-Validation**: 95.83% (±4.56%)
- **Classes**: 3 (Setosa, Versicolor, Virginica)

## 🎨 Features

### Both Applications Include:

✅ **Web Interface**: Beautiful, responsive HTML forms
✅ **Jinja2 Templates**: Server-side rendering for results
✅ **REST API**: JSON endpoints for programmatic access
✅ **Input Validation**: Pydantic models for data validation
✅ **Error Handling**: Graceful error pages
✅ **Auto Documentation**: Swagger UI and ReDoc
✅ **Health Checks**: `/health` endpoints

## 🧪 Testing the Applications

### Test Furniture Prediction

**Example 1 - Chair**:
- Category: Chairs
- Sellable Online: Yes
- Other Colors: No
- Depth: 50 cm
- Height: 100 cm
- Width: 60 cm

**Example 2 - Table**:
- Category: Tables & desks
- Sellable Online: Yes
- Other Colors: Yes
- Depth: 80 cm
- Height: 75 cm
- Width: 120 cm



## 🎓 Learning Outcomes

- ✅ FastAPI framework basics
- ✅ Jinja2 template engine with FastAPI
- ✅ Pydantic data validation
- ✅ REST API design and documentation
- ✅ ML model deployment
- ✅ Model serialization with pickle
- ✅ Web form handling



**Built for Cloud & Big Data Course - MLOps Lab Assignment** 🚀
