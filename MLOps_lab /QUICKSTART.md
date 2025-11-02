# 🎯 QUICK START - MLOps FastAPI Lab

## ✅ What's Done

Both FastAPI applications are ready to run!

### 1. 🛋️ Furniture Price Prediction
- ✅ Model trained (Decision Tree Regressor)
- ✅ FastAPI application created (`main.py`)
- ✅ Jinja2 templates created
- ✅ REST API with validation

### 2. 🌸 Iris Flower Classification
- ✅ Using model from Flask_Lab
- ✅ FastAPI application created (`main_iris.py`)
- ✅ Jinja2 templates created
- ✅ REST API with validation

## 🚀 Running the Applications

### Furniture Price Prediction (Port 8000)

```bash
cd "MLOps_lab "
python main.py
```

Open: **http://localhost:8000**

### Iris Classification (Port 8001)

```bash
cd "MLOps_lab "
python main_iris.py
```

Open: **http://localhost:8001**

## 🧪 Quick Tests

### Test Furniture API
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"category": "Chairs", "sellable_online": "Yes", "other_colors": "No", "depth": 50.0, "height": 100.0, "width": 60.0}'
```

### Test Iris API
```bash
curl -X POST "http://localhost:8001/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## 📚 Interactive Documentation

- Furniture API Docs: http://localhost:8000/docs
- Iris API Docs: http://localhost:8001/docs

## 📝 Lab Assignment Completed

1. ✅ Notebook executed → model.pkl generated
2. ✅ FastAPI created with POST /predict endpoint
3. ✅ Jinja2 templates implemented for results
4. ✅ Same approach applied to Iris dataset

---

**All done! Both applications are working! 🎉**
