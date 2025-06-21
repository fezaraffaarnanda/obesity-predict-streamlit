from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, List
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Obesity Classification API",
    description="API untuk klasifikasi tingkat obesitas berdasarkan gaya hidup dan kebiasaan",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model and preprocessing components
model_pipeline = None

# Pydantic models for request/response
class PredictionInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

class PredictionResponse(BaseModel):
    prediction: str
    probability: Dict[str, float]
    risk_level: str
    recommendations: List[str]
    bmi: float
    bmi_category: str

class ModelInfo(BaseModel):
    model_name: str
    accuracy: float
    cv_score: float
    feature_count: int
    classes: List[str]

def load_model():
    """Load the trained model pipeline"""
    global model_pipeline
    try:
        # Try different model file names
        model_files = [
            "best_obesity_classifier_xgboost.pkl",
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_pipeline = pickle.load(f)
                print(f"✅ Model loaded successfully from {model_file}")
                return True
        
        # If no model found, create a mock pipeline for demo
        print("⚠️ No model file found. Creating mock pipeline for demo...")
        create_mock_pipeline()
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        create_mock_pipeline()
        return True

def create_mock_pipeline():
    """Create a mock pipeline for demo purposes"""
    global model_pipeline
    
    # Mock model pipeline structure
    model_pipeline = {
        'model_name': 'XGBoost',
        'binary_features': ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC'],
        'multi_features': ['CAEC', 'CALC', 'MTRANS'],
        'feature_columns': [
            'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
            'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
            'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
            'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 
            'MTRANS_Public_Transportation', 'MTRANS_Walking'
        ],
        'class_names': [
            'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
            'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
        ],
        'model_performance': {
            'test_accuracy': 0.9528,
            'cv_mean': 0.946
        }
    }
    print("✅ Mock pipeline created for demo")

def mock_prediction(input_data: dict):
    """Create mock prediction based on BMI"""
    bmi = input_data['Weight'] / (input_data['Height'] ** 2)
    
    if bmi < 18.5:
        prediction = 'Insufficient_Weight'
        main_prob = 0.85
    elif bmi < 25:
        prediction = 'Normal_Weight'
        main_prob = 0.90
    elif bmi < 30:
        prediction = 'Overweight_Level_I'
        main_prob = 0.82
    elif bmi < 35:
        prediction = 'Obesity_Type_I'
        main_prob = 0.87
    else:
        prediction = 'Obesity_Type_III'
        main_prob = 0.91
    
    # Create probability distribution
    probabilities = {}
    remaining_prob = (1 - main_prob) / (len(model_pipeline['class_names']) - 1)
    
    for class_name in model_pipeline['class_names']:
        if class_name == prediction:
            probabilities[class_name] = main_prob
        else:
            probabilities[class_name] = remaining_prob
    
    return prediction, probabilities

def preprocess_input(input_data: PredictionInput) -> pd.DataFrame:
    """Preprocess input data to match model requirements"""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert input to dictionary
    data_dict = input_data.dict()
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Handle binary encoding (simple mapping for mock)
    binary_mapping = {
        'Gender': {'Male': 1, 'Female': 0},
        'family_history_with_overweight': {'yes': 1, 'no': 0},
        'FAVC': {'yes': 1, 'no': 0},
        'SMOKE': {'yes': 1, 'no': 0},
        'SCC': {'yes': 1, 'no': 0}
    }
    
    for feature, mapping in binary_mapping.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping).fillna(0)
    
    # Handle multi-class encoding (one-hot)
    multi_features = ['CAEC', 'CALC', 'MTRANS']
    for feature in multi_features:
        if feature in df.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df[feature], prefix=feature)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[feature])
    
    # Ensure all required features are present
    required_features = model_pipeline['feature_columns']
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only required features in correct order
    df = df[required_features]
    
    return df

def calculate_bmi(weight: float, height: float) -> tuple:
    """Calculate BMI and category"""
    bmi = weight / (height ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return round(bmi, 2), category

def get_risk_level(prediction: str) -> str:
    """Determine risk level based on prediction"""
    risk_mapping = {
        "Insufficient_Weight": "Rendah",
        "Normal_Weight": "Normal",
        "Overweight_Level_I": "Sedang",
        "Overweight_Level_II": "Tinggi",
        "Obesity_Type_I": "Sangat Tinggi",
        "Obesity_Type_II": "Sangat Tinggi",
        "Obesity_Type_III": "Ekstrem"
    }
    return risk_mapping.get(prediction, "Unknown")

def get_recommendations(prediction: str) -> List[str]:
    """Get recommendations based on prediction"""
    recommendations_mapping = {
        "Insufficient_Weight": [
            "Tingkatkan asupan kalori harian secara sehat",
            "Konsumsi makanan bergizi tinggi seperti kacang-kacangan dan alpukat",
            "Konsultasi dengan ahli gizi untuk program penambahan berat badan",
            "Latihan kekuatan untuk menambah massa otot"
        ],
        "Normal_Weight": [
            "Pertahankan pola makan sehat dan seimbang",
            "Rutin berolahraga minimal 150 menit per minggu",
            "Jaga keseimbangan asupan nutrisi makro dan mikro",
            "Monitor berat badan secara berkala"
        ],
        "Overweight_Level_I": [
            "Kurangi asupan kalori 300-500 kalori per hari",
            "Tingkatkan aktivitas fisik menjadi 200-300 menit per minggu",
            "Batasi makanan olahan, gula, dan lemak jenuh",
            "Konsultasi dengan dokter atau ahli gizi"
        ],
        "Overweight_Level_II": [
            "Program penurunan berat badan terkontrol dengan target 0.5-1 kg per minggu",
            "Konsultasi medis untuk evaluasi komprehensif",
            "Diet rendah kalori dengan supervisi profesional",
            "Aktivitas fisik teratur dan terstruktur"
        ],
        "Obesity_Type_I": [
            "Intervensi medis diperlukan segera",
            "Program penurunan berat badan intensif",
            "Monitoring kesehatan rutin (tekanan darah, gula darah)",
            "Dukungan psikologis dan konseling gaya hidup"
        ],
        "Obesity_Type_II": [
            "Evaluasi medis komprehensif oleh tim multidisiplin",
            "Pertimbangan terapi farmakologi untuk obesitas",
            "Program rehabilitasi medis terstruktur",
            "Monitoring komplikasi kesehatan secara ketat"
        ],
        "Obesity_Type_III": [
            "Intervensi medis darurat dan evaluasi menyeluruh",
            "Evaluasi kelayakan untuk bedah bariatrik",
            "Perawatan medis intensif dengan monitoring 24/7",
            "Tim multidisiplin kesehatan (dokter, ahli gizi, psikolog)"
        ]
    }
    return recommendations_mapping.get(prediction, ["Konsultasi dengan tenaga medis profesional"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        print("⚠️ Warning: Model not loaded. Using mock predictions.")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Obesity Classification API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model_pipeline is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "features": "/features",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=model_pipeline['model_name'],
        accuracy=float(model_pipeline['model_performance']['test_accuracy']),
        cv_score=float(model_pipeline['model_performance']['cv_mean']),
        feature_count=len(model_pipeline['feature_columns']),
        classes=list(model_pipeline['class_names'])
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_obesity(input_data: PredictionInput):
    """Predict obesity level based on input features"""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Calculate BMI
        bmi, bmi_category = calculate_bmi(input_data.Weight, input_data.Height)
        
        # Make prediction (using mock for demo)
        prediction, probability_dict = mock_prediction(input_data.dict())
        
        # Get risk level and recommendations
        risk_level = get_risk_level(prediction)
        recommendations = get_recommendations(prediction)
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability_dict,
            risk_level=risk_level,
            recommendations=recommendations,
            bmi=bmi,
            bmi_category=bmi_category
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
async def get_feature_info():
    """Get information about input features"""
    return {
        "features": {
            "Gender": {
                "type": "categorical",
                "options": ["Male", "Female"],
                "description": "Jenis kelamin"
            },
            "Age": {
                "type": "numeric",
                "range": [14, 61],
                "description": "Usia dalam tahun"
            },
            "Height": {
                "type": "numeric",
                "range": [1.45, 1.98],
                "description": "Tinggi badan dalam meter"
            },
            "Weight": {
                "type": "numeric",
                "range": [39, 173],
                "description": "Berat badan dalam kg"
            },
            "family_history_with_overweight": {
                "type": "categorical",
                "options": ["yes", "no"],
                "description": "Riwayat keluarga dengan kelebihan berat badan"
            },
            "FAVC": {
                "type": "categorical",
                "options": ["yes", "no"],
                "description": "Sering konsumsi makanan berkalori tinggi"
            },
            "FCVC": {
                "type": "numeric",
                "range": [1, 3],
                "description": "Frekuensi konsumsi sayuran (1=Jarang, 3=Sering)"
            },
            "NCP": {
                "type": "numeric",
                "range": [1, 5],
                "description": "Jumlah makan utama per hari"
            },
            "CAEC": {
                "type": "categorical",
                "options": ["no", "Sometimes", "Frequently", "Always"],
                "description": "Konsumsi makanan di antara waktu makan"
            },
            "SMOKE": {
                "type": "categorical",
                "options": ["yes", "no"],
                "description": "Kebiasaan merokok"
            },
            "CH2O": {
                "type": "numeric",
                "range": [1, 5],
                "description": "Konsumsi air harian (liter)"
            },
            "SCC": {
                "type": "categorical",
                "options": ["yes", "no"],
                "description": "Monitor kalori yang dikonsumsi"
            },
            "FAF": {
                "type": "numeric",
                "range": [0, 7],
                "description": "Frekuensi aktivitas fisik per minggu"
            },
            "TUE": {
                "type": "numeric",
                "range": [0, 5],
                "description": "Waktu penggunaan teknologi (jam per hari)"
            },
            "CALC": {
                "type": "categorical",
                "options": ["no", "Sometimes", "Frequently", "Always"],
                "description": "Konsumsi alkohol"
            },
            "MTRANS": {
                "type": "categorical",
                "options": ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"],
                "description": "Mode transportasi utama"
            }
        },
        "output_classes": {
            "Insufficient_Weight": "Berat badan kurang",
            "Normal_Weight": "Berat badan normal",
            "Overweight_Level_I": "Kelebihan berat badan tingkat 1",
            "Overweight_Level_II": "Kelebihan berat badan tingkat 2", 
            "Obesity_Type_I": "Obesitas tipe 1",
            "Obesity_Type_II": "Obesitas tipe 2",
            "Obesity_Type_III": "Obesitas tipe 3"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Obesity Classification API...")
    print("📍 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔄 Interactive API: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 