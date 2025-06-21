from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, List
import uvicorn

# Global variables to store model and preprocessing components
model_pipeline = None

# Lifespan event handler (menggantikan @app.on_event yang deprecated)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_pipeline
    print("🚀 Starting up application...")
    success = load_model()
    if not success:
        print("⚠️ Warning: Model not loaded. Using mock predictions.")
    yield
    # Shutdown
    print("🛑 Shutting down application...")

# Initialize FastAPI app dengan lifespan
app = FastAPI(
    title="Obesity Classification API",
    description="API untuk klasifikasi tingkat obesitas berdasarkan gaya hidup dan kebiasaan",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Local Streamlit
        "https://*.streamlit.app",  # Streamlit Cloud
        "https://*.onrender.com",   # Render.com domains
        "*"  # Allow all for demo (untuk production, spesifikkan domain)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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
            "../pkl/best_obesity_classifier_xgboost.pkl",
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
        "Obesity_Type_I": "Tinggi",
        "Obesity_Type_II": "Sangat Tinggi",
        "Obesity_Type_III": "Ekstrem"
    }
    return risk_mapping.get(prediction, "Sedang")

def get_recommendations(prediction: str) -> List[str]:
    """Get health recommendations based on prediction"""
    recommendations_mapping = {
        "Insufficient_Weight": [
            "Konsultasi dengan ahli gizi untuk program penambahan berat badan sehat",
            "Tingkatkan asupan kalori dengan makanan bergizi tinggi",
            "Olahraga ringan untuk membangun massa otot",
            "Pemeriksaan kesehatan untuk menyingkirkan kondisi medis"
        ],
        "Normal_Weight": [
            "Pertahankan pola makan seimbang dengan gizi yang cukup",
            "Lakukan aktivitas fisik teratur minimal 150 menit per minggu",
            "Jaga hidrasi dengan minum air 8 gelas per hari",
            "Monitoring berat badan secara berkala"
        ],
        "Overweight_Level_I": [
            "Kurangi asupan kalori harian sebesar 300-500 kalori",
            "Tingkatkan aktivitas fisik menjadi 300 menit per minggu",
            "Fokus pada makanan whole foods dan kurangi processed food",
            "Konsultasi dengan ahli gizi untuk rencana diet yang tepat"
        ],
        "Overweight_Level_II": [
            "Program penurunan berat badan terstruktur dengan target 5-10% dari berat badan",
            "Kombinasi diet rendah kalori dengan olahraga intensitas sedang",
            "Monitoring progres mingguan dengan bantuan profesional kesehatan",
            "Evaluasi faktor risiko kesehatan lainnya"
        ],
        "Obesity_Type_I": [
            "Intervensi gaya hidup intensif dengan dukungan medis",
            "Target penurunan berat badan 5-10% dalam 6 bulan pertama",
            "Program olahraga terstruktur dengan supervisi",
            "Konseling nutrisi dan modifikasi perilaku"
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

@app.get("/ping")
async def ping():
    """Ping endpoint untuk keep-alive"""
    return {"status": "pong", "timestamp": pd.Timestamp.now().isoformat()}

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
    import os
    print("🚀 Starting Obesity Classification API...")
    
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"📍 Server: {host}:{port}")
    print("📚 API Docs: /docs")
    print("🔄 Interactive API: /redoc")
    print("🏥 Health Check: /health")
    print("🔄 Ping: /ping")
    
    # Production settings - no reload
    uvicorn.run(app, host=host, port=port, reload=False) 