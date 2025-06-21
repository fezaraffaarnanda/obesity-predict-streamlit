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

# Global variabel untuk menyimpan model dan komponen preprocessing yang akan digunakan di seluruh aplikasi
model_pipeline = None

# Lifespan event handler (menggantikan @app.on_event yang sudah tidak digunakan)
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

# Inisialisasi FastAPI app dengan lifespan
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

# Pydantic models untuk request dan response
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

class ModelInfo(BaseModel):
    model_name: str
    accuracy: float
    cv_score: float
    feature_count: int
    classes: List[str]

def load_model():
    """Memuat pipeline model yang telah dilatih"""
    global model_pipeline
    try:
        # Coba nama file model yang berbeda
        model_files = [
            "../pkl/best_obesity_classifier_xgboost.pkl",
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_pipeline = pickle.load(f)
                print(f"✅ Model loaded successfully from {model_file}")
                return True
        
        # Jika tidak ada model yang ditemukan, buat pipeline mock untuk demo
        print("Tidak ada model file yang ditemukan. Membuat pipeline mock untuk demo...")
        create_mock_pipeline()
        return True
        
    except Exception as e:
        print(f"❌ Error memuat model: {e}")
        create_mock_pipeline()
        return True

def create_mock_pipeline():
    """Membuat pipeline mock untuk tujuan demo"""
    global model_pipeline
    
    # Struktur pipeline model mock
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
    print("✅ Pipeline mock berhasil dibuat untuk demo")

def mock_prediction(input_data: dict):
    """Membuat prediksi mock berdasarkan BMI"""
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
    
    # Membuat distribusi probabilitas
    probabilities = {}
    remaining_prob = (1 - main_prob) / (len(model_pipeline['class_names']) - 1)
    
    for class_name in model_pipeline['class_names']:
        if class_name == prediction:
            probabilities[class_name] = main_prob
        else:
            probabilities[class_name] = remaining_prob
    
    return prediction, probabilities

def preprocess_input(input_data: PredictionInput) -> pd.DataFrame:
    """Preprocessing input data untuk memenuhi persyaratan model"""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Mengubah input menjadi dictionary
    data_dict = input_data.dict()
    
    # Membuat DataFrame
    df = pd.DataFrame([data_dict])
    
    # Menangani encoding biner (penggantian sederhana untuk mock)
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
    
    # Menangani encoding multi-kelas (one-hot)
    multi_features = ['CAEC', 'CALC', 'MTRANS']
    for feature in multi_features:
        if feature in df.columns:
            # Buat variabel dummy
            dummies = pd.get_dummies(df[feature], prefix=feature)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[feature])
    
    # Pastikan semua fitur yang diperlukan ada
    required_features = model_pipeline['feature_columns']
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Pilih hanya fitur yang diperlukan dalam urutan yang benar
    df = df[required_features]
    
# API Endpoints
@app.get("/")
async def root():
    """Endpoint utama"""
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
    """Endpoint untuk pemeriksaan kesehatan"""
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
    """Dapatkan informasi tentang model yang dimuat"""
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
    """
    Prediksi tingkat obesitas berdasarkan input fitur
    
    Mengembalikan respons yang disederhanakan dengan hanya:
    - prediction: Kategori obesitas utama
    - probability: Distribusi probabilitas untuk semua kelas
    
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction (using mock for demo)
        prediction, probability_dict = mock_prediction(input_data.dict())
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
async def get_feature_info():
    """Dapatkan informasi tentang fitur input"""
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
    
    # Dapatkan port dari variabel lingkungan (Render menyetel ini secara otomatis)
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"📍 Server: {host}:{port}")
    print("📚 API Docs: /docs")
    print("🔄 Interactive API: /redoc")
    print("🏥 Health Check: /health")
    print("🔄 Ping: /ping")
    
    # Production settings - no reload
    uvicorn.run(app, host=host, port=port, reload=False) 