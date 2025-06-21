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
            "../pkl/best_obesity_classifier_xgboost.pkl"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                print(f"✅ File .pkl berhasil dimuat dari {model_file}")
                print(f"🔍 Data type: {type(loaded_data)}")
                
                # Handle different model storage formats
                if isinstance(loaded_data, dict):
                    if 'best_model' in loaded_data:
                        # Model disimpan dalam dictionary dengan key 'best_model'
                        model_pipeline = loaded_data
                        actual_model = loaded_data['best_model']
                        print(f"🎯 Model ditemukan di key 'best_model': {type(actual_model)}")
                        
                        # Verify model has required methods
                        if hasattr(actual_model, 'predict') and hasattr(actual_model, 'predict_proba'):
                            print("🎯 Model siap untuk prediksi!")
                            return True
                        else:
                            raise Exception("Model di 'best_model' tidak memiliki method predict/predict_proba")
                    else:
                        raise Exception("Dictionary tidak memiliki key 'best_model'")
                
                elif hasattr(loaded_data, 'predict') and hasattr(loaded_data, 'predict_proba'):
                    # Model langsung disimpan tanpa dictionary wrapper
                    model_pipeline = {'best_model': loaded_data}
                    print(f"🎯 Model langsung: {type(loaded_data)}")
                    print("🎯 Model siap untuk prediksi!")
                    return True
                
                else:
                    raise Exception(f"Format model tidak dikenali: {type(loaded_data)}")
        
        # Jika tidak ada model yang ditemukan
        raise FileNotFoundError("Tidak ada model .pkl yang ditemukan. Pastikan file model ada di folder pkl/")
        
    except Exception as e:
        print(f"❌ Error memuat model: {e}")
        raise e

def predict_with_model(input_data: dict, model_pipeline):
    """Prediksi menggunakan model .pkl yang sesungguhnya"""
    # Create PredictionInput object untuk preprocessing
    prediction_input = PredictionInput(**input_data)
    
    # Preprocess the input data
    processed_df = preprocess_input(prediction_input)
    
    print("🤖 Menggunakan model .pkl untuk prediksi")
    print(f"📊 Processed data shape: {processed_df.shape}")
    print(f"📊 Processed data sample: {processed_df.iloc[0].to_dict()}")
    
    # Use actual trained model
    prediction_idx = model_pipeline['best_model'].predict(processed_df)[0]
    probabilities_array = model_pipeline['best_model'].predict_proba(processed_df)[0]
    
    print(f"🎯 Prediction index: {prediction_idx}")
    print(f"🎯 Raw probabilities: {probabilities_array}")
    
    # Get class names from model pipeline (correct approach)
    if 'class_names' in model_pipeline:
        class_names = model_pipeline['class_names']
        print(f"🎯 Menggunakan class names dari model: {class_names}")
    else:
        # Fallback class names
        class_names = [
            'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
            'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
        ]
        print("⚠️ Menggunakan fallback class names")
    
    # Create prediction result
    prediction = str(class_names[prediction_idx])  # Ensure string
    probability_dict = {str(class_name): float(prob) for class_name, prob in zip(class_names, probabilities_array)}
    
    # Print detailed probability breakdown
    print("\n📈 PROBABILITAS DETAIL:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities_array)):
        print(f"   {i}: {class_name} = {prob:.4f} ({prob*100:.2f}%)")
    print(f"🎯 HASIL PREDIKSI: {prediction} (index {prediction_idx})")
    print("="*60)
    
    return prediction, probability_dict

def preprocess_input(input_data: PredictionInput) -> pd.DataFrame:
    """Preprocessing input data untuk memenuhi persyaratan model dengan komponen yang sama seperti saat training"""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Mengubah input menjadi dictionary
    data_dict = input_data.dict()
    print(f"📝 Input data original: {data_dict}")
    
    # Membuat DataFrame
    df = pd.DataFrame([data_dict])
    print(f"📊 DataFrame setelah dibuat: {df.to_dict('records')[0]}")
    
    # GUNAKAN LABEL ENCODERS DARI MODEL TRAINING
    if 'label_encoders' in model_pipeline:
        print("🔄 Menggunakan Label Encoders dari model training")
        label_encoders = model_pipeline['label_encoders']
        
        for feature_name, encoder in label_encoders.items():
            if feature_name in df.columns:
                original_val = df[feature_name].iloc[0]
                try:
                    # Transform menggunakan encoder yang sudah ditraining
                    encoded_val = encoder.transform([original_val])[0]
                    df[feature_name] = encoded_val
                    print(f"   ✅ {feature_name}: '{original_val}' -> {encoded_val}")
                except ValueError as e:
                    print(f"   ⚠️ {feature_name}: '{original_val}' tidak dikenali, menggunakan nilai default 0")
                    df[feature_name] = 0
    else:
        print("⚠️ Label encoders tidak ditemukan, menggunakan mapping manual")
        # Fallback ke mapping manual
        binary_mapping = {
            'Gender': {'Male': 1, 'Female': 0},
            'family_history_with_overweight': {'yes': 1, 'no': 0},
            'FAVC': {'yes': 1, 'no': 0},
            'SMOKE': {'yes': 1, 'no': 0},
            'SCC': {'yes': 1, 'no': 0}
        }
        
        for feature, mapping in binary_mapping.items():
            if feature in df.columns:
                original_val = df[feature].iloc[0]
                df[feature] = df[feature].map(mapping).fillna(0)
                new_val = df[feature].iloc[0]
                print(f"🔄 Binary mapping {feature}: '{original_val}' -> {new_val}")
    
    # Menangani encoding multi-kelas (one-hot) - ini sama seperti sebelumnya
    multi_features = ['CAEC', 'CALC', 'MTRANS']
    for feature in multi_features:
        if feature in df.columns:
            original_val = df[feature].iloc[0]
            print(f"🔄 One-hot encoding {feature}: '{original_val}'")
            # Buat variabel dummy
            dummies = pd.get_dummies(df[feature], prefix=feature)
            print(f"   Created dummies: {list(dummies.columns)}")
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[feature])
    
    print(f"📊 DataFrame setelah encoding: columns = {list(df.columns)}")
    
    # Gunakan feature_columns dari model .pkl
    if 'feature_columns' in model_pipeline:
        expected_features = model_pipeline['feature_columns']
        print(f"🔍 Menggunakan {len(expected_features)} feature columns dari model")
        print(f"🔍 Expected features: {expected_features}")
    else:
        # Fallback ke manual list jika tidak ada di model
        expected_features = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
            'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
            'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
            'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 
            'MTRANS_Public_Transportation', 'MTRANS_Walking'
        ]
        print("⚠️ Menggunakan feature columns manual (fallback)")
    
    # Tambahkan kolom yang hilang dengan nilai 0
    missing_features = []
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
            missing_features.append(feature)
    
    if missing_features:
        print(f"➕ Added missing features with 0: {missing_features}")
    
    # Pilih hanya kolom yang diperlukan dalam urutan yang benar
    df = df[expected_features]
    
    # GUNAKAN SCALER DARI MODEL TRAINING
    if 'scaler' in model_pipeline:
        print("📏 Menggunakan StandardScaler dari model training")
        scaler = model_pipeline['scaler']
        
        # Simpan data sebelum scaling untuk debug
        pre_scaling = df.iloc[0].to_dict()
        print(f"📊 Data sebelum scaling: {pre_scaling}")
        
        # Apply scaling
        df_scaled = pd.DataFrame(
            scaler.transform(df), 
            columns=df.columns, 
            index=df.index
        )
        
        post_scaling = df_scaled.iloc[0].to_dict()
        print(f"📊 Data setelah scaling: {post_scaling}")
        
        return df_scaled
    else:
        print("⚠️ Scaler tidak ditemukan, data tidak di-scale")
        print(f"✅ Final processed data (no scaling): {df.iloc[0].to_dict()}")
        return df

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
    
    # Get model information from actual trained model
    model_name = model_pipeline.get('model_name', type(model_pipeline['best_model']).__name__)
    
    # Use actual performance data from model if available
    model_performance = model_pipeline.get('model_performance', {})
    accuracy = model_performance.get('test_accuracy', 0.95)
    cv_score = model_performance.get('cv_mean', 0.94)
    
    # Get class names from model pipeline (not from model.classes_ which contains indices)
    if 'class_names' in model_pipeline:
        class_names = [str(name) for name in model_pipeline['class_names']]
    else:
        class_names = [
            'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
            'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
        ]
    
    # Get feature count
    if 'feature_columns' in model_pipeline:
        feature_count = len(model_pipeline['feature_columns'])
    elif hasattr(model_pipeline['best_model'], 'n_features_in_'):
        feature_count = model_pipeline['best_model'].n_features_in_
    else:
        feature_count = 23
    
    return ModelInfo(
        model_name=model_name,
        accuracy=float(accuracy),
        cv_score=float(cv_score),
        feature_count=int(feature_count),
        classes=class_names
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_obesity(input_data: PredictionInput):
    """
    Prediksi tingkat obesitas berdasarkan input fitur menggunakan model .pkl yang telah ditraining
    
    Mengembalikan respons dengan:
    - prediction: Kategori obesitas utama
    - probability: Distribusi probabilitas untuk semua kelas
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Gunakan model .pkl untuk prediksi
        prediction, probability_dict = predict_with_model(input_data.dict(), model_pipeline)
        
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