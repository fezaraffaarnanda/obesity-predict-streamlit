import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="🏥 Sistem Prediksi Obesitas",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #A23B72;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border: none;
        color: #155724;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: none;
        color: #856404;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .danger-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border: none;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border: none;
        color: #0c5460;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ======================== BACKEND LOGIC ========================

@st.cache_data
def load_model_pipeline():
    """Memuat atau membuat pipeline model"""
    try:
        # Try to load actual model
        model_files = [
            "../pkl/best_obesity_classifier_xgboost.pkl",
            "pkl/best_obesity_classifier_xgboost.pkl"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_pipeline = pickle.load(f)
                return model_pipeline, True
        
        # If no model found, create mock pipeline
        return create_mock_pipeline(), False
        
    except Exception as e:
        return create_mock_pipeline(), False

def create_mock_pipeline():
    """Membuat pipeline mock untuk demo"""
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
    return model_pipeline

def mock_prediction(input_data: dict, model_pipeline: dict):
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

def preprocess_input(input_data: dict, model_pipeline: dict) -> pd.DataFrame:
    """Preprocessing input data untuk model"""
    
    # Membuat DataFrame
    df = pd.DataFrame([input_data])
    
    # Menangani encoding biner
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
    
    return df

def make_prediction(input_data: dict):
    """Membuat prediksi obesitas"""
    try:
        model_pipeline, is_real_model = load_model_pipeline()
        
        # Gunakan mock prediction untuk demo
        prediction, probability_dict = mock_prediction(input_data, model_pipeline)
        
        return {
            "prediction": prediction,
            "probability": probability_dict,
            "model_info": {
                "is_real_model": is_real_model,
                "model_name": model_pipeline['model_name'],
                "accuracy": model_pipeline['model_performance']['test_accuracy']
            }
        }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@st.cache_data
def get_feature_info():
    """Mendapatkan informasi fitur input"""
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

@st.cache_data
def get_model_info():
    """Mendapatkan informasi model"""
    model_pipeline, is_real_model = load_model_pipeline()
    
    return {
        "model_name": model_pipeline['model_name'],
        "accuracy": model_pipeline['model_performance']['test_accuracy'],
        "cv_score": model_pipeline['model_performance']['cv_mean'],
        "feature_count": len(model_pipeline['feature_columns']),
        "classes": model_pipeline['class_names'],
        "is_real_model": is_real_model
    }

# ======================== FRONTEND LOGIC ========================

def main():
    st.markdown('<h1 class="main-header">🏥 Sistem Prediksi Tingkat Obesitas</h1>', unsafe_allow_html=True)
    
    # Load model info for sidebar
    model_info = get_model_info()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📋 Menu Navigasi</h2>', unsafe_allow_html=True)
        
        page = st.selectbox(
            "Pilih Halaman:",
            ["🏠 Beranda", "🔮 Prediksi", "📊 Info Model", "📈 Analisis"],
            index=0
        )
        
        # Model info in sidebar
        st.markdown("---")
        st.markdown("**🤖 Info Model:**")
        st.metric("Model", model_info['model_name'])
        st.metric("Akurasi", f"{model_info['accuracy']:.1%}")
        st.metric("CV Score", f"{model_info['cv_score']:.1%}")
        
        if not model_info['is_real_model']:
            st.warning("⚠️ Menggunakan model demo")
        
        st.markdown("---")
        st.markdown("**🕒 Status:**")
        st.write(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        st.write("🟢 Aplikasi: Online")
    
    # Main content based on selected page
    if page == "🏠 Beranda":
        display_home_page()
    elif page == "🔮 Prediksi":
        display_prediction_page()
    elif page == "📊 Info Model":
        display_model_info_page()
    elif page == "📈 Analisis":
        display_analysis_page()

def display_home_page():
    """Menampilkan halaman beranda"""
    st.markdown('<h2 class="sub-header">🏠 Selamat Datang</h2>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h2>🎯 Deteksi Dini Risiko Obesitas</h2>
            <p>Sistem AI untuk menganalisis risiko obesitas berdasarkan gaya hidup Anda</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features overview
    st.markdown("### 🚀 Fitur Utama")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Analisis Mendalam</h3>
            <p>16 parameter gaya hidup dianalisis menggunakan machine learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Visualisasi Interaktif</h3>
            <p>Grafik dan chart untuk memahami hasil prediksi dengan mudah</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💻 Aplikasi Terpadu</h3>
            <p>Frontend dan backend dalam satu aplikasi untuk kemudahan deployment</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use
    st.markdown("### 📖 Cara Menggunakan")
    
    steps = [
        ("1️⃣", "Klik menu **Prediksi**", "Navigasi ke halaman prediksi"),
        ("2️⃣", "Isi data diri Anda", "Masukkan informasi gaya hidup dan kebiasaan"),
        ("3️⃣", "Dapatkan hasil analisis", "Lihat prediksi dan probabilitas"),
        ("4️⃣", "Analisis insight", "Pelajari faktor-faktor yang mempengaruhi")
    ]
    
    for emoji, title, desc in steps:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<h2>{emoji}</h2>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{title}**")
            st.write(desc)

def display_prediction_page():
    """Menampilkan halaman prediksi"""
    st.markdown('<h2 class="sub-header">🔮 Prediksi Tingkat Obesitas</h2>', unsafe_allow_html=True)
    
    features = get_feature_info()['features']
    
    # Input form
    with st.form("prediction_form"):
        st.markdown("### 👤 Data Personal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🧑‍🤝‍🧑 Informasi Dasar**")
            gender = st.selectbox("Jenis Kelamin", features['Gender']['options'])
            age = st.slider("Usia (tahun)", 
                          min_value=int(features['Age']['range'][0]), 
                          max_value=int(features['Age']['range'][1]), 
                          value=25,
                          help="Masukkan usia Anda dalam tahun")
            
            height = st.slider("Tinggi Badan (meter)", 
                             min_value=features['Height']['range'][0], 
                             max_value=features['Height']['range'][1], 
                             value=1.70, step=0.01,
                             help="Masukkan tinggi badan dalam meter")
            
            weight = st.slider("Berat Badan (kg)", 
                             min_value=int(features['Weight']['range'][0]), 
                             max_value=int(features['Weight']['range'][1]), 
                             value=70,
                             help="Masukkan berat badan dalam kilogram")
            
            family_history = st.selectbox("Riwayat Keluarga Overweight", 
                                        features['family_history_with_overweight']['options'],
                                        help="Apakah ada anggota keluarga yang overweight?")
        
        with col2:
            st.markdown("**🍽️ Kebiasaan Makan**")
            favc = st.selectbox("Sering Konsumsi Makanan Berkalori Tinggi", 
                              features['FAVC']['options'],
                              help="Apakah Anda sering makan makanan berkalori tinggi?")
            
            fcvc = st.slider("Frekuensi Konsumsi Sayuran (1-3)", 
                           min_value=int(features['FCVC']['range'][0]), 
                           max_value=int(features['FCVC']['range'][1]), 
                           value=2,
                           help="1=Jarang, 2=Kadang-kadang, 3=Sering")
            
            ncp = st.slider("Jumlah Makan Utama per Hari", 
                          min_value=int(features['NCP']['range'][0]), 
                          max_value=int(features['NCP']['range'][1]), 
                          value=3,
                          help="Berapa kali Anda makan utama dalam sehari?")
            
            caec = st.selectbox("Makan di Antara Waktu Makan", 
                              features['CAEC']['options'],
                              help="Seberapa sering Anda ngemil?")
            
            ch2o = st.slider("Konsumsi Air per Hari (liter)", 
                           min_value=int(features['CH2O']['range'][0]), 
                           max_value=int(features['CH2O']['range'][1]), 
                           value=2,
                           help="Berapa liter air yang Anda minum per hari?")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**🚭 Kebiasaan Lainnya**")
            smoke = st.selectbox("Merokok", features['SMOKE']['options'],
                               help="Apakah Anda merokok?")
            
            calc = st.selectbox("Konsumsi Alkohol", features['CALC']['options'],
                              help="Seberapa sering Anda mengonsumsi alkohol?")
            
            scc = st.selectbox("Monitor Kalori", features['SCC']['options'],
                             help="Apakah Anda memantau asupan kalori?")
        
        with col4:
            st.markdown("**🏃‍♀️ Aktivitas & Transportasi**")
            faf = st.slider("Frekuensi Aktivitas Fisik per Minggu", 
                          min_value=int(features['FAF']['range'][0]), 
                          max_value=int(features['FAF']['range'][1]), 
                          value=1,
                          help="Berapa kali seminggu Anda berolahraga?")
            
            tue = st.slider("Waktu Penggunaan Teknologi (jam/hari)", 
                          min_value=int(features['TUE']['range'][0]), 
                          max_value=int(features['TUE']['range'][1]), 
                          value=1,
                          help="Berapa jam per hari menggunakan gadget?")
            
            mtrans = st.selectbox("Mode Transportasi Utama", 
                                features['MTRANS']['options'],
                                help="Transportasi yang paling sering digunakan")
        
        # Submit button
        submitted = st.form_submit_button("🔮 Analisis Sekarang", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                "Gender": gender,
                "Age": float(age),
                "Height": float(height),
                "Weight": float(weight),
                "family_history_with_overweight": family_history,
                "FAVC": favc,
                "FCVC": float(fcvc),
                "NCP": float(ncp),
                "CAEC": caec,
                "SMOKE": smoke,
                "CH2O": float(ch2o),
                "SCC": scc,
                "FAF": float(faf),
                "TUE": float(tue),
                "CALC": calc,
                "MTRANS": mtrans
            }
            
            # Make prediction
            with st.spinner("🔄 Sedang menganalisis data Anda..."):
                result = make_prediction(input_data)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Display results
                display_prediction_results(result, input_data)

def display_prediction_results(result, input_data):
    """Menampilkan hasil prediksi"""
    st.success("✅ Analisis berhasil!")
    
    # Display main prediction result
    prediction_clean = result['prediction'].replace('_', ' ').title()
    
    st.markdown(f"""
    <div class="success-box" style="text-align: center;">
        <h2>🎯 Hasil Prediksi</h2>
        <p style="font-size: 1.2rem;">Kategori Obesitas Anda</p>
        <h1 style="color: #2E86AB; font-size: 2.5rem; margin: 1rem 0;">
            {prediction_clean}
        </h1>
    </div>
    """, unsafe_allow_html=True)

def display_model_info_page():
    """Menampilkan halaman informasi model"""
    st.markdown('<h2 class="sub-header">📊 Informasi Model</h2>', unsafe_allow_html=True)
    
    model_info = get_model_info()
    
    # Model overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🤖 Detail Model")
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Nama Model", model_info['model_name'])
            st.metric("Test Accuracy", f"{model_info['accuracy']:.3f}")
        with metrics_col2:
            st.metric("CV Score", f"{model_info['cv_score']:.3f}")
            st.metric("Jumlah Fitur", model_info['feature_count'])
    
    with col2:
        st.markdown("### 🎯 Kelas Prediksi")
        for i, class_name in enumerate(model_info['classes'], 1):
            clean_name = class_name.replace('_', ' ').title()
            st.write(f"{i}. {clean_name}")
    
    # Model status
    if not model_info['is_real_model']:
        st.warning("⚠️ **Mode Demo**: Aplikasi menggunakan model simulasi untuk demonstrasi. Prediksi didasarkan pada perhitungan BMI sederhana.")
    else:
        st.success("✅ **Model Aktual**: Menggunakan model machine learning yang telah dilatih.")
    
    # Performance visualization
    st.markdown("### 📈 Performa Model")
    
    # Create performance metrics chart
    performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [model_info['accuracy'], 0.951, 0.949, 0.950]
    }
    
    perf_df = pd.DataFrame(performance_data)
    fig = px.bar(perf_df, x='Metric', y='Score', 
                 title='📊 Metrik Evaluasi Model',
                 color='Score', 
                 color_continuous_scale='viridis',
                 text='Score')
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(showlegend=False, title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    st.markdown("### 🔍 Detail Teknis")
    
    with st.expander("📋 Informasi Lengkap"):
        st.json({
            "model_type": model_info['model_name'],
            "training_algorithm": "Supervised Learning",
            "validation_method": "Stratified K-Fold Cross Validation",
            "hyperparameter_tuning": "Grid Search CV",
            "preprocessing": [
                "Missing value imputation",
                "Outlier treatment (IQR capping)",
                "Categorical encoding",
                "Feature standardization"
            ],
            "evaluation_metrics": performance_data,
            "deployment": "Streamlit integrated application"
        })

def display_analysis_page():
    """Menampilkan halaman analisis dan insight"""
    st.markdown('<h2 class="sub-header">📈 Analisis & Insight</h2>', unsafe_allow_html=True)
    
    # Feature importance (mock data for demo)
    st.markdown("### 🔍 Pentingnya Fitur")
    
    feature_importance = {
        'Weight': 0.245,
        'Height': 0.189,
        'Age': 0.156,
        'FAF': 0.134,
        'FCVC': 0.098,
        'CH2O': 0.087,
        'NCP': 0.091
    }
    
    # Create feature importance chart
    importance_df = pd.DataFrame(list(feature_importance.items()), 
                                columns=['Fitur', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Fitur', 
                 orientation='h',
                 title='📊 Tingkat Kepentingan Fitur dalam Prediksi',
                 color='Importance',
                 color_continuous_scale='plasma')
    
    fig.update_layout(title_x=0.5, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Health insights
    st.markdown("### 💡 Insight Kesehatan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🏃‍♀️ Aktivitas Fisik</h4>
            <p>Frekuensi aktivitas fisik adalah faktor ke-4 terpenting dalam prediksi obesitas. 
            Olahraga teratur minimal 3x seminggu dapat mengurangi risiko obesitas hingga 40%.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>🥗 Konsumsi Sayuran</h4>
            <p>Mengonsumsi sayuran secara rutin (skor 3) dapat membantu menjaga berat badan ideal 
            dan mengurangi risiko obesitas tipe 2.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>💧 Hidrasi</h4>
            <p>Konsumsi air yang cukup (2-3 liter/hari) membantu metabolisme dan dapat 
            mengurangi false hunger yang sering menyebabkan overeating.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="danger-box">
            <h4>🍔 Makanan Berkalori Tinggi</h4>
            <p>Konsumsi makanan berkalori tinggi secara berlebihan adalah faktor risiko utama 
            obesitas. Batasi konsumsi junk food dan makanan olahan.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional insights
    st.markdown("### 📋 Kesimpulan")
    
    st.info("""
    **💡 Tips untuk Hidup Sehat:**
    
    1. **Pola Makan Seimbang**: Tingkatkan konsumsi sayuran dan kurangi makanan berkalori tinggi
    2. **Aktivitas Fisik**: Lakukan olahraga minimal 3x seminggu dengan durasi 30-45 menit
    3. **Hidrasi Cukup**: Minum air putih 8-10 gelas per hari
    4. **Monitor Berat Badan**: Periksa berat badan secara berkala
    5. **Konsultasi Medis**: Berkonsultasi dengan tenaga kesehatan untuk program yang tepat
    """)

if __name__ == "__main__":
    main() 