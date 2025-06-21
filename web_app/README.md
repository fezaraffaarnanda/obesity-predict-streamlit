# 🏥 Sistem Prediksi Tingkat Obesitas

Aplikasi web untuk klasifikasi tingkat obesitas berdasarkan gaya hidup dan kebiasaan sehari-hari menggunakan machine learning.

## 📋 Deskripsi

Sistem ini menggunakan machine learning untuk memprediksi tingkat obesitas seseorang berdasarkan 16 parameter gaya hidup, termasuk kebiasaan makan, aktivitas fisik, dan faktor demografis. Aplikasi terdiri dari:

- **Backend API** (FastAPI) - Endpoint untuk prediksi machine learning
- **Frontend Web** (Streamlit) - Interface pengguna yang interaktif
- **ML Pipeline** - Model yang telah dilatih untuk klasifikasi obesitas

## 🏗️ Arsitektur

```
📁 Project Structure
├── app.py              # FastAPI Backend
├── streamlit.py        # Streamlit Frontend
├── run_app.py          # Script menjalankan kedua aplikasi
├── requirements.txt    # Dependencies
└── README.md          # Dokumentasi
```

## 🚀 Quick Start

### 1. Instalasi Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
python run_app.py
```

Aplikasi akan otomatis:

- ✅ Menjalankan backend di port 8000
- ✅ Menjalankan frontend di port 8501
- ✅ Membuka browser ke halaman aplikasi

### 3. Akses Aplikasi

- **Web App**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API Interactive**: http://localhost:8000/redoc

## 📊 Fitur Utama

### 🤖 Backend API (FastAPI)

- ✅ RESTful API dengan dokumentasi otomatis
- ✅ Endpoint prediksi machine learning
- ✅ Validasi input dengan Pydantic
- ✅ Error handling yang robust
- ✅ CORS support
- ✅ Health check endpoints

### 🖥️ Frontend Web (Streamlit)

- ✅ Interface yang user-friendly
- ✅ Multi-page navigation
- ✅ Form input interaktif
- ✅ Visualisasi hasil dengan charts
- ✅ Dashboard informasi model
- ✅ Export hasil analisis

## 🎯 Input Features

Aplikasi meminta 16 parameter input:

### 👤 Data Personal

- **Gender**: Jenis kelamin (Male/Female)
- **Age**: Usia (14-61 tahun)
- **Height**: Tinggi badan (1.45-1.98 meter)
- **Weight**: Berat badan (39-173 kg)

### 🧬 Riwayat Keluarga

- **Family History**: Riwayat keluarga overweight (yes/no)

### 🍽️ Kebiasaan Makan

- **FAVC**: Konsumsi makanan berkalori tinggi (yes/no)
- **FCVC**: Frekuensi konsumsi sayuran (1-3)
- **NCP**: Jumlah makan utama (1-4)
- **CAEC**: Makan di antara waktu makan (no/Sometimes/Frequently/Always)

### 🚭 Gaya Hidup

- **SMOKE**: Kebiasaan merokok (yes/no)
- **CH2O**: Konsumsi air harian (1-3 liter)
- **SCC**: Monitor kalori (yes/no)
- **CALC**: Konsumsi alkohol (no/Sometimes/Frequently/Always)

### 🏃‍♀️ Aktivitas

- **FAF**: Frekuensi aktivitas fisik (0-3 per minggu)
- **TUE**: Waktu penggunaan teknologi (0-2 jam/hari)
- **MTRANS**: Mode transportasi utama

## 📈 Output Prediksi

### 🎯 Kategori Obesitas

1. **Insufficient Weight** - Berat badan kurang
2. **Normal Weight** - Berat badan normal
3. **Overweight Level I** - Kelebihan berat badan tingkat 1
4. **Overweight Level II** - Kelebihan berat badan tingkat 2
5. **Obesity Type I** - Obesitas tipe 1
6. **Obesity Type II** - Obesitas tipe 2
7. **Obesity Type III** - Obesitas tipe 3

## 🔗 API Endpoints

| Method | Endpoint      | Deskripsi         |
| ------ | ------------- | ----------------- |
| GET    | `/`           | Info API          |
| GET    | `/health`     | Health check      |
| GET    | `/model-info` | Informasi model   |
| GET    | `/features`   | Info fitur input  |
| POST   | `/predict`    | Prediksi obesitas |

### Contoh Request

```json
POST /predict
{
  "Gender": "Male",
  "Age": 25,
  "Height": 1.75,
  "Weight": 70,
  "family_history_with_overweight": "yes",
  "FAVC": "no",
  "FCVC": 2.0,
  "NCP": 3.0,
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "CH2O": 2.0,
  "SCC": "yes",
  "FAF": 1.0,
  "TUE": 1.0,
  "CALC": "no",
  "MTRANS": "Public_Transportation"
}
```

### Contoh Response

```json
{
  "prediction": "Normal_Weight",
  "probability": {
    "Normal_Weight": 0.85,
    "Overweight_Level_I": 0.1,
    "Insufficient_Weight": 0.05,
    "Obesity_Type_I": 0.0,
    "Overweight_Level_II": 0.0,
    "Obesity_Type_II": 0.0,
    "Obesity_Type_III": 0.0
  }
}
```

## 🛠️ Teknologi

- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit, Plotly
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express & Graph Objects

## 📞 Help & Support

### Commands

```bash
python run_app.py --help     # Bantuan
python run_app.py --check    # Cek dependencies
```
