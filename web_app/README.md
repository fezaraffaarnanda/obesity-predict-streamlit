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

## 🔧 Manual Setup

### Jalankan Backend Saja

```bash
python app.py
# atau
uvicorn app:app --reload --port 8000
```

### Jalankan Frontend Saja

```bash
streamlit run streamlit.py
```

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

### ⚠️ Tingkat Risiko

- **Normal** - Kondisi sehat
- **Rendah** - Perlu perhatian minimal
- **Sedang** - Perlu modifikasi gaya hidup
- **Tinggi** - Perlu intervensi medis
- **Sangat Tinggi** - Perlu penanganan intensif
- **Ekstrem** - Perlu intervensi darurat

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
    "Insufficient_Weight": 0.05
  },
  "risk_level": "Normal",
  "recommendations": [
    "Pertahankan pola makan sehat",
    "Rutin berolahraga minimal 150 menit per minggu"
  ],
  "bmi": 22.86,
  "bmi_category": "Normal"
}
```

## 🛠️ Teknologi

- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit, Plotly
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express & Graph Objects

## 🐛 Troubleshooting

### Port Conflicts

```bash
# Jika port 8000 atau 8501 sudah digunakan
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac
```

### Missing Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Backend Connection Issues

1. Pastikan backend berjalan di port 8000
2. Cek firewall settings
3. Verifikasi tidak ada aplikasi lain menggunakan port tersebut

## ⚠️ Disclaimer

**PENTING**: Aplikasi ini dirancang sebagai alat bantu screening awal dan **TIDAK DAPAT MENGGANTIKAN** diagnosis medis profesional. Selalu konsultasikan hasil dengan tenaga kesehatan yang kompeten.

## 📞 Help & Support

### Commands

```bash
python run_app.py --help     # Bantuan
python run_app.py --check    # Cek dependencies
```

### Common Issues

- **"Model not loaded"**: Normal untuk demo (menggunakan mock prediction)
- **"Connection refused"**: Backend belum siap, tunggu beberapa detik
- **"Import errors"**: Install ulang dependencies

## 📄 License

MIT License - Free for educational and research purposes.

---

**Developed with ❤️ for Healthcare Innovation**

🏥 Sistem Prediksi Obesitas v1.0.0
