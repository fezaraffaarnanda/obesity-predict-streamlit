# 🏥 Sistem Prediksi Obesitas - Streamlit App

Aplikasi web terpadu untuk klasifikasi tingkat obesitas menggunakan machine learning, dikemas dalam satu file Streamlit untuk kemudahan deployment.

## 📋 Deskripsi

Ini adalah versi **all-in-one** dari Sistem Prediksi Obesitas yang menggabungkan backend dan frontend dalam satu aplikasi Streamlit. Aplikasi ini tidak memerlukan setup server terpisah dan dapat langsung di-deploy ke Streamlit Cloud.

## 🚀 Quick Start

### 1. Instalasi Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run streamlit_app.py
```

### 3. Akses Aplikasi

Aplikasi akan otomatis terbuka di browser pada:

- **Local**: http://localhost:8501

## 🏗️ Arsitektur Aplikasi

```
📁 Streamlit App (Single File)
├── 🧠 Backend Logic (Python Functions)
│   ├── Model Pipeline Loading
│   ├── Prediction Functions
│   └── Data Preprocessing
└── 🎨 Frontend UI (Streamlit Components)
    ├── Multi-page Navigation
    ├── Interactive Forms
    └── Data Visualization
```

## ✨ Keunggulan Versi Streamlit

### 🎯 **Kemudahan Deployment**

- ✅ Single file application
- ✅ No server setup required
- ✅ Direct deployment to Streamlit Cloud
- ✅ Built-in caching for performance

### 🔧 **Arsitektur Terpadu**

- ✅ Backend logic integrated as Python functions
- ✅ No API calls - direct function calls
- ✅ Simplified error handling
- ✅ Faster response time

### 📱 **User Experience**

- ✅ Responsive design
- ✅ Real-time predictions
- ✅ Interactive visualizations
- ✅ Export functionality

## 📊 Fitur Utama

### 🏠 **Halaman Beranda**

- Overview sistem dan fitur
- Panduan penggunaan
- Status aplikasi

### 🔮 **Halaman Prediksi**

- Form input 16 parameter gaya hidup
- Real-time prediction
- Distribusi probabilitas dengan chart
- Export hasil dalam format JSON

### 📊 **Info Model**

- Detail performa model
- Metrik evaluasi dengan visualisasi
- Informasi teknis lengkap

### 📈 **Analisis & Insight**

- Feature importance chart
- Health insights dan rekomendasi
- Tips hidup sehat

## 🎯 Input Parameters

| Kategori       | Parameter      | Tipe        | Range/Options                      |
| -------------- | -------------- | ----------- | ---------------------------------- |
| **Personal**   | Gender         | Categorical | Male/Female                        |
|                | Age            | Numeric     | 14-61 tahun                        |
|                | Height         | Numeric     | 1.45-1.98 meter                    |
|                | Weight         | Numeric     | 39-173 kg                          |
| **Keluarga**   | Family History | Categorical | yes/no                             |
| **Makan**      | FAVC           | Categorical | yes/no                             |
|                | FCVC           | Numeric     | 1-3                                |
|                | NCP            | Numeric     | 1-5                                |
|                | CAEC           | Categorical | no/Sometimes/Frequently/Always     |
| **Gaya Hidup** | SMOKE          | Categorical | yes/no                             |
|                | CH2O           | Numeric     | 1-5 liter                          |
|                | SCC            | Categorical | yes/no                             |
|                | CALC           | Categorical | no/Sometimes/Frequently/Always     |
| **Aktivitas**  | FAF            | Numeric     | 0-7 per minggu                     |
|                | TUE            | Numeric     | 0-5 jam/hari                       |
|                | MTRANS         | Categorical | Walking/Bike/Motorbike/Public/Auto |

## 🎭 Output Classifications

1. **Insufficient Weight** - Berat badan kurang
2. **Normal Weight** - Berat badan normal
3. **Overweight Level I** - Kelebihan berat badan tingkat 1
4. **Overweight Level II** - Kelebihan berat badan tingkat 2
5. **Obesity Type I** - Obesitas tipe 1
6. **Obesity Type II** - Obesitas tipe 2
7. **Obesity Type III** - Obesitas tipe 3

## 🚀 Deployment ke Streamlit Cloud

### 1. Upload ke GitHub

```bash
git add streamlit_app.py requirements_streamlit.txt
git commit -m "Add Streamlit integrated app"
git push origin main
```

### 2. Deploy ke Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Connect dengan GitHub repository
3. Pilih file: `streamlit_app.py`
4. Pilih requirements: `requirements_streamlit.txt`
5. Click "Deploy"

### 3. URL Deployment

Aplikasi akan tersedia di:

```
https://[username]-[repo-name]-[branch]-streamlitapp-[hash].streamlit.app
```

## 🔧 Konfigurasi

### Environment Variables (Optional)

```bash
# Untuk production deployment
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### .streamlit/config.toml (Optional)

```toml
[theme]
primaryColor = "#2E86AB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
enableCORS = false
```

## 🧪 Test Cases

### Test Case 1: Normal Weight

```python
input_data = {
    "Gender": "Male",
    "Age": 25,
    "Height": 1.75,
    "Weight": 70,
    "family_history_with_overweight": "no",
    # ... other parameters
}
```

**Expected**: Prediction "Normal Weight" dengan probabilitas tinggi

### Test Case 2: Obesity

```python
input_data = {
    "Gender": "Male",
    "Age": 35,
    "Height": 1.70,
    "Weight": 120,
    "family_history_with_overweight": "yes",
    # ... other parameters
}
```

**Expected**: Prediction "Obesity Type III" dengan probabilitas tinggi

## 🛠️ Teknologi

- **Frontend & Backend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express & Graph Objects
- **Machine Learning**: Scikit-learn (for future real model)
- **Caching**: Streamlit's built-in caching

## 🔄 Perbedaan dengan Versi Terpisah

| Aspek             | Versi Terpisah       | Versi Streamlit       |
| ----------------- | -------------------- | --------------------- |
| **Files**         | app.py + frontend.py | streamlit_app.py      |
| **Communication** | HTTP API calls       | Direct function calls |
| **Deployment**    | 2 services           | 1 service             |
| **Setup**         | Backend + Frontend   | Streamlit only        |
| **Performance**   | Network latency      | In-memory             |
| **Complexity**    | Medium               | Simple                |

## 💡 Best Practices

### Development

```bash
# Install in virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

### Production

```bash
# Use specific versions in requirements
streamlit==1.28.0
pandas==1.5.3
numpy==1.24.3
plotly==5.15.0
```

## 📞 Support

### Local Development

```bash
streamlit run streamlit_app.py --server.port 8501
```

### Debug Mode

```bash
streamlit run streamlit_app.py --logger.level debug
```

### Clear Cache

- Tekan **"R"** di browser untuk reload
- Atau gunakan tombol "Clear Cache" di menu Streamlit

## 🎉 Hasil

Aplikasi Streamlit ini memberikan pengalaman yang sama dengan versi terpisah namun dengan:

- ✅ Setup yang lebih mudah
- ✅ Deployment yang lebih cepat
- ✅ Performance yang lebih baik
- ✅ Maintenance yang lebih simple

Perfect untuk demo, prototype, dan production deployment! 🚀
