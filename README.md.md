<div align="center">

# 🩺 Prediksi Risiko Diabetes Menggunakan Algoritma Decision Tree dan K-Nearest Neighbor (KNN)

### Ujian Akhir Semester (UAS) Mata Kuliah Kecerdasan Buatan

Machine Learning Classification Project menggunakan **Decision Tree** dan **K-Nearest Neighbor (KNN)** untuk memprediksi risiko diabetes berdasarkan data kesehatan pasien.

---

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)]()
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow?logo=scikitlearn)]()
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?logo=pandas)]()
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)]()
[![License](https://img.shields.io/badge/License-Education-green)]()

</div>

---

# 📖 Deskripsi Proyek

Proyek ini merupakan tugas **Ujian Akhir Semester (UAS) Kecerdasan Buatan** yang bertujuan membangun model Machine Learning untuk memprediksi apakah seseorang memiliki risiko terkena diabetes berdasarkan beberapa indikator kesehatan.

Model klasifikasi dibangun menggunakan dua algoritma, yaitu:

- 🌳 Decision Tree
- 👥 K-Nearest Neighbor (KNN)

Selanjutnya kedua algoritma dibandingkan berdasarkan performanya menggunakan metrik evaluasi seperti Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.

---

# 🎯 Tujuan Proyek

- Memprediksi risiko diabetes berdasarkan data pasien.
- Membandingkan performa algoritma Decision Tree dan KNN.
- Menentukan algoritma terbaik berdasarkan hasil evaluasi.
- Memberikan contoh penerapan Artificial Intelligence di bidang kesehatan.

---

# 📊 Dataset

Dataset yang digunakan adalah:

**Pima Indians Diabetes Database**

Dataset berisi data kesehatan pasien wanita berusia minimal 21 tahun dengan keturunan Pima Indian.

Jumlah Data

- 768 Data
- 8 Fitur
- 1 Target Klasifikasi

Target

- 0 = Tidak Diabetes
- 1 = Diabetes

---

# 📑 Deskripsi Atribut

| No | Atribut | Keterangan |
|----|----------|------------|
|1|Pregnancies|Jumlah kehamilan|
|2|Glucose|Kadar glukosa darah|
|3|BloodPressure|Tekanan darah|
|4|SkinThickness|Ketebalan lipatan kulit|
|5|Insulin|Kadar insulin|
|6|BMI|Body Mass Index|
|7|DiabetesPedigreeFunction|Riwayat diabetes keluarga|
|8|Age|Usia|
|9|Outcome|Target klasifikasi|

---

# ⚙️ Algoritma yang Digunakan

## 🌳 Decision Tree

Decision Tree merupakan algoritma klasifikasi yang bekerja dengan membentuk struktur pohon keputusan berdasarkan atribut yang paling informatif.

### Kelebihan

- Mudah dipahami
- Mudah divisualisasikan
- Cepat diproses

---

## 👥 K-Nearest Neighbor (KNN)

KNN mengklasifikasikan data baru berdasarkan sejumlah tetangga terdekat (Nearest Neighbor).

### Kelebihan

- Sederhana
- Akurasi cukup tinggi
- Cocok untuk data klasifikasi

---

# 🔬 Tahapan Penelitian

```
Business Understanding
        │
        ▼
Data Understanding
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Preparation
        │
        ▼
Modeling
        │
        ▼
Evaluation
        │
        ▼
Kesimpulan
```

---

# 📈 Exploratory Data Analysis

EDA yang dilakukan meliputi:

- Histogram
- Boxplot
- Heatmap Korelasi
- Distribusi Target
- Pairplot
- Analisis Missing Value
- Analisis Outlier

---

# 🛠 Data Preparation

Tahapan preprocessing meliputi:

- Data Cleaning
- Penanganan Missing Value
- Normalisasi Data
- Train-Test Split
- Feature Selection

---

# 🤖 Modeling

Model yang dibandingkan

|Model|
|------|
|Decision Tree|
|K-Nearest Neighbor|

---

# 📊 Evaluasi

Evaluasi dilakukan menggunakan

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Model terbaik dipilih berdasarkan nilai evaluasi tertinggi.

---

# 💻 Library

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📁 Struktur Repository

```
UAS-KecerdasanBuatan/
│
├── README.md
├── Laporan_uas.md
├── uas_model.ipynb
│
├── data/
│   ├── diabetes.csv
│   └── jurnal/
│       ├── jurnal1.pdf
│       ├── jurnal2.pdf
│       ├── jurnal3.pdf
│       ├── jurnal4.pdf
│       └── jurnal5.pdf
│
└── images/
    ├── heatmap.png
    ├── histogram.png
    ├── confusion_matrix_dt.png
    └── confusion_matrix_knn.png
```

---

# ▶️ Cara Menjalankan

1. Clone repository

```bash
git clone https://github.com/username/UAS-KecerdasanBuatan.git
```

2. Masuk ke folder

```bash
cd UAS-KecerdasanBuatan
```

3. Install library

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

4. Jalankan Notebook

```bash
jupyter notebook
```

5. Buka file

```
uas_model.ipynb
```

---

# 📌 Hasil yang Diharapkan

- Model mampu mengklasifikasikan risiko diabetes.
- Membandingkan performa Decision Tree dan KNN.
- Menentukan model dengan akurasi terbaik.

---

# 📚 Referensi

- International Diabetes Federation (IDF)
- UCI Machine Learning Repository
- Kaggle
- Scikit-Learn Documentation
- Minimal 5 jurnal ilmiah

---

# 👨‍💻 Penyusun

**Agni Kirani**

Ujian Akhir Semester

Mata Kuliah Kecerdasan Buatan

Program Studi Teknik Informatika

Universitas __________

2026

---

<div align="center">

⭐ Terima kasih telah mengunjungi repository ini.

</div>