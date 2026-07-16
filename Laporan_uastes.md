
# LAPORAN UJIAN AKHIR SEMESTER (UAS) KECERDASAN BUATAN MENGGUNAKAN MACHINE LEARNING LIFE CYCLE

**Mata Kuliah:** Kecerdasan Buatan

**Topik Proyek:** Klasifikasi Penyakit Diabetes Menggunakan Pima Indians Diabetes Dataset dengan Pendekatan Machine Learning Life Cycle

**Anggota Kelompok:**

1. Julfa Nurhakiki 2406096
2. Agni Kirani 2406104

**LINK dataset** : Pima Indians Diabetes Dataset (`pima-indians-diabetes.csv`)

---

## 1. Judul Proyek

**"Analisis Perbandingan Performa Enam Algoritma Machine Learning (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, dan Gradient Boosting) untuk Klasifikasi Diabetes Berbasis Pima Indians Diabetes Dataset"**

---

## 2. Business Understanding

### Latar Belakang Masalah (Domain Proyek)

Diabetes melitus merupakan salah satu penyakit tidak menular dengan prevalensi yang terus meningkat di berbagai negara, termasuk pada populasi dengan riwayat genetik rentan seperti yang direkam pada dataset Pima Indians Diabetes. Diabetes yang tidak terdeteksi secara dini berisiko menimbulkan komplikasi serius, mulai dari gangguan kardiovaskular, kerusakan ginjal, hingga gangguan penglihatan. Proses diagnosis konvensional umumnya membutuhkan pemeriksaan laboratorium seperti tes glukosa darah, yang meskipun akurat, tetap membutuhkan waktu, biaya, dan akses fasilitas kesehatan yang memadai.

Dengan memanfaatkan data rekam medis dasar seperti kadar glukosa, tekanan darah, BMI, dan riwayat kehamilan, model *Machine Learning* dapat dibangun untuk membantu melakukan skrining awal risiko diabetes secara cepat sebagai alat bantu pendukung keputusan medis, bukan pengganti diagnosis dokter.

### Permasalahan Dunia Nyata

1. **Keterlambatan Deteksi Dini:** Banyak pasien tidak menyadari indikasi diabetes karena gejala awal yang tidak spesifik, sehingga pemeriksaan baru dilakukan setelah komplikasi muncul.
2. **Keterbatasan Skrining Massal:** Pemeriksaan laboratorium untuk seluruh populasi berisiko membutuhkan biaya dan sumber daya besar, sehingga skrining berbasis data sederhana (usia, BMI, tekanan darah, dan riwayat kehamilan) menjadi alternatif yang lebih efisien sebagai tahap penyaringan awal.

### Tujuan Proyek

1. **Membangun Model Klasifikasi Biner:** Mengimplementasikan dan membandingkan enam algoritma *Machine Learning* (Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, dan Gradient Boosting) untuk memprediksi status `Outcome` (diabetes/tidak diabetes).
2. **Menangani Kualitas Data Medis:** Mengidentifikasi dan menangani nilai 0 yang secara medis tidak wajar pada kolom seperti `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI`, karena nilai tersebut kemungkinan besar merupakan *missing value* tersembunyi.
3. **Mengatasi Ketidakseimbangan Kelas:** Menerapkan *stratified split* agar proporsi kelas `Outcome = 0` dan `Outcome = 1` tetap terjaga pada data latih maupun data uji, serta menggunakan metrik evaluasi yang sesuai untuk data tidak seimbang.
4. **Menentukan Model Terbaik:** Membandingkan performa keenam model menggunakan metrik Accuracy, Precision, Recall, F1-Score, dan Balanced Accuracy untuk menentukan algoritma paling andal dalam mendeteksi pasien diabetes.

### Target Pengguna (User)

1. **Tenaga Medis Puskesmas/Klinik:** Sebagai alat bantu skrining awal risiko diabetes sebelum dilakukan pemeriksaan laboratorium lanjutan.
2. **Peneliti Kesehatan Masyarakat:** Sebagai dasar analisis faktor risiko diabetes berbasis data.
3. **Pengembang Aplikasi Kesehatan Digital:** Sebagai fondasi model prediktif yang dapat diintegrasikan ke dalam aplikasi kesehatan.

### Solusi dan Manfaat

Proyek ini membangun model klasifikasi biner berbasis data tabular medis. Manfaat konkret yang dihadirkan meliputi:

- **Efisiensi Skrining:** Mempercepat proses identifikasi awal risiko diabetes tanpa menunggu hasil laboratorium lengkap.
- **Dukungan Keputusan Klinis:** Memberikan indikasi awal risiko diabetes berdasarkan data dasar pasien seperti glukosa, BMI, dan tekanan darah.
- **Efisiensi Sumber Daya:** Membantu memprioritaskan pasien yang perlu pemeriksaan lanjutan secara lebih terarah.

---

## 3. Data Understanding

### Sumber Dataset

Dataset yang digunakan dalam eksperimen ini adalah **Pima Indians Diabetes Dataset** (`pima-indians-diabetes.csv`). Dataset ini berisi data rekam medis dasar dari pasien perempuan keturunan Pima Indian, dengan target prediksi berupa kolom `Outcome`, yaitu apakah seseorang terindikasi diabetes (`1`) atau tidak (`0`).

### Karakteristik dan Spesifikasi Data

- **Ukuran Dataset:** 768 baris dan 9 kolom.
- **Tipe Data:** Seluruh fitur berbentuk numerik (7 kolom `int64`, 2 kolom `float64`).
- **Fitur Input:**

| Kolom | Keterangan |
| :--- | :--- |
| Pregnancies | Jumlah kehamilan |
| Glucose | Kadar glukosa |
| BloodPressure | Tekanan darah diastolik |
| SkinThickness | Ketebalan lipatan kulit |
| Insulin | Kadar insulin |
| BMI | Body Mass Index |
| DiabetespedigreeFunction | Riwayat/indikasi faktor keturunan diabetes |
| Age | Usia pasien |
| Outcome | Target klasifikasi diabetes (0/1) |

- **Missing Value:** Tidak ditemukan nilai `NaN` pada seluruh kolom.
- **Data Duplikat:** Tidak ditemukan data duplikat (0 baris).
- **Distribusi Target:** `Outcome = 0` (tidak diabetes) sebanyak 500 data (65,1%), `Outcome = 1` (diabetes) sebanyak 268 data (34,9%). Dataset ini tergolong **tidak seimbang (class imbalance)**.
- **Nilai 0 pada Fitur Medis:** Ditemukan nilai 0 yang kemungkinan besar merupakan *missing value* tersembunyi pada beberapa kolom:

| Kolom | Jumlah Nilai 0 | Persentase |
| :--- | :---: | :---: |
| Glucose | 5 | 0,65% |
| BloodPressure | 35 | 4,56% |
| SkinThickness | 227 | 29,56% |
| Insulin | 374 | 48,70% |
| BMI | 11 | 1,43% |

---

## 4. Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami pola, distribusi, dan hubungan antar fitur sebelum masuk ke tahap preprocessing dan pemodelan.

### 4.1 Distribusi Target

![alt text](image.png)

Grafik di atas menunjukkan bahwa kelas `Outcome = 0` (tidak diabetes) memiliki jumlah data jauh lebih banyak dibanding kelas `Outcome = 1` (diabetes), yaitu 500 berbanding 268 data. Kondisi *class imbalance* ini membuat evaluasi model tidak cukup hanya mengandalkan *accuracy*, melainkan juga *precision*, *recall*, *F1-score*, dan *confusion matrix*, terutama untuk memastikan model mampu mengenali pasien yang benar-benar diabetes.

### 4.2 Distribusi Fitur Numerik

[alt text](image-3.png)

Histogram menunjukkan bahwa sebagian besar fitur tidak berdistribusi normal. Fitur seperti `Pregnancies`, `Insulin`, `DiabetespedigreeFunction`, dan `Age` cenderung *skewed* ke kanan, sementara `SkinThickness` dan `Insulin` menunjukkan lonjakan nilai 0 yang mencurigakan sebagai *missing value* tersembunyi, bukan nilai medis yang valid.

### 4.3 Distribusi Fitur Berdasarkan Outcome

![alt text](image-4.png)

Visualisasi KDE per kelas menunjukkan bahwa fitur `Glucose` memiliki perbedaan distribusi paling jelas antara kelompok diabetes dan tidak diabetes, di mana kelompok diabetes cenderung memiliki kadar glukosa lebih tinggi. Fitur `BMI`, `Age`, dan `DiabetespedigreeFunction` juga menunjukkan kecenderungan nilai lebih tinggi pada kelompok diabetes meskipun distribusinya masih tumpang tindih.

### 4.4 Deteksi Outlier

![alt text](image-5.png)

Boxplot menunjukkan adanya kemungkinan outlier pada hampir seluruh fitur, terutama `Pregnancies`, `SkinThickness`, `Insulin`, dan `DiabetespedigreeFunction`. Karena dataset ini bersifat medis, outlier tidak dihapus secara langsung karena bisa saja merepresentasikan kondisi pasien yang valid.

### 4.5 Korelasi Antar Fitur

[alt text](image-6.png)

Fitur `Glucose` memiliki korelasi tertinggi terhadap `Outcome` (0,467), diikuti oleh `BMI` (0,293) dan `Age` (0,238). Korelasi antar fitur input tergolong lemah hingga sedang, misalnya `SkinThickness` dan `Insulin` (0,437), sehingga risiko multikolinearitas signifikan relatif rendah.

### 4.6 Scatter Plot Fitur Penting

[alt text](image-7.png)

Kombinasi `Glucose` dengan `BMI` dan `Age` menunjukkan pola pemisahan yang cukup terlihat antara kelas diabetes dan tidak diabetes, mengindikasikan bahwa kombinasi fitur ini berpotensi menjadi pembeda penting dalam proses klasifikasi.

### 4.7 Proporsi Diabetes Berdasarkan Kelompok Fitur (Binning Medis)

![alt text](image-8.png)

Beberapa fitur dikelompokkan berdasarkan ambang medis untuk melihat pola risiko diabetes:

- **Usia:** Kelompok Dewasa (18–59 tahun) berjumlah 736 data dengan tingkat diabetes 35,19%, sedangkan Lansia (≥60 tahun) berjumlah 32 data dengan tingkat diabetes 28,12%.
- **BMI:** Kelompok Obesitas (BMI ≥27) mendominasi dengan 585 data dan tingkat diabetes tertinggi sebesar 42,74%, jauh di atas kelompok Normal (6,86%).
- **Glucose:** Kelompok Prediabetes (140–199) menunjukkan tingkat diabetes sebesar 68,53%, jauh lebih tinggi dibanding kelompok Normal (<140) sebesar 23,14%.
- **Pregnancies:** Kelompok dengan ≥6 kali kehamilan menunjukkan tingkat diabetes tertinggi sebesar 50,68%, dibandingkan kelompok 1 kali kehamilan yang hanya 21,48%.

### 4.8 Analisis Nilai 0 pada Fitur Insulin

[alt text](image-9.png)

Nilai `Insulin = 0` ditemukan pada 236 pasien tidak diabetes (47,20%) dan 138 pasien diabetes (51,49%). Karena proporsi nilai 0 sangat besar dan tersebar di kedua kelas, nilai tersebut lebih mungkin merepresentasikan data yang tidak tercatat dibandingkan kondisi medis yang valid, sehingga diperlakukan sebagai kandidat *missing value* tersembunyi pada tahap preprocessing.

---

## 5. Data Preparation

### 5.1 Pemisahan Fitur dan Target

Kolom `Outcome` dipisahkan sebagai target, sedangkan 8 kolom lainnya digunakan sebagai fitur input. Ukuran data: `X` (768, 8) dan `y` (768,).

### 5.2 Train-Test Split

Dataset dibagi menggunakan *stratified split* dengan rasio **80% data latih dan 20% data uji** agar proporsi kelas tetap terjaga.

| Subset | Jumlah Data | Outcome 0 | Outcome 1 |
| :--- | :---: | :---: | :---: |
| Training | 614 | 400 (65,15%) | 214 (34,85%) |
| Testing | 154 | 100 (64,94%) | 54 (35,06%) |

### 5.3 Penanganan Nilai 0 sebagai Missing Value Tersembunyi

Kolom `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` diperlakukan sebagai kandidat *missing value* tersembunyi karena nilai 0 pada kolom-kolom tersebut tidak wajar secara medis. Nilai 0 pada `Pregnancies` dan `Outcome` tidak diubah karena masih valid secara konteks.

Sebelum dilakukan konversi ke NaN, dibuat fitur tambahan berupa **indikator biner** (`_was_zero`) untuk menyimpan informasi bahwa suatu nilai awalnya bernilai 0, agar informasi tersebut tidak hilang dan tetap dapat dimanfaatkan model.

| Kolom | Nilai 0 di X_train | Nilai 0 di X_test |
| :--- | :---: | :---: |
| Glucose | 4 | 1 |
| BloodPressure | 23 | 12 |
| SkinThickness | 175 | 52 |
| Insulin | 290 | 84 |
| BMI | 9 | 2 |

### 5.4 Imputasi Missing Value

Imputasi dilakukan menggunakan strategi **median**, karena median lebih *robust* terhadap outlier dan beberapa fitur memiliki distribusi *skewed*. Imputer hanya di-*fit* pada data training untuk mencegah *data leakage*.

| Fitur | Nilai Median (dari X_train) |
| :--- | :---: |
| Glucose | 117,0 |
| BloodPressure | 72,0 |
| SkinThickness | 29,0 |
| Insulin | 125,0 |
| BMI | 32,4 |

Setelah imputasi, seluruh *missing value* pada `X_train` dan `X_test` berhasil ditangani (total missing value = 0).

### 5.5 Feature Scaling

*Feature scaling* dilakukan menggunakan `StandardScaler` yang di-*fit* hanya pada data training, menghasilkan rata-rata mendekati 0 dan standar deviasi mendekati 1 pada seluruh fitur.

### 5.6 Menyiapkan Dua Versi Data untuk Modeling

Disiapkan dua versi data dengan total 13 fitur (8 fitur asli + 5 indikator `_was_zero`):

1. **Data Imputed tanpa Scaling** — untuk model berbasis tree (Decision Tree, Random Forest, Gradient Boosting). Ukuran: `X_train_tree` (614, 13), `X_test_tree` (154, 13).
2. **Data Imputed dengan Scaling** — untuk model yang sensitif terhadap skala (Logistic Regression, KNN, SVM). Ukuran: `X_train_scaled_model` (614, 13), `X_test_scaled_model` (154, 13).!!

---

## 6. Modeling

Proyek ini mengimplementasikan **enam algoritma** *Machine Learning* sebagai model baseline, dibagi menjadi dua kelompok berdasarkan kebutuhan *scaling*:

| Kelompok | Algoritma | Data yang Digunakan |
| :--- | :--- | :--- |
| Sensitif Skala | Logistic Regression, K-Nearest Neighbors, Support Vector Machine | Data Imputed + Scaling |
| Berbasis Tree | Decision Tree, Random Forest, Gradient Boosting | Data Imputed tanpa Scaling |

Seluruh model dilatih menggunakan parameter default (belum dilakukan *hyperparameter tuning*) dengan `random_state=42` untuk menjamin reprodusibilitas, karena tahap ini bertujuan membangun performa *baseline* terlebih dahulu.

### Sanity Check: Training Accuracy

| Model | Training Accuracy |
| :--- | :---: |
| Random Forest | 1,0000 |
| Decision Tree | 1,0000 |
| Gradient Boosting | 0,9235 |
| KNN | 0,8371 |
| SVM | 0,8306 |
| Logistic Regression | 0,7948 |

Training accuracy yang sangat tinggi pada Random Forest dan Decision Tree (mencapai 100%) mengindikasikan potensi *overfitting* yang baru dapat dipastikan setelah evaluasi pada data testing.

---

## 7. Evaluation & Perbandingan Model

Evaluasi dilakukan pada **Test Set** (154 data) yang belum pernah dilihat model selama pelatihan, menggunakan metrik Accuracy, Precision, Recall, F1-Score, dan Balanced Accuracy — mengingat dataset bersifat tidak seimbang.

### 7.1 Tabel Komparasi Metrik

| Model | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0,7532** | **0,6600** | **0,6111** | **0,6346** | **0,7206** |
| SVM | 0,7273 | 0,6250 | 0,5556 | 0,5882 | 0,6878 |
| Random Forest | 0,7338 | 0,6512 | 0,5185 | 0,5773 | 0,6843 |
| KNN | 0,7078 | 0,5818 | 0,5926 | 0,5872 | 0,6813 |
| Logistic Regression | 0,7143 | 0,6042 | 0,5370 | 0,5686 | 0,6735 |
| Decision Tree | 0,6753 | 0,5526 | 0,3889 | 0,4565 | 0,6094 |

Berdasarkan F1-Score, **Gradient Boosting** tampil sebagai model dengan performa terbaik dan paling seimbang antara *precision* dan *recall*, sekaligus meraih *balanced accuracy* tertinggi (0,7206). Sebaliknya, **Decision Tree** menunjukkan performa terlemah pada seluruh metrik, khususnya pada *recall* (0,3889) yang berarti model ini paling sering gagal mengenali pasien yang sebenarnya diabetes.

### 7.2 Visualisasi Perbandingan Metrik

[alt text](image-10.png)

Grafik batang di atas menegaskan bahwa Gradient Boosting secara konsisten unggul di hampir seluruh metrik evaluasi dibandingkan lima model lainnya.

### 7.3 Classification Report per Model

**Logistic Regression** — Accuracy 0,71; Recall kelas Diabetes 0,54; F1 kelas Diabetes 0,57.
**KNN** — Accuracy 0,71; Recall kelas Diabetes 0,59; F1 kelas Diabetes 0,59.
**SVM** — Accuracy 0,73; Recall kelas Diabetes 0,56; F1 kelas Diabetes 0,59.
**Decision Tree** — Accuracy 0,68; Recall kelas Diabetes 0,39; F1 kelas Diabetes 0,46.
**Random Forest** — Accuracy 0,73; Recall kelas Diabetes 0,52; F1 kelas Diabetes 0,58.
**Gradient Boosting** — Accuracy 0,75; Recall kelas Diabetes 0,61; F1 kelas Diabetes 0,63.

Recall kelas `Diabetes` menjadi perhatian khusus karena merepresentasikan kemampuan model mengenali pasien yang benar-benar terindikasi diabetes (*False Negative* rendah). Gradient Boosting unggul pada aspek ini dengan recall tertinggi (0,61) di antara seluruh model.

### 7.4 Confusion Matrix

| Logistic Regression | KNN | SVM |
| :---: | :---: | :---: |
| ![alt text](image-11.png) | ![KNN](image-12.png) | [SVM](image-13.png) |

| Decision Tree | Random Forest | Gradient Boosting |
| :---: | :---: | :---: |
|![alt text](image-14.png) |![alt text](image-15.png) | ![alt text](image-16.png) |

Confusion matrix menampilkan True Negative, False Positive, False Negative, dan True Positive dari 154 data uji (100 tidak diabetes, 54 diabetes). Decision Tree memiliki jumlah *False Negative* terbanyak, sejalan dengan recall-nya yang paling rendah, sedangkan Gradient Boosting mencatatkan keseimbangan terbaik antara *False Positive* dan *False Negative*.

### 7.5 Perbandingan Training Accuracy vs Testing Accuracy (Analisis Overfitting)

![alt text](image-17.png)

| Model | Training Accuracy | Testing Accuracy | Accuracy Gap |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0,7948 | 0,7143 | 0,0805 |
| KNN | 0,8371 | 0,7078 | 0,1293 |
| SVM | 0,8306 | 0,7273 | 0,1033 |
| Decision Tree | 1,0000 | 0,6753 | 0,3247 |
| Random Forest | 1,0000 | 0,7338 | 0,2662 |
| Gradient Boosting | 0,9235 | 0,7532 | 0,1702 |

Decision Tree dan Random Forest menunjukkan *gap* akurasi yang sangat besar (masing-masing 0,3247 dan 0,2662), mengindikasikan **overfitting** yang cukup parah terhadap data training. Logistic Regression memiliki *gap* paling kecil (0,0805), menandakan generalisasi yang relatif baik meskipun akurasinya bukan yang tertinggi. Gradient Boosting berada pada posisi yang cukup seimbang, di mana meskipun terdapat *gap* sebesar 0,1702, model ini tetap mencatatkan **testing accuracy tertinggi** di antara seluruh model.

---

## 8. Kesimpulan dan Rekomendasi

### Kesimpulan Hasil

1. **Kualitas Data:** Dataset Pima Indians Diabetes tidak memiliki *missing value* eksplisit maupun data duplikat, namun memiliki banyak nilai 0 yang secara medis tidak wajar pada kolom `Insulin` (48,70%), `SkinThickness` (29,56%), dan beberapa kolom lain, sehingga penanganan nilai tersembunyi menjadi krusial pada tahap preprocessing.
2. **Ketidakseimbangan Kelas:** Distribusi target 65,1% berbanding 34,9% menuntut evaluasi yang tidak hanya berfokus pada *accuracy*, melainkan juga *precision*, *recall*, *F1-score*, dan *balanced accuracy*.
3. **Model Terbaik:** **Gradient Boosting** menjadi model dengan performa terbaik secara keseluruhan pada data testing, dengan Accuracy 75,32%, F1-Score 0,6346, dan Balanced Accuracy 0,7206, sekaligus memiliki recall tertinggi dalam mendeteksi kelas diabetes.
4. **Overfitting pada Model Tree Sederhana:** Decision Tree dan Random Forest tanpa *tuning* mengalami overfitting signifikan (training accuracy 100%), sehingga performa keduanya pada data testing justru lebih rendah dibanding Gradient Boosting.
5. **Fitur Paling Berpengaruh:** `Glucose` merupakan fitur dengan korelasi tertinggi terhadap `Outcome`, sejalan dengan temuan EDA bahwa kelompok Prediabetes/Diabetes range memiliki tingkat diabetes jauh lebih tinggi dibanding kelompok Normal.

### Rekomendasi Pengembangan Masa Depan

- **Hyperparameter Tuning:** Melakukan pencarian parameter optimal (misalnya `GridSearchCV` atau `RandomizedSearchCV`) khususnya untuk Gradient Boosting dan Random Forest agar performa dan generalisasi model dapat lebih ditingkatkan.
- **Penanganan Class Imbalance Lanjutan:** Menerapkan teknik seperti `SMOTE` atau pembobotan kelas (`class_weight`) untuk meningkatkan recall pada kelas diabetes tanpa mengorbankan precision secara signifikan.
- **Eksplorasi Fitur Tambahan:** Menambahkan fitur turunan (*feature engineering*) berbasis interaksi antar fitur medis, misalnya rasio `Glucose` terhadap `BMI`, untuk menangkap pola non-linear yang lebih kaya.
- **Validasi Model:** Menggunakan `k-fold cross-validation` untuk memastikan hasil evaluasi lebih stabil dan tidak bergantung pada satu pembagian data saja.

---

## 9. Referensi (APA Style)

1. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.
2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
3. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189-1232.
4. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
5. American Diabetes Association. (2026). Diabetes Diagnosis. Diabetes.org.
6. Kementerian Kesehatan Republik Indonesia. (2026). Kategori Usia. Ayo Sehat Kemenkes.

---

## 10. Lampiran

Lampiran berupa dataset mentah, notebook eksperimen (`dataset_diabetes_MLLC.ipynb`), serta seluruh visualisasi EDA dan hasil evaluasi model dapat dilihat pada folder `images/` yang menyertai laporan ini.
