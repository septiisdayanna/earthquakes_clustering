# Global Earthquake Clustering & Classification

Machine learning project untuk menganalisis pola kejadian gempa bumi global tahun 2023 melalui dua tahap: **unsupervised learning menggunakan K-Means clustering**, kemudian **supervised learning untuk memprediksi label cluster** menggunakan beberapa algoritma klasifikasi.

## Overview

Project ini menggunakan dataset **Earthquakes 2023 Global** yang berisi data kejadian gempa bumi di seluruh dunia selama tahun 2023.

Analisis dilakukan dalam dua tahap utama:

1. **Clustering** — menemukan pola alami pada data gempa menggunakan K-Means dan mengevaluasi kualitas cluster dengan Silhouette Score.
2. **Classification** — menggunakan label cluster yang dihasilkan sebagai target untuk membandingkan beberapa algoritma supervised learning.

Pendekatan ini membentuk alur:

**Raw Earthquake Data → Preprocessing → K-Means Clustering → PCA → Cluster Labels → Classification → Model Evaluation**

---

## Objectives

* Mengeksplorasi karakteristik data gempa bumi global tahun 2023.
* Melakukan preprocessing terhadap data numerik dan kategorikal.
* Menentukan jumlah cluster menggunakan **Elbow Method**.
* Meningkatkan kualitas clustering melalui seleksi fitur dan reduksi dimensi menggunakan **PCA**.
* Mengevaluasi kualitas clustering menggunakan **Silhouette Score**.
* Menggunakan hasil clustering sebagai label untuk tahap klasifikasi.
* Membandingkan performa lima algoritma klasifikasi.
* Mengevaluasi model menggunakan accuracy, precision, recall, F1-score, confusion matrix, cross-validation, MSE, dan learning curve.

---

## Workflow

```text
Earthquake Dataset
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Preprocessing
        │
        ├── Missing Value Handling
        ├── Duplicate Removal
        ├── Feature Selection
        ├── Categorical Encoding
        └── Feature Scaling
        │
        ▼
K-Means Clustering
        │
        ├── Elbow Method
        └── Silhouette Score
        │
        ▼
PCA Dimensionality Reduction
        │
        ▼
Cluster Labels
        │
        ▼
Classification Dataset
        │
        ├── KNN
        ├── Decision Tree
        ├── Random Forest
        ├── SVM
        └── Naive Bayes
        │
        ▼
Model Evaluation
```

---

## Dataset

**Dataset:** Earthquakes 2023 Global

**Source:** Kaggle

**Original size:** 26,642 rows × 22 columns

**Data scope:** Kejadian gempa bumi global sepanjang tahun 2023.

Dataset awal memiliki fitur numerik dan kategorikal, termasuk informasi seperti:

* `latitude`
* `longitude`
* `depth`
* `mag`
* `magType`
* `nst`
* `gap`
* `dmin`
* `rms`
* `net`
* `horizontalError`
* `depthError`
* `magError`
* `magNst`
* `status`
* `locationSource`
* `magSource`
* serta atribut waktu dan lokasi.

Dalam preprocessing, beberapa kolom dengan informasi yang tidak digunakan untuk pemodelan dihapus. Missing values ditangani berdasarkan jumlah missing value, kemudian data diproses lebih lanjut untuk kebutuhan clustering. Setelah preprocessing, **24,682 baris** digunakan dalam tahap modeling.

---

## Methodology

### 1. Exploratory Data Analysis

Tahap awal dilakukan untuk memahami:

* struktur dataset;
* tipe data;
* distribusi fitur;
* missing values;
* korelasi antar fitur;
* distribusi kejadian gempa;
* dan duplikasi data.

Dataset awal terdiri dari **26,642 baris**. Pemeriksaan data menemukan **1,960 baris duplikat** pada tahap EDA.

### 2. Data Preprocessing

Preprocessing mencakup:

* menghapus kolom yang tidak digunakan;
* menangani missing values;
* menghapus data duplikat;
* melakukan encoding terhadap fitur kategorikal;
* melakukan normalisasi fitur;
* dan mempersiapkan data untuk clustering.

Beberapa atribut seperti `time`, `updated`, `place`, `id`, `type`, `status`, `locationSource`, dan `magSource` dihapus sebelum proses modeling.

### 3. K-Means Clustering

Clustering dilakukan menggunakan **K-Means**.

Jumlah cluster dievaluasi menggunakan **Elbow Method**. Berdasarkan hasil evaluasi pada notebook, model dengan:

```text
K = 3
```

digunakan sebagai salah satu konfigurasi clustering.

Evaluasi awal menghasilkan:

| Configuration | Silhouette Score |
| ------------- | ---------------: |
| K-Means, K=3  |           0.5285 |
| K-Means, K=5  |           0.5608 |

Hasil tersebut kemudian dioptimalkan melalui proses seleksi fitur dan reduksi dimensi.

### 4. Feature Selection & PCA

Feature selection dilakukan dengan menghapus:

* `nst`
* `gap`
* `rms`
* `depthError`
* seluruh fitur yang memiliki prefix `net_`

Setelah itu, PCA digunakan dengan:

```python
PCA(n_components=2)
```

untuk mereduksi data menjadi dua komponen.

Hasil clustering setelah reduksi dimensi menghasilkan **Silhouette Score sebesar 0.8613**, meningkat dibandingkan skor sebelum optimasi.

### 5. Classification

Hasil clustering disimpan dalam:

```text
hasil_clustering.csv
```

Dataset ini memiliki **24,682 baris** dan label `cluster` yang digunakan sebagai target klasifikasi.

Data kemudian dibagi menggunakan:

```text
Training: 19,745 samples
Testing : 4,937 samples
Test size: 20%
Random state: 42
```

Fitur kategorikal diproses menggunakan **One-Hot Encoding**, sedangkan fitur numerik diproses menggunakan **StandardScaler**.

Model yang dibandingkan:

1. K-Nearest Neighbors (KNN)
2. Decision Tree (DT)
3. Random Forest (RF)
4. Support Vector Machine (SVM)
5. Gaussian Naive Bayes (NB)

---

## Results

### Clustering

| Stage        | Silhouette Score |
| ------------ | ---------------: |
| K-Means, K=3 |           0.5285 |
| K-Means, K=5 |           0.5608 |
| After PCA    |       **0.8613** |

Hasil menunjukkan peningkatan Silhouette Score setelah proses seleksi fitur dan reduksi dimensi menggunakan PCA.

### Classification

Hasil evaluasi pada testing set:

| Model         |  Accuracy |   F1-Score |
| ------------- | --------: | ---------: |
| KNN           |  99.9797% | ≈99.9653%* |
| Decision Tree | 100.0000% |  100.0000% |
| Random Forest | 100.0000% |  100.0000% |
| SVM           | 100.0000% |  100.0000% |
| Naive Bayes   | 100.0000% |  100.0000% |

*F1-score KNN merupakan macro average yang dihitung dari tiga nilai F1 per kelas yang tercatat pada notebook. Nilai accuracy dan F1 per kelas berasal langsung dari hasil evaluasi notebook.

Selain evaluasi testing, notebook juga melakukan:

* confusion matrix;
* 5-fold cross-validation;
* Mean Squared Error;
* learning curve.

Cross-validation menghasilkan mean score **0.99954 untuk KNN**, **1.00000 untuk Decision Tree**, **1.00000 untuk Random Forest**, **0.99995 untuk SVM**, dan **1.00000 untuk Naive Bayes**.

---

## Key Findings

* K-Means dengan `K=3` menghasilkan Silhouette Score **0.5285** pada evaluasi awal.
* Seleksi fitur dan reduksi dimensi menggunakan PCA meningkatkan Silhouette Score menjadi **0.8613**.
* Label hasil clustering dapat digunakan sebagai target untuk tahap supervised classification.
* Lima algoritma klasifikasi menunjukkan performa sangat tinggi pada testing set.
* Decision Tree, Random Forest, SVM, dan Naive Bayes menghasilkan accuracy **100%** pada testing set, sedangkan KNN menghasilkan **99.9797%**.
* Hasil yang sangat tinggi perlu diinterpretasikan dalam konteks bahwa target klasifikasi merupakan **label yang dihasilkan oleh proses clustering**, bukan ground-truth label independen.

---

## Limitations

* Label pada tahap klasifikasi berasal dari hasil K-Means clustering sehingga performa klasifikasi tidak dapat diartikan sebagai akurasi terhadap kategori gempa yang telah diverifikasi secara eksternal.
* Silhouette Score mengukur kualitas pemisahan cluster berdasarkan representasi data yang digunakan, bukan validitas ilmiah dari kategori gempa.
* Hasil klasifikasi yang sangat tinggi perlu dipahami sebagai kemampuan model dalam mempelajari pola label cluster yang telah dibuat pada tahap sebelumnya.
* PCA mereduksi data menjadi dua komponen sehingga interpretasi hasil clustering perlu mempertimbangkan informasi yang dipertahankan oleh kedua komponen tersebut.
* Project ini berfokus pada analisis dan pemodelan data tahun 2023 dan belum melakukan validasi terhadap periode waktu atau dataset eksternal.

---

## Future Improvements

Beberapa pengembangan yang dapat dilakukan:

* Membandingkan K-Means dengan algoritma clustering lain seperti DBSCAN atau hierarchical clustering.
* Melakukan evaluasi cluster menggunakan lebih dari satu internal clustering metric.
* Menambahkan visualisasi geospasial untuk melihat persebaran cluster berdasarkan latitude dan longitude.
* Melakukan analisis karakteristik masing-masing cluster agar label cluster dapat diinterpretasikan secara lebih bermakna.
* Menguji hasil model pada data gempa dari periode atau sumber lain.
* Mengevaluasi apakah performa klasifikasi tetap tinggi ketika menggunakan data baru yang tidak berasal dari dataset hasil clustering yang sama.

---

## Technologies

* **Python**
* **pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **scikit-learn**
* **Yellowbrick**

### Machine Learning

* K-Means Clustering
* Principal Component Analysis (PCA)
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Support Vector Machine
* Gaussian Naive Bayes

### Evaluation

* Elbow Method
* Silhouette Score
* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* 5-Fold Cross-Validation
* Mean Squared Error
* Learning Curve

---

## Repository Structure

```text
earthquakes_clustering/
│
├── earthquakes_2023_global.csv
│   └── Dataset gempa bumi global tahun 2023
│
├── [Clustering]_Submission_Akhir_BMLP_Septi_Isdayanna.ipynb
│   └── EDA, preprocessing, K-Means clustering,
│       feature selection, PCA, dan evaluasi clustering
│
├── hasil_clustering.csv
│   └── Dataset hasil clustering dengan label cluster
│
└── [Klasifikasi]_Submission_Akhir_BMLP_Septi_Isdayanna.ipynb
    └── Preprocessing, training, perbandingan model,
        dan evaluasi klasifikasi
```

Repository saat ini memang berisi dua notebook, dataset mentah, dan dataset hasil clustering tersebut.

---

## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/septiisdayanna/earthquakes_clustering.git
cd earthquakes_clustering
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick jupyter
```

### 3. Run the Clustering Notebook

Jalankan:

```text
[Clustering]_Submission_Akhir_BMLP_Septi_Isdayanna.ipynb
```

Notebook ini melakukan preprocessing, clustering, feature selection, PCA, dan menghasilkan:

```text
hasil_clustering.csv
```

### 4. Run the Classification Notebook

Setelah `hasil_clustering.csv` tersedia, jalankan:

```text
[Klasifikasi]_Submission_Akhir_BMLP_Septi_Isdayanna.ipynb
```

Notebook tersebut menggunakan label `cluster` sebagai target klasifikasi dan membandingkan lima algoritma machine learning.

---

## Author
**Septi Isdayanna**
