# [ai] #01 - ML Introduction to Machine Learning

![ml](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ml.png)

## What is Machine Learning?

---

Machine Learning adalah cabang dari Artificial Intelligence yang memungkinkan komputer untuk belajar dan membuat keputusan tanpa harus diprogram secara eksplisit untuk setiap situasi. Bayangkan seperti mengajari anak kecil mengenali kucing - daripada memberikan daftar ciri-ciri kucing yang panjang, kita tunjukkan ribuan foto kucing sampai dia bisa mengenali sendiri.

Dalam konteks teknis, ML adalah algoritma yang dapat menemukan pola dalam data dan menggunakan pola tersebut untuk membuat prediksi atau keputusan pada data yang belum pernah dilihat sebelumnya.

## Types of Machine Learning

---

### Supervised Learning

Seperti belajar dengan guru yang selalu memberikan jawaban yang benar. Kita punya data input (X) dan target output (y) yang sudah diketahui.

**Contoh:**

- Prediksi harga rumah berdasarkan luas, lokasi, dll
- Klasifikasi email spam atau bukan
- Diagnosa penyakit berdasarkan gejala

**Sub-kategori:**

- **Classification**: Output berupa kategori (spam/not spam, kucing/anjing)
- **Regression**: Output berupa nilai kontinyu (harga, suhu, rating)

### Unsupervised Learning

Seperti menjelajahi perpustakaan tanpa panduan - kita cari pola tersembunyi dalam data tanpa tahu jawaban yang "benar".

**Contoh:**

- Clustering customer berdasarkan behavior
- Anomaly detection untuk fraud
- Dimensionality reduction untuk visualisasi

### Reinforcement Learning

Seperti bermain game - kita belajar melalui trial and error dengan reward dan punishment.

**Contoh:**

- Game AI (AlphaGo, game strategi)
- Robot navigation
- Recommendation systems

## Key Concepts and Terminology

---

### Dataset

Kumpulan data yang kita gunakan untuk training. Seperti buku pelajaran yang berisi contoh-contoh soal.

### Features (X)

Variabel input atau karakteristik yang kita gunakan untuk membuat prediksi. Seperti tinggi, berat, umur untuk prediksi kesehatan.

### Target/Label (y)

Output yang ingin kita prediksi. Dalam supervised learning, ini adalah "jawaban yang benar".

### Training vs Testing

- **Training**: Fase pembelajaran dimana model belajar dari data
- **Testing**: Fase evaluasi dimana kita test performa model pada data yang belum pernah dilihat

### Overfitting vs Underfitting

- **Overfitting**: Model terlalu "hafal" training data, seperti murid yang cuma bisa jawab soal yang persis sama dengan contoh
- **Underfitting**: Model terlalu sederhana, seperti murid yang belum paham konsep dasarnya

## The Machine Learning Pipeline

---

### 1. Problem Definition

Tentukan apa yang ingin kita solve dan jenis ML apa yang cocok.

### 2. Data Collection

Kumpulkan data yang relevan dan berkualitas - "garbage in, garbage out".

### 3. Data Preprocessing

Bersihkan dan siapkan data:

- Handle missing values
- Remove outliers
- Feature scaling
- Encoding categorical variables

### 4. Model Selection

Pilih algoritma yang sesuai dengan problem dan data kita.

### 5. Training

Latih model dengan training data.

### 6. Evaluation

Evaluasi performa dengan metrics yang sesuai.

### 7. Deployment

Deploy model ke production untuk digunakan real-world.

## Popular Machine Learning Libraries

---

```python
# Core libraries untuk data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Example basic workflow
# Load data
data = pd.read_csv('dataset.csv')

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
```

## Why Learn Machine Learning?

---

Machine Learning adalah fondasi untuk memahami AI modern. Dengan memahami konsep dasar ML, kamu akan lebih mudah memahami:

- **Deep Learning**: Neural networks yang lebih kompleks
- **Computer Vision**: ML untuk memproses gambar
- **Natural Language Processing**: ML untuk bahasa
- **Reinforcement Learning**: AI yang belajar melalui interaksi

Konsep seperti loss functions, optimization, regularization, dan evaluation metrics yang kamu pelajari di ML akan sangat berguna saat masuk ke deep learning.

## Next Steps

---

Sekarang kamu sudah paham gambaran besar ML. Di file selanjutnya, kita akan mulai hands-on dengan data preprocessing dan exploratory data analysis - skill fundamental yang harus dikuasai sebelum masuk ke algoritma ML yang lebih kompleks.