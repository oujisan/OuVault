# [ai] #01 - AI Introduction & Fundamentals
![banner](https://raw.githubusercontent.com/oujisan/OuVault/main/img/py-ai.png)
## What is Artificial Intelligence?

---

Artificial Intelligence (AI) adalah kemampuan mesin atau komputer untuk meniru kecerdasan manusia dalam menyelesaikan masalah, mengambil keputusan, dan belajar dari pengalaman. AI mencakup berbagai teknik dan pendekatan untuk membuat sistem yang dapat berpikir dan bertindak secara cerdas.

### Sejarah Singkat AI

AI pertama kali diperkenalkan pada tahun 1956 oleh John McCarthy. Sejak saat itu, AI telah mengalami perkembangan pesat dengan berbagai breakthrough seperti Deep Learning, Machine Learning, dan Neural Networks.

## Types of AI
---
### 1. Narrow AI (Weak AI)

AI yang dirancang untuk tugas spesifik seperti:

- Voice assistants (Siri, Alexa)
- Image recognition
- Game playing (Chess, Go)

### 2. General AI (Strong AI)

AI yang memiliki kemampuan kognitif setara dengan manusia di semua domain. Masih dalam tahap penelitian dan belum tercapai.

### 3. Super AI

AI yang melampaui kemampuan manusia di semua aspek. Masih konsep teoretis.

## Core Components of AI

---

### Machine Learning (ML)

Kemampuan mesin untuk belajar dan meningkatkan performa tanpa diprogram secara eksplisit.

```python
# Contoh sederhana machine learning
from sklearn.linear_model import LinearRegression
import numpy as np

# Data training
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Membuat model
model = LinearRegression()
model.fit(X, y)

# Prediksi
prediction = model.predict([[6]])
print(f"Prediksi untuk input 6: {prediction[0]}")
```

### Deep Learning

Subset dari machine learning yang menggunakan neural networks dengan banyak layer untuk meniru cara kerja otak manusia.

### Natural Language Processing (NLP)

Kemampuan komputer untuk memahami, memproses, dan menghasilkan bahasa manusia.

### Computer Vision

Kemampuan komputer untuk "melihat" dan memahami konten visual seperti gambar dan video.

## Python for AI Development

---

Python menjadi bahasa pemrograman pilihan untuk AI karena:

### Keunggulan Python dalam AI:

- **Sintaks sederhana**: Mudah dipelajari dan dibaca
- **Library lengkap**: NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch
- **Komunitas besar**: Support dan dokumentasi yang luas
- **Fleksibilitas**: Cocok untuk prototyping dan production

### Essential Libraries untuk AI:

```python
# Data manipulation
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Deep Learning
import tensorflow as tf
import torch

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

## Setting Up AI Development Environment

---

### 1. Install Python

Pastikan Python 3.8+ terinstall di sistem Anda.

### 2. Virtual Environment

```python
# Membuat virtual environment
python -m venv ai_env

# Aktivasi (Windows)
ai_env\Scripts\activate

# Aktivasi (macOS/Linux)
source ai_env/bin/activate
```

### 3. Install Essential Libraries

```python
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install tensorflow
pip install torch torchvision
pip install jupyter notebook
```

## First AI Project: Simple Prediction

---

Mari kita buat proyek AI sederhana untuk prediksi harga rumah:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
size = np.random.normal(150, 50, 100)  # Ukuran rumah (m2)
location_score = np.random.uniform(1, 10, 100)  # Skor lokasi
price = size * 2000 + location_score * 50000 + np.random.normal(0, 20000, 100)

# Membuat DataFrame
data = pd.DataFrame({
    'size': size,
    'location_score': location_score,
    'price': price
})

# Mempersiapkan data
X = data[['size', 'location_score']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualisasi hasil
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('House Price Prediction Results')
plt.show()
```

## Key Takeaways

---

1. **AI adalah masa depan**: Teknologi yang akan mengubah banyak industri
2. **Python adalah tools terbaik**: Untuk memulai journey AI development
3. **Start small**: Mulai dengan proyek sederhana sebelum ke yang kompleks
4. **Practice makes perfect**: Konsistensi dalam belajar dan praktik sangat penting

Pada file selanjutnya, kita akan membahas lebih dalam tentang Machine Learning dan implementasinya dengan Python!