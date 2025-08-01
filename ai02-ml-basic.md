# [python] #02 - AI Machine Learning Basics
![banner](https://raw.githubusercontent.com/oujisan/OuVault/main/img/py-ai.png)
## Introduction to Machine Learning

---

Machine Learning adalah cabang AI yang memungkinkan komputer belajar dan membuat keputusan dari data tanpa diprogram secara eksplisit untuk setiap tugas spesifik. ML menggunakan algoritma untuk menganalisis pola dalam data dan membuat prediksi atau klasifikasi.

### Mengapa Machine Learning Penting?

- **Otomatisasi**: Mengotomatisasi tugas-tugas kompleks
- **Skalabilitas**: Menangani data dalam jumlah besar
- **Adaptasi**: Belajar dari data baru secara otomatis
- **Akurasi**: Sering lebih akurat dari rule-based systems

## Types of Machine Learning

---

### 1. Supervised Learning

Belajar dari data yang sudah memiliki label atau target yang diketahui.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict dan evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

**Contoh kasus**: Email spam detection, image classification, stock price prediction

### 2. Unsupervised Learning

Belajar dari data tanpa label untuk menemukan pola tersembunyi.

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualisasi
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering')
plt.show()
```

**Contoh kasus**: Customer segmentation, anomaly detection, data compression

### 3. Reinforcement Learning

Belajar melalui trial and error dengan sistem reward dan punishment.

```python
import numpy as np
import random

class SimpleQLearning:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.95):
        self.q_table = np.zeros((states, actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        
    def choose_action(self, state, epsilon=0.1):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)
        else:
            return np.argmax(self.q_table[state, :])
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action] = new_value

# Contoh penggunaan
agent = SimpleQLearning(states=5, actions=2)
print("Q-Learning Agent initialized!")
```

**Contoh kasus**: Game AI, robotics, autonomous vehicles

## Key Machine Learning Algorithms

---

### 1. Linear Regression

Untuk prediksi nilai kontinyu dengan hubungan linear.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Sample data: Hours studied vs Exam score
hours = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
scores = np.array([50, 55, 65, 70, 80, 85, 90, 95])

# Train model
model = LinearRegression()
model.fit(hours, scores)

# Predict
new_hours = np.array([[9], [10]])
predictions = model.predict(new_hours)

print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Predictions for 9 and 10 hours: {predictions}")
```

### 2. Logistic Regression

Untuk klasifikasi binary atau multiclass.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample data: Student scores and pass/fail
study_hours = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
pass_fail = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])  # 0: Fail, 1: Pass

# Train model
log_model = LogisticRegression()
log_model.fit(study_hours, pass_fail)

# Predict probability
probabilities = log_model.predict_proba(study_hours)
print("Probabilitas lulus untuk setiap jam belajar:")
for i, prob in enumerate(probabilities):
    print(f"Jam ke-{i+1}: {prob[1]:.2f}")
```

### 3. Decision Trees

Model yang mudah diinterpretasi dengan struktur seperti pohon.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

# Weather data: [Temperature, Humidity, Wind] -> Play Tennis?
weather_data = [
    [85, 85, 0], [80, 90, 1], [83, 86, 0], [70, 96, 0],
    [68, 80, 0], [65, 70, 1], [64, 65, 1], [72, 95, 0],
    [69, 70, 0], [75, 80, 0], [75, 70, 1], [72, 90, 1]
]
play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]

# Train model
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(weather_data, play_tennis)

# Print rules
feature_names = ['Temperature', 'Humidity', 'Wind']
tree_rules = export_text(tree_model, feature_names=feature_names)
print("Decision Tree Rules:")
print(tree_rules)
```

## Data Preprocessing

---

### 1. Data Cleaning

```python
import pandas as pd
import numpy as np

# Sample data dengan missing values
data = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 40],
    'salary': [50000, np.nan, 70000, 80000, 90000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
})

print("Data asli:")
print(data)

# Handle missing values
data['age'].fillna(data['age'].mean(), inplace=True)
data['salary'].fillna(data['salary'].median(), inplace=True)

print("\nData setelah cleaning:")
print(data)
```

### 2. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample data
data = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])

# Standardization (mean=0, std=1)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

# Normalization (min=0, max=1)
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(data)

print("Original data:")
print(data)
print("\nStandardized data:")
print(standardized_data)
print("\nNormalized data:")
print(normalized_data)
```

### 3. Feature Engineering

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'medium', 'large', 'medium', 'small'],
    'price': [10, 15, 20, 12, 8]
})

# One-hot encoding untuk categorical variables
df_encoded = pd.get_dummies(df, columns=['color', 'size'])
print("One-hot encoded data:")
print(df_encoded)

# Label encoding
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
print("\nLabel encoded data:")
print(df[['color', 'color_encoded']])
```

## Model Evaluation

---

### 1. Train-Validation-Test Split

```python
from sklearn.model_selection import train_test_split

# Assuming we have X (features) and y (target)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples") 
print(f"Test set: {len(X_test)} samples")
```

### 2. Cross Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load sample dataset
from sklearn.datasets import load_wine
wine = load_wine()
X, y = wine.data, wine.target

# Cross validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### 3. Metrics untuk Evaluasi

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming we have y_true and y_pred
# y_true = [0, 1, 0, 1, 1, 0, 1, 0]
# y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# evaluate_model(y_true, y_pred)
```

## Practical Project: Customer Segmentation

---

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate sample customer data
np.random.seed(42)
n_customers = 200

data = pd.DataFrame({
    'annual_spending': np.random.normal(5000, 2000, n_customers),
    'frequency_visits': np.random.poisson(12, n_customers),
    'avg_purchase': np.random.normal(150, 50, n_customers),
    'age': np.random.randint(18, 70, n_customers)
})

# Data preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to original data
data['cluster'] = clusters

# Analyze clusters
print("Analisis Segmentasi Customer:")
for i in range(3):
    cluster_data = data[data['cluster'] == i]
    print(f"\nCluster {i}:")
    print(f"- Rata-rata pengeluaran tahunan: ${cluster_data['annual_spending'].mean():.2f}")
    print(f"- Rata-rata kunjungan: {cluster_data['frequency_visits'].mean():.1f} kali/tahun")
    print(f"- Rata-rata pembelian: ${cluster_data['avg_purchase'].mean():.2f}")
    print(f"- Rata-rata umur: {cluster_data['age'].mean():.1f} tahun")

# Visualisasi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data['annual_spending'], data['frequency_visits'], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel('Annual Spending ($)')
plt.ylabel('Frequency of Visits')
plt.title('Customer Segmentation')

plt.subplot(1, 2, 2)
plt.scatter(data['age'], data['avg_purchase'], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Average Purchase ($)')
plt.title('Age vs Purchase Behavior')

plt.tight_layout()
plt.show()
```

## Next Steps

---

Setelah memahami dasar-dasar Machine Learning, langkah selanjutnya adalah:

1. **Praktek lebih banyak**: Coba berbagai dataset dan algoritma
2. **Pelajari Deep Learning**: Neural networks dan aplikasinya
3. **Explore specialized domains**: Computer Vision, NLP, Time Series
4. **Build portfolio**: Buat proyek-proyek yang bisa dipamerkan

Pada file selanjutnya, kita akan membahas Deep Learning dan Neural Networks secara mendalam!