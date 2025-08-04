# [data] #01 - Data World Introduction

![data](https://raw.githubusercontent.com/oujisan/OuVault/main/img/data.png)

## Understanding the Data World: Types, Formats, and Representation

---

Selamat datang di dunia data! Ini adalah langkah pertama yang krusial dalam perjalanan menuju machine learning. Data adalah bahan bakar dari AI - tanpa pemahaman yang solid tentang data, kita tidak akan bisa membangun model yang efektif.

## What is Data?

---

Data adalah kumpulan informasi yang dapat diukur, diamati, atau dikumpulkan. Dalam konteks machine learning, data adalah representasi digital dari fenomena dunia nyata yang akan kita gunakan untuk melatih algoritma.

### Karakteristik Data yang Baik:

- **Akurat**: Mencerminkan realitas dengan tepat
- **Lengkap**: Tidak ada informasi penting yang hilang
- **Konsisten**: Format dan struktur yang seragam
- **Relevan**: Berkaitan dengan masalah yang ingin diselesaikan
- **Terkini**: Up-to-date sesuai kebutuhan

## Data Types: Fundamental Classification

---

### 1. Numerical Data (Data Numerik)

Data yang dapat diukur dan dihitung dengan angka.

#### Continuous Data (Data Kontinu)

- Dapat mengambil nilai apa pun dalam rentang tertentu
- Contoh: tinggi badan (170.5 cm), suhu (25.7°C), harga saham ($45.23)

```python
import numpy as np
import pandas as pd

# Contoh data kontinu
tinggi_badan = [165.2, 170.8, 175.1, 168.9, 172.3]
suhu_harian = [25.5, 27.8, 24.2, 26.1, 28.3]

print("Tinggi badan:", tinggi_badan)
print("Range tinggi:", min(tinggi_badan), "-", max(tinggi_badan))
```

#### Discrete Data (Data Diskrit)

- Hanya dapat mengambil nilai tertentu (biasanya bilangan bulat)
- Contoh: jumlah anak (0, 1, 2, 3...), jumlah mobil terjual

```python
# Contoh data diskrit
jumlah_anak = [0, 2, 1, 3, 2, 1, 0, 4]
penjualan_mobil = [15, 23, 18, 31, 27, 19, 22]

print("Jumlah anak per keluarga:", jumlah_anak)
print("Nilai unik:", set(jumlah_anak))
```

### 2. Categorical Data (Data Kategorikal)

Data yang mengelompokkan objek ke dalam kategori atau kelas.

#### Nominal Data

- Kategori tanpa urutan natural
- Contoh: warna (merah, biru, hijau), jenis kelamin (pria, wanita)

```python
# Contoh data nominal
warna_favorit = ['merah', 'biru', 'hijau', 'merah', 'kuning', 'biru']
jenis_kelamin = ['pria', 'wanita', 'pria', 'wanita', 'pria']

# Menggunakan pandas untuk analisis kategorikal
df = pd.DataFrame({
    'warna': warna_favorit,
    'gender': jenis_kelamin
})

print(df['warna'].value_counts())
```

#### Ordinal Data

- Kategori dengan urutan yang bermakna
- Contoh: tingkat pendidikan (SD, SMP, SMA, S1), rating (1-5 bintang)

```python
# Contoh data ordinal
pendidikan = ['SMA', 'S1', 'SD', 'S2', 'SMP', 'S1', 'SMA']
rating = [5, 3, 4, 2, 5, 4, 3]

# Mapping ordinal ke numerical untuk analisis
edu_mapping = {'SD': 1, 'SMP': 2, 'SMA': 3, 'S1': 4, 'S2': 5}
pendidikan_numeric = [edu_mapping[edu] for edu in pendidikan]

print("Pendidikan mapping:", pendidikan_numeric)
```

## Data Formats and Structures

---

### 1. Structured Data

Data yang terorganisir dalam format tabel dengan baris dan kolom yang jelas.

```python
# Contoh structured data dengan pandas
data_siswa = {
    'nama': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'umur': [20, 22, 21, 23],
    'nilai': [85.5, 92.0, 78.5, 96.2],
    'jurusan': ['Informatika', 'Matematika', 'Fisika', 'Informatika']
}

df_siswa = pd.DataFrame(data_siswa)
print(df_siswa)
print("\nInfo dataset:")
print(df_siswa.info())
```

### 2. Semi-structured Data

Data yang memiliki struktur tetapi tidak dalam format tabel rigid.

```python
import json

# Contoh semi-structured data (JSON)
data_json = {
    "mahasiswa": [
        {
            "id": 1,
            "nama": "Alice",
            "mata_kuliah": ["Algoritma", "Database"],
            "nilai": {"Algoritma": 85, "Database": 90}
        },
        {
            "id": 2,
            "nama": "Bob", 
            "mata_kuliah": ["Machine Learning", "Statistik"],
            "nilai": {"Machine Learning": 88, "Statistik": 92}
        }
    ]
}

# Convert ke DataFrame
import pandas as pd
from pandas import json_normalize

df_json = json_normalize(data_json['mahasiswa'])
print(df_json)
```

### 3. Unstructured Data

Data yang tidak memiliki struktur yang jelas seperti teks, gambar, audio.

```python
# Contoh handling unstructured data (text)
import re

teks_review = [
    "Produk ini sangat bagus dan berkualitas tinggi!",
    "Pelayanan kurang memuaskan, akan coba tempat lain.",
    "Harga terjangkau dengan kualitas yang ok.",
    "Sangat recommended! Akan beli lagi."
]

# Simple text processing
def analisis_sentimen_sederhana(teks):
    kata_positif = ['bagus', 'berkualitas', 'recommended', 'terjangkau']
    kata_negatif = ['kurang', 'tidak', 'buruk', 'jelek']
    
    skor_positif = sum(1 for kata in kata_positif if kata in teks.lower())
    skor_negatif = sum(1 for kata in kata_negatif if kata in teks.lower())
    
    if skor_positif > skor_negatif:
        return 'positif'
    elif skor_negatif > skor_positif:
        return 'negatif'
    else:
        return 'netral'

for i, review in enumerate(teks_review):
    sentimen = analisis_sentimen_sederhana(review)
    print(f"Review {i+1}: {sentimen}")
```

## Data Representation in Python

---

### 1. Lists and Arrays

```python
# Python Lists
data_list = [1, 2, 3, 4, 5]
mixed_list = [1, 'hello', 3.14, True]

# NumPy Arrays (lebih efisien untuk numerical data)
import numpy as np

# 1D Array
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr_1d)
print("Shape:", arr_1d.shape)
print("Data type:", arr_1d.dtype)

# 2D Array (Matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Array:")
print(arr_2d)
print("Shape:", arr_2d.shape)
```

### 2. Pandas DataFrames

```python
# DataFrame adalah struktur data utama untuk machine learning
import pandas as pd

# Membuat DataFrame dari dictionary
data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': ['x', 'y', 'x', 'z']
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)

# Basic information
print("\nDataFrame Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")
```

### 3. Series (1D labeled data)

```python
# Series adalah struktur 1D dalam pandas
series_data = pd.Series([10, 20, 30, 40], 
                       index=['a', 'b', 'c', 'd'], 
                       name='values')

print("Series:")
print(series_data)
print(f"\nAccess by index 'b': {series_data['b']}")
```

## Data Quality Assessment

---

### Identifying Data Issues

```python
# Contoh dataset dengan berbagai issues
data_messy = {
    'nama': ['Alice', 'Bob', '', 'Diana', 'Eve'],
    'umur': [25, None, 30, 28, 35],
    'gaji': [50000, 60000, 55000, None, 70000],
    'department': ['IT', 'Finance', 'IT', 'HR', 'Finance']
}

df_messy = pd.DataFrame(data_messy)
print("Dataset dengan issues:")
print(df_messy)

# Check for missing values
print("\nMissing values:")
print(df_messy.isnull().sum())

# Check for empty strings
print("\nEmpty strings:")
print((df_messy == '').sum())

# Basic statistics
print("\nBasic statistics:")
print(df_messy.describe())
```

## Common Data Sources

---

### 1. CSV Files

```python
# Membaca CSV file
# df = pd.read_csv('data.csv')

# Simulasi membaca CSV
csv_data = """nama,umur,kota
Alice,25,Jakarta
Bob,30,Bandung
Charlie,28,Surabaya"""

from io import StringIO
df_csv = pd.read_csv(StringIO(csv_data))
print("Data from CSV:")
print(df_csv)
```

### 2. Excel Files

```python
# Untuk Excel files (membutuhkan openpyxl atau xlrd)
# df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Simulasi Excel data
excel_data = {
    'Produk': ['Laptop', 'Mouse', 'Keyboard'],
    'Harga': [15000000, 250000, 500000],
    'Stok': [10, 50, 30]
}

df_excel = pd.DataFrame(excel_data)
print("Excel-like data:")
print(df_excel)
```

### 3. APIs and Web Data

```python
# Contoh struktur data dari API
api_response = {
    "status": "success",
    "data": [
        {"id": 1, "temperature": 25.5, "humidity": 60, "timestamp": "2024-01-01T10:00:00"},
        {"id": 2, "temperature": 26.0, "humidity": 58, "timestamp": "2024-01-01T11:00:00"}
    ]
}

# Convert API response to DataFrame
df_api = pd.DataFrame(api_response['data'])
print("API data:")
print(df_api)
```

## Best Practices for Data Handling

---

### 1. Data Validation

```python
def validate_data(df):
    """Simple data validation function"""
    issues = []
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        issues.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
    
    # Check for duplicate rows
    if df.duplicated().any():
        issues.append(f"Duplicate rows found: {df.duplicated().sum()}")
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].nunique()
            total_vals = len(df[col])
            if unique_vals == total_vals:
                issues.append(f"Column '{col}' might be an identifier (all unique values)")
    
    return issues

# Test validation
validation_results = validate_data(df_messy)
for issue in validation_results:
    print(f"⚠️ {issue}")
```

### 2. Data Documentation

```python
def document_dataset(df, dataset_name="Unknown"):
    """Generate basic documentation for a dataset"""
    print(f"📊 Dataset Documentation: {dataset_name}")
    print("=" * 50)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"Created: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📋 Column Information:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        
        print(f"  • {col}: {dtype} ({null_count} nulls, {null_pct:.1f}%)")

# Example usage
document_dataset(df_siswa, "Student Data")
```

## Summary dan Next Steps

---

Dalam modul ini, kita telah mempelajari:

1. **Tipe-tipe data**: Numerical (continuous/discrete) dan Categorical (nominal/ordinal)
2. **Format data**: Structured, semi-structured, dan unstructured
3. **Representasi data** dalam Python menggunakan lists, arrays, dan DataFrames
4. **Penilaian kualitas data** dan identifikasi masalah umum
5. **Sumber data** yang umum digunakan dalam machine learning
6. **Best practices** untuk penanganan data

### 🎯 Key Takeaways:

- Data adalah fondasi dari semua proyek machine learning
- Memahami tipe dan struktur data sangat penting untuk memilih algoritma yang tepat
- Pandas dan NumPy adalah tools utama untuk manipulasi data di Python
- Kualitas data menentukan kualitas model yang akan kita bangun

### 🔜 Selanjutnya:

Di modul berikutnya, kita akan belajar manipulasi data menggunakan NumPy dan Pandas secara lebih mendalam, termasuk teknik cleaning, transformation, dan preparation data untuk machine learning.