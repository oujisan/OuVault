# [data] #06 - Data Cleaning

![data](https://raw.githubusercontent.com/oujisan/OuVault/main/img/data.png)

## Introduction to Data Cleaning

---

Data cleaning adalah proses penting yang harus dilakukan sebelum membangun model machine learning. Bayangkan data sebagai bahan makanan - sebelum memasak, kita perlu mencuci dan menyiapkan bahan dengan baik agar hasil masakan optimal. Begitu juga dengan data, kita perlu memastikan data bersih dan berkualitas tinggi.

Dalam dunia nyata, data jarang sekali sempurna. Kita akan menemui berbagai masalah seperti data hilang (missing values), duplikasi, atau inkonsistensi format. Jika tidak ditangani dengan baik, masalah ini bisa membuat model machine learning kita tidak akurat atau bahkan gagal total.

## Handling Missing Values (Null Data)

---

Missing values adalah salah satu tantangan paling umum dalam data cleaning. Ada beberapa strategi yang bisa kita gunakan:

### Identifikasi Missing Values

```python
import pandas as pd
import numpy as np

# Membaca dataset
df = pd.read_csv('dataset.csv')

# Melihat jumlah missing values per kolom
print(df.isnull().sum())

# Melihat persentase missing values
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent)

# Visualisasi missing values
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
```

### Strategi Penanganan Missing Values

**1. Deletion (Menghapus Data)**

```python
# Menghapus baris dengan missing values
df_cleaned = df.dropna()

# Menghapus kolom dengan missing values
df_cleaned = df.dropna(axis=1)

# Menghapus baris jika missing values lebih dari threshold tertentu
df_cleaned = df.dropna(thresh=len(df.columns) * 0.7)
```

**2. Imputation (Mengisi Data)**

```python
from sklearn.impute import SimpleImputer

# Mengisi dengan mean untuk data numerik
imputer_num = SimpleImputer(strategy='mean')
df['numerical_column'] = imputer_num.fit_transform(df[['numerical_column']])

# Mengisi dengan mode untuk data kategorikal
imputer_cat = SimpleImputer(strategy='most_frequent')
df['categorical_column'] = imputer_cat.fit_transform(df[['categorical_column']])

# Mengisi dengan nilai konstant
df['column'].fillna('Unknown', inplace=True)

# Forward fill dan backward fill
df['column'].fillna(method='ffill', inplace=True)  # Menggunakan nilai sebelumnya
df['column'].fillna(method='bfill', inplace=True)  # Menggunakan nilai sesudahnya
```

**3. Advanced Imputation**

```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# KNN Imputation - menggunakan tetangga terdekat
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(knn_imputer.fit_transform(df.select_dtypes(include=[np.number])))

# Iterative Imputation - menggunakan model prediksi
iterative_imputer = IterativeImputer(random_state=42)
df_iterative = pd.DataFrame(iterative_imputer.fit_transform(df.select_dtypes(include=[np.number])))
```

## Handling Duplicate Data

---

Data duplikat bisa muncul karena error dalam pengumpulan data atau proses input yang berulang. Ini bisa membuat model kita bias karena menganggap satu observasi lebih penting dari yang seharusnya.

### Identifikasi dan Penanganan Duplikat

```python
# Mengecek duplikat
print(f"Jumlah baris duplikat: {df.duplicated().sum()}")

# Melihat baris duplikat
duplicate_rows = df[df.duplicated()]
print(duplicate_rows)

# Menghapus duplikat
df_no_duplicates = df.drop_duplicates()

# Menghapus duplikat berdasarkan kolom tertentu saja
df_no_duplicates = df.drop_duplicates(subset=['column1', 'column2'])

# Mempertahankan duplikat terakhir alih-alih yang pertama
df_no_duplicates = df.drop_duplicates(keep='last')
```

### Duplikat Partial (Fuzzy Duplicates)

```python
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def find_fuzzy_duplicates(df, column, threshold=80):
    """
    Mencari duplikat yang tidak persis sama tapi mirip
    """
    duplicates = []
    for i, name1 in enumerate(df[column]):
        for j, name2 in enumerate(df[column]):
            if i < j:  # Hindari perbandingan yang sama
                similarity = fuzz.ratio(str(name1), str(name2))
                if similarity > threshold:
                    duplicates.append((i, j, name1, name2, similarity))
    return duplicates

# Contoh penggunaan
fuzzy_dupes = find_fuzzy_duplicates(df, 'company_name', 85)
print(fuzzy_dupes)
```

## Handling Inconsistent Data

---

Data tidak konsisten bisa berupa format yang berbeda, typo, atau standar yang tidak seragam. Ini perlu diperbaiki agar model bisa memproses data dengan benar.

### Standardisasi Format

```python
# Standardisasi teks
df['name'] = df['name'].str.lower().str.strip()  # Lowercase dan hapus spasi
df['name'] = df['name'].str.title()  # Title case

# Standardisasi tanggal
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Standardisasi angka
df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Standardisasi kategori
mapping = {'Y': 'Yes', 'N': 'No', '1': 'Yes', '0': 'No'}
df['column'] = df['column'].map(mapping)
```

### Penanganan Outlier

```python
# Identifikasi outlier dengan IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Identifikasi outlier dengan Z-score
from scipy import stats

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = df[z_scores > threshold]
    return outliers

# Penanganan outlier
outliers = detect_outliers_iqr(df, 'price')
print(f"Jumlah outlier: {len(outliers)}")

# Opsi 1: Menghapus outlier
df_no_outliers = df.drop(outliers.index)

# Opsi 2: Capping outlier
df['price_capped'] = np.where(df['price'] > upper_bound, upper_bound, df['price'])
df['price_capped'] = np.where(df['price_capped'] < lower_bound, lower_bound, df['price_capped'])

# Opsi 3: Transformasi log
df['price_log'] = np.log1p(df['price'])
```

## Data Validation & Quality Check

---

Setelah cleaning, penting untuk memvalidasi bahwa data kita sudah berkualitas baik.

```python
def data_quality_report(df):
    """
    Membuat laporan kualitas data
    """
    report = {}
    
    # Basic info
    report['total_rows'] = len(df)
    report['total_columns'] = len(df.columns)
    
    # Missing values
    report['missing_values'] = df.isnull().sum().sum()
    report['missing_percentage'] = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    # Duplicates
    report['duplicate_rows'] = df.duplicated().sum()
    
    # Data types
    report['data_types'] = df.dtypes.value_counts().to_dict()
    
    # Unique values per column
    report['unique_values'] = {col: df[col].nunique() for col in df.columns}
    
    return report

# Generate report
quality_report = data_quality_report(df)
print(quality_report)
```

## Best Practices & Tips

---

1. **Dokumentasi**: Selalu dokumentasikan langkah-langkah cleaning yang dilakukan

```python
# Buat log cleaning steps
cleaning_log = []

def log_cleaning_step(step_description, rows_before, rows_after):
    log_entry = {
        'step': step_description,
        'rows_before': rows_before,
        'rows_after': rows_after,
        'rows_removed': rows_before - rows_after
    }
    cleaning_log.append(log_entry)
    print(f"{step_description}: {rows_before} → {rows_after} rows ({rows_before - rows_after} removed)")

# Contoh penggunaan
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
log_cleaning_step("Remove missing values", rows_before, rows_after)
```

2. **Backup Data**: Selalu simpan versi original data sebelum cleaning

```python
df_original = df.copy()  # Backup data original
```

3. **Iterative Process**: Data cleaning adalah proses iteratif, jangan takut untuk kembali dan memperbaiki
    
4. **Domain Knowledge**: Gunakan pemahaman bisnis untuk menentukan strategi cleaning yang tepat
    

Data cleaning yang baik adalah fondasi dari model machine learning yang sukses. Ingat, "Garbage in, garbage out" - jika data input kita buruk, hasil model juga akan buruk. Luangkan waktu yang cukup untuk tahap ini, karena investasi waktu di awal akan menghemat banyak waktu dan frustasi nanti!