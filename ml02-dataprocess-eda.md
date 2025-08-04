# [ai] #02 - ML Data Preprocessing and EDA

![ml](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ml.png)

## Understanding Your Data

---

Sebelum terjun ke algoritma ML yang fancy, kita harus benar-benar mengenal data kita. Ini seperti seorang chef yang harus tahu kualitas bahan-bahannya sebelum memasak. Data preprocessing sering kali memakan 70-80% waktu dalam project ML, tapi ini adalah langkah yang paling krusial.

## Exploratory Data Analysis (EDA)

---

EDA adalah proses investigasi data untuk memahami struktur, pola, dan anomali yang ada. Think of it sebagai detective work sebelum kita mulai modeling.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Basic info about dataset
print("Dataset shape:", df.shape)
print("\nColumn info:")
print(df.info())

# Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check data types
print("\nData types:")
print(df.dtypes)
```

### Visualizing Data Distribution

```python
# Distribution of numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i+1)
    df[col].hist(bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

## Handling Missing Values

---

Missing data adalah masalah umum yang harus ditangani dengan hati-hati. Ada beberapa strategi yang bisa digunakan:

### 1. Remove Missing Data

```python
# Remove rows with any missing value
df_dropped = df.dropna()

# Remove columns with too many missing values (>50%)
threshold = len(df) * 0.5
df_dropped_cols = df.dropna(axis=1, thresh=threshold)

# Remove rows where specific column is missing
df_specific = df.dropna(subset=['important_column'])
```

### 2. Imputation (Fill Missing Values)

```python
from sklearn.impute import SimpleImputer

# Fill with mean (for numerical data)
numerical_imputer = SimpleImputer(strategy='mean')
df['numerical_column'] = numerical_imputer.fit_transform(df[['numerical_column']])

# Fill with mode (for categorical data)
categorical_imputer = SimpleImputer(strategy='most_frequent')
df['categorical_column'] = categorical_imputer.fit_transform(df[['categorical_column']])

# Fill with median (robust to outliers)
df['column'].fillna(df['column'].median(), inplace=True)

# Forward fill (useful for time series)
df['column'].fillna(method='ffill', inplace=True)

# Custom value
df['column'].fillna('Unknown', inplace=True)
```

### 3. Advanced Imputation

```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(knn_imputer.fit_transform(df.select_dtypes(include=[np.number])), 
                      columns=df.select_dtypes(include=[np.number]).columns)

# Iterative Imputation (MICE)
iterative_imputer = IterativeImputer(random_state=42)
df_iterative = pd.DataFrame(iterative_imputer.fit_transform(df.select_dtypes(include=[np.number])), 
                           columns=df.select_dtypes(include=[np.number]).columns)
```

## Handling Outliers

---

Outliers bisa memberikan insight penting atau malah noise yang mengganggu model. Kita perlu identify dan decide apakah akan dipertahankan atau dihilangkan.

### Detection Methods

```python
# IQR Method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)  
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Z-score method
from scipy import stats
def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = df[z_scores > threshold]
    return outliers

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df.boxplot(column='numerical_column')
plt.title('Boxplot - Shows Outliers')

plt.subplot(1, 2, 2)
df['numerical_column'].hist(bins=30)
plt.title('Histogram - Shows Distribution')
plt.show()
```

### Handling Outliers

```python
# Remove outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Cap outliers (winsorization)
def cap_outliers(df, column, percentile=0.95):
    upper_limit = df[column].quantile(percentile)
    lower_limit = df[column].quantile(1 - percentile)
    
    df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
    df[column] = np.where(df[column] < lower_limit, lower_limit, df[column])
    
    return df

# Transform data (log transformation)
df['log_column'] = np.log1p(df['skewed_column'])  # log1p untuk handle nilai 0
```

## Feature Scaling

---

Banyak algoritma ML sensitif terhadap skala data. Bayangkan membandingkan gaji (jutaan) dengan umur (puluhan) - algoritma akan bias ke feature yang nilainya lebih besar.

### Standardization (Z-score normalization)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=[np.number])), 
                        columns=df.select_dtypes(include=[np.number]).columns)

# Manual calculation
# standardized = (x - mean) / std
```

### Min-Max Normalization

```python
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(minmax_scaler.fit_transform(df.select_dtypes(include=[np.number])), 
                            columns=df.select_dtypes(include=[np.number]).columns)

# Manual calculation
# normalized = (x - min) / (max - min)
```

### Robust Scaling

```python
from sklearn.preprocessing import RobustScaler

# Less sensitive to outliers
robust_scaler = RobustScaler()
df_robust = pd.DataFrame(robust_scaler.fit_transform(df.select_dtypes(include=[np.number])), 
                        columns=df.select_dtypes(include=[np.number]).columns)
```

## Encoding Categorical Variables

---

Machine Learning algorithms bekerja dengan angka, jadi kita perlu convert categorical data ke numerical.

### One-Hot Encoding

```python
# Untuk categorical variables tanpa urutan
df_encoded = pd.get_dummies(df, columns=['categorical_column'], drop_first=True)

# Using sklearn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(df[['categorical_column']])
encoded_df = pd.DataFrame(encoded_features, 
                         columns=encoder.get_feature_names_out(['categorical_column']))
```

### Label Encoding

```python
# Untuk ordinal data (ada urutan)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['encoded_column'] = label_encoder.fit_transform(df['categorical_column'])

# Manual ordinal encoding
size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
df['size_encoded'] = df['size'].map(size_mapping)
```

### Target Encoding

```python
# Encode based on target variable (hati-hati dengan overfitting)
def target_encode(df, column, target):
    target_means = df.groupby(column)[target].mean()
    df[f'{column}_encoded'] = df[column].map(target_means)
    return df
```

## Feature Engineering

---

Membuat features baru dari data yang ada untuk memberikan informasi yang lebih berguna ke model.

```python
# Date features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Mathematical transformations
df['feature_squared'] = df['feature'] ** 2
df['feature_log'] = np.log1p(df['feature'])
df['feature_sqrt'] = np.sqrt(df['feature'])

# Interaction features
df['feature1_x_feature2'] = df['feature1'] * df['feature2']
df['feature1_div_feature2'] = df['feature1'] / (df['feature2'] + 1e-8)  # avoid division by zero

# Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], 
                        labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Text features (basic)
df['text_length'] = df['text_column'].str.len()
df['word_count'] = df['text_column'].str.split().str.len()
```

## Data Quality Checks

---

```python
def data_quality_report(df):
    print("=== DATA QUALITY REPORT ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n--- Missing Values ---")
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    missing_table = pd.DataFrame({
        'Missing Count': missing_data,
        'Percentage': missing_percent
    })
    print(missing_table[missing_table['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
    
    print("\n--- Duplicate Rows ---")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())
    
    print("\n--- Potential Issues ---")
    # Check for columns with single value
    single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
    if single_value_cols:
        print(f"Columns with single value: {single_value_cols}")
    
    # Check for high cardinality categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
    if high_cardinality:
        print(f"High cardinality categorical columns: {high_cardinality}")

# Run the report
data_quality_report(df)
```

## Best Practices

---

1. **Always split data before preprocessing** - hindari data leakage
2. **Save preprocessing steps** - untuk consistency saat deploy
3. **Document your decisions** - kenapa remove/keep outliers, pilih imputation method, etc.
4. **Validate preprocessing results** - check distribusi sebelum dan sesudah
5. **Consider domain knowledge** - jangan cuma rely on statistical methods

Data preprocessing yang baik adalah fondasi model ML yang baik. Di file selanjutnya, kita akan mulai masuk ke algoritma supervised learning yang sesungguhnya!