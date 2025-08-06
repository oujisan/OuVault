# [data] #07 - Data Feature Engineering

![data](https://raw.githubusercontent.com/oujisan/OuVault/main/img/data.png)

## Introduction to Feature Engineering

---

Feature engineering adalah seni dan sains dalam menciptakan fitur (variabel) baru dari data yang sudah ada untuk meningkatkan performa model machine learning. Bayangkan seperti seorang chef yang mengolah bahan mentah menjadi hidangan lezat - kita mengubah data mentah menjadi "hidangan" yang lebih mudah dicerna oleh algoritma.

Fitur yang baik bisa membuat perbedaan dramatis pada performa model. Kadang-kadang, feature engineering yang pintar bisa membuat model sederhana mengalahkan model kompleks dengan fitur yang buruk. Ini adalah skill yang sangat berharga dalam machine learning!

## Understanding Features

---

### Types of Features

```python
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Contoh dataset untuk demonstrasi
np.random.seed(42)
data = {
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'purchase_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'city': np.random.choice(['Jakarta', 'Surabaya', 'Bandung', 'Medan'], 1000),
    'purchase_amount': np.random.gamma(2, 50, 1000)
}
df = pd.DataFrame(data)

# Melihat tipe data
print(df.dtypes)
print("\nInfo dataset:")
print(df.info())
```

### Feature Categories

1. **Numerical Features**: Data angka (age, income, purchase_amount)
2. **Categorical Features**: Data kategori (education, city)
3. **Temporal Features**: Data waktu (purchase_date)
4. **Text Features**: Data teks
5. **Boolean Features**: Data true/false

## Creating New Features from Existing Data

---

### Numerical Feature Engineering

```python
# 1. Mathematical Transformations
df['income_log'] = np.log1p(df['income'])  # Log transformation
df['income_sqrt'] = np.sqrt(df['income'])   # Square root
df['income_squared'] = df['income'] ** 2    # Squared

# 2. Binning/Discretization
df['age_group'] = pd.cut(df['age'], 
                        bins=[0, 25, 35, 50, 100], 
                        labels=['Young', 'Adult', 'Middle', 'Senior'])

df['income_bracket'] = pd.qcut(df['income'], 
                              q=5, 
                              labels=['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High'])

# 3. Ratios and Proportions
df['purchase_to_income_ratio'] = df['purchase_amount'] / df['income']

# 4. Aggregations and Rolling Statistics
# (Untuk time series data)
df['rolling_avg_7d'] = df['purchase_amount'].rolling(window=7).mean()
df['cumulative_purchases'] = df['purchase_amount'].cumsum()
```

### Categorical Feature Engineering

```python
# 1. One-Hot Encoding
education_encoded = pd.get_dummies(df['education'], prefix='education')
df = pd.concat([df, education_encoded], axis=1)

# 2. Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

# 3. Target Encoding (Mean Encoding)
def target_encode(df, categorical_col, target_col):
    mean_target = df.groupby(categorical_col)[target_col].mean()
    return df[categorical_col].map(mean_target)

# Contoh jika kita punya target variable
df['city_mean_purchase'] = target_encode(df, 'city', 'purchase_amount')

# 4. Frequency Encoding
city_counts = df['city'].value_counts()
df['city_frequency'] = df['city'].map(city_counts)

# 5. Rare Category Handling
def handle_rare_categories(series, min_freq=50):
    counts = series.value_counts()
    rare_categories = counts[counts < min_freq].index
    return series.replace(rare_categories, 'Other')

df['city_cleaned'] = handle_rare_categories(df['city'])
```

### Temporal Feature Engineering

```python
# Extract various time components
df['year'] = df['purchase_date'].dt.year
df['month'] = df['purchase_date'].dt.month
df['day'] = df['purchase_date'].dt.day
df['day_of_week'] = df['purchase_date'].dt.dayofweek
df['day_of_year'] = df['purchase_date'].dt.dayofyear
df['week_of_year'] = df['purchase_date'].dt.isocalendar().week
df['quarter'] = df['purchase_date'].dt.quarter

# Cyclical features (untuk capture periodicity)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Boolean time features
df['is_weekend'] = df['day_of_week'].isin([5, 6])
df['is_month_start'] = df['purchase_date'].dt.is_month_start
df['is_month_end'] = df['purchase_date'].dt.is_month_end

# Time since/until features
reference_date = pd.to_datetime('2023-01-01')
df['days_since_reference'] = (df['purchase_date'] - reference_date).dt.days
```

## Advanced Feature Engineering Techniques

---

### Polynomial and Interaction Features

```python
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
numerical_cols = ['age', 'income']
poly_features = poly.fit_transform(df[numerical_cols])

# Get feature names
poly_feature_names = poly.get_feature_names_out(numerical_cols)
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

# Manual interaction features
df['age_income_interaction'] = df['age'] * df['income']
df['age_education_interaction'] = df['age'] * df['education_Bachelor']  # jika sudah di-encode

# Multiple interactions
def create_interactions(df, cols, max_degree=2):
    for degree in range(2, max_degree + 1):
        for combo in combinations(cols, degree):
            col_name = '_x_'.join(combo)
            df[col_name] = df[combo].prod(axis=1)
    return df

df = create_interactions(df, ['age', 'income'], max_degree=2)
```

### Statistical Features

```python
# Group-based statistical features
groupby_features = df.groupby('city').agg({
    'purchase_amount': ['mean', 'std', 'min', 'max', 'count'],
    'age': ['mean', 'std'],
    'income': ['mean', 'std']
})

# Flatten column names
groupby_features.columns = ['_'.join(col).strip() for col in groupby_features.columns.values]
groupby_features = groupby_features.add_prefix('city_')

# Merge back to original dataframe
df = df.merge(groupby_features, left_on='city', right_index=True, how='left')

# Rank-based features
df['income_rank'] = df['income'].rank(pct=True)  # Percentile rank
df['age_rank_by_city'] = df.groupby('city')['age'].rank(pct=True)

# Deviation from group mean
df['income_dev_from_city_mean'] = df['income'] - df.groupby('city')['income'].transform('mean')
```

### Distance and Similarity Features

```python
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# Distance from centroid/mean
def calculate_distance_from_center(df, features):
    center = df[features].mean()
    distances = []
    for _, row in df[features].iterrows():
        dist = euclidean(row.values, center.values)
        distances.append(dist)
    return distances

numerical_features = ['age', 'income', 'purchase_amount']
df['distance_from_center'] = calculate_distance_from_center(df, numerical_features)

# Similarity with reference points
reference_customer = df[numerical_features].iloc[0]  # Customer pertama sebagai reference
similarities = []
for _, row in df[numerical_features].iterrows():
    sim = cosine_similarity([reference_customer.values], [row.values])[0][0]
    similarities.append(sim)

df['similarity_to_ref'] = similarities
```

## Domain-Specific Feature Engineering

---

### E-commerce Features

```python
# Customer behavior features
def create_ecommerce_features(df):
    # Recency, Frequency, Monetary (RFM)
    current_date = df['purchase_date'].max()
    
    customer_features = df.groupby('customer_id').agg({
        'purchase_date': lambda x: (current_date - x.max()).days,  # Recency
        'purchase_amount': ['count', 'sum', 'mean', 'std'],       # Frequency, Monetary
        'product_category': 'nunique'                             # Diversity
    })
    
    customer_features.columns = ['recency', 'frequency', 'monetary_total', 
                               'monetary_avg', 'monetary_std', 'category_diversity']
    
    # Customer lifetime value estimate
    customer_features['estimated_clv'] = (
        customer_features['monetary_avg'] * 
        customer_features['frequency'] * 
        (365 / customer_features['recency'].clip(lower=1))
    )
    
    return customer_features

# Time-based patterns
def create_time_patterns(df):
    # Shopping patterns
    df['is_holiday_season'] = df['month'].isin([11, 12])  # Nov-Dec
    df['is_payday'] = df['day'].isin([1, 15])  # Asumssi payday tanggal 1 dan 15
    
    # Purchase velocity
    df = df.sort_values(['customer_id', 'purchase_date'])
    df['days_since_last_purchase'] = df.groupby('customer_id')['purchase_date'].diff().dt.days
    
    return df
```

### Financial Features

```python
def create_financial_features(df):
    # Risk indicators
    df['debt_to_income'] = df['debt'] / df['income']
    df['savings_rate'] = df['savings'] / df['income']
    
    # Credit utilization
    df['credit_utilization'] = df['credit_used'] / df['credit_limit']
    
    # Financial stability indicators
    df['income_volatility'] = df.groupby('customer_id')['monthly_income'].rolling(12).std()
    df['expense_volatility'] = df.groupby('customer_id')['monthly_expense'].rolling(12).std()
    
    # Trend indicators
    df['income_trend'] = df.groupby('customer_id')['monthly_income'].pct_change(12)  # YoY change
    
    return df
```

## Feature Selection and Evaluation

---

### Automatic Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Assume we have a target variable
target = 'high_value_customer'  # Example binary target

# Statistical feature selection
selector_stats = SelectKBest(score_func=f_classif, k=10)
selected_features_stats = selector_stats.fit_transform(df.select_dtypes(include=[np.number]), 
                                                      df[target])

# Mutual information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
selected_features_mi = selector_mi.fit_transform(df.select_dtypes(include=[np.number]), 
                                                df[target])

# Recursive Feature Elimination
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector_rfe = RFE(estimator=rf, n_features_to_select=10)
selected_features_rfe = selector_rfe.fit_transform(df.select_dtypes(include=[np.number]), 
                                                  df[target])

# Feature importance dari tree-based models
rf.fit(df.select_dtypes(include=[np.number]), df[target])
feature_importance = pd.DataFrame({
    'feature': df.select_dtypes(include=[np.number]).columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))
```

### Feature Quality Assessment

```python
def assess_feature_quality(df, feature_col, target_col=None):
    """
    Assess the quality of a feature
    """
    assessment = {}
    
    # Basic statistics
    assessment['missing_rate'] = df[feature_col].isnull().mean()
    assessment['unique_values'] = df[feature_col].nunique()
    assessment['unique_rate'] = assessment['unique_values'] / len(df)
    
    # Variance
    if df[feature_col].dtype in ['int64', 'float64']:
        assessment['variance'] = df[feature_col].var()
        assessment['coefficient_of_variation'] = df[feature_col].std() / df[feature_col].mean()
    
    # Target correlation (if target provided)
    if target_col and df[target_col].dtype in ['int64', 'float64']:
        if df[feature_col].dtype in ['int64', 'float64']:
            assessment['correlation_with_target'] = df[feature_col].corr(df[target_col])
        
    # Distribution assessment
    if df[feature_col].dtype in ['int64', 'float64']:
        from scipy import stats
        _, p_value = stats.normaltest(df[feature_col].dropna())
        assessment['is_normal_distribution'] = p_value > 0.05
    
    return assessment

# Example usage
feature_quality = assess_feature_quality(df, 'income', 'purchase_amount')
print(feature_quality)
```

## Feature Engineering Pipeline

---

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer untuk feature engineering
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Create interaction features
        if 'age' in X.columns and 'income' in X.columns:
            X['age_income_interaction'] = X['age'] * X['income']
        
        # Create polynomial features
        if 'income' in X.columns:
            X['income_squared'] = X['income'] ** 2
            X['income_log'] = np.log1p(X['income'])
        
        # Create categorical aggregations
        if 'city' in X.columns and 'purchase_amount' in X.columns:
            city_means = X.groupby('city')['purchase_amount'].transform('mean')
            X['city_purchase_mean'] = city_means
        
        return X

# Create preprocessing pipeline
numeric_features = ['age', 'income', 'purchase_amount']
categorical_features = ['education', 'city']

numeric_transformer = Pipeline(steps=[
    ('engineer', CustomFeatureEngineer()),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Use in machine learning pipeline
from sklearn.linear_model import LogisticRegression

ml_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

## Best Practices & Tips

---

### 1. Feature Engineering Workflow

```python
def feature_engineering_workflow(df):
    """
    Systematic approach to feature engineering
    """
    # Step 1: Understand the data
    print("Data shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Step 2: Domain-specific features
    # (implement based on your domain)
    
    # Step 3: Statistical features
    # (calculate aggregations, ratios, etc.)
    
    # Step 4: Interaction features
    # (create meaningful combinations)
    
    # Step 5: Feature selection
    # (remove redundant or irrelevant features)
    
    return df
```

### 2. Validation Strategy

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def validate_feature_engineering(original_features, engineered_features, target, model):
    """
    Compare performance before and after feature engineering
    """
    # Original performance
    scores_original = cross_val_score(model, original_features, target, 
                                    cv=5, scoring='neg_mean_squared_error')
    
    # Engineered performance  
    scores_engineered = cross_val_score(model, engineered_features, target, 
                                      cv=5, scoring='neg_mean_squared_error')
    
    print(f"Original features - Mean CV Score: {-scores_original.mean():.4f} (+/- {scores_original.std() * 2:.4f})")
    print(f"Engineered features - Mean CV Score: {-scores_engineered.mean():.4f} (+/- {scores_engineered.std() * 2:.4f})")
    
    improvement = (-scores_engineered.mean()) - (-scores_original.mean())
    print(f"Improvement: {improvement:.4f}")
    
    return improvement
```

### 3. Important Reminders

- **Less is More**: Jangan buat terlalu banyak fitur yang tidak berguna
- **Domain Knowledge**: Gunakan pemahaman bisnis untuk membuat fitur yang meaningful
- **Validation**: Selalu validasi apakah fitur baru benar-benar meningkatkan performa
- **Interpretability**: Pertimbangkan apakah fitur masih bisa diinterpretasi
- **Leakage**: Hati-hati jangan sampai ada data leakage dari masa depan