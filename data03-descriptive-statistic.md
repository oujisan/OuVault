# [ai] #03 - ML Descriptive Statistics

![data](https://raw.githubusercontent.com/oujisan/OuVault/main/img/data.png)

## Descriptive Statistics: Mean, Median, Mode, and Standard Deviation

---

Statistik deskriptif adalah fondasi untuk memahami data kita. Sebelum membangun model machine learning, kita perlu tahu "cerita" yang diceritakan oleh data. Mari kita pelajari measures of central tendency dan variability yang akan menjadi kompas kita dalam eksplorasi data.

## Understanding Central Tendency

---

Central tendency mengukur nilai "tengah" atau "tipikal" dari sebuah dataset. Tiga ukuran utama adalah mean, median, dan mode.

### Mean (Rata-rata)

Mean adalah jumlah semua nilai dibagi dengan jumlah observasi.

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Sample data: skor ujian mahasiswa
exam_scores = [85, 92, 78, 96, 73, 88, 91, 82, 95, 87, 79, 93, 86, 90, 84]

# Calculating mean
mean_score = np.mean(exam_scores)
print(f"Mean exam score: {mean_score:.2f}")

# Using pandas
df_scores = pd.DataFrame({'scores': exam_scores})
print(f"Mean using pandas: {df_scores['scores'].mean():.2f}")

# Manual calculation to understand the concept
manual_mean = sum(exam_scores) / len(exam_scores)
print(f"Manual calculation: {manual_mean:.2f}")
```

#### Karakteristik Mean:

- **Sensitif terhadap outliers**: Nilai ekstrem dapat menggeser mean secara signifikan
- **Menggunakan semua data points**: Setiap nilai mempengaruhi hasil
- **Paling umum digunakan**: Familiar dan mudah dipahami

```python
# Demonstrasi sensitivitas mean terhadap outliers
normal_data = [80, 82, 85, 87, 88, 90, 92, 93, 95, 97]
with_outlier = normal_data + [150]  # Adding an outlier

print(f"Mean without outlier: {np.mean(normal_data):.2f}")
print(f"Mean with outlier: {np.mean(with_outlier):.2f}")
print(f"Difference: {np.mean(with_outlier) - np.mean(normal_data):.2f}")
```

### Median (Nilai Tengah)

Median adalah nilai tengah ketika data diurutkan dari terkecil ke terbesar.

```python
# Calculating median
median_score = np.median(exam_scores)
print(f"Median exam score: {median_score:.2f}")

# Manual calculation for understanding
sorted_scores = sorted(exam_scores)
n = len(sorted_scores)

if n % 2 == 1:
    # Odd number of observations
    manual_median = sorted_scores[n // 2]
else:
    # Even number of observations
    mid1 = sorted_scores[n // 2 - 1]
    mid2 = sorted_scores[n // 2]
    manual_median = (mid1 + mid2) / 2

print(f"Manual median calculation: {manual_median:.2f}")
print(f"Sorted scores: {sorted_scores}")
```

#### Karakteristik Median:

- **Robust terhadap outliers**: Tidak terpengaruh oleh nilai ekstrem
- **Posisi-based**: Hanya bergantung pada posisi, bukan nilai aktual
- **Baik untuk data skewed**: Lebih representatif untuk distribusi tidak normal

```python
# Demonstrasi robustness median terhadap outliers
print(f"Median without outlier: {np.median(normal_data):.2f}")
print(f"Median with outlier: {np.median(with_outlier):.2f}")
print(f"Difference: {np.median(with_outlier) - np.median(normal_data):.2f}")
```

### Mode (Nilai yang Paling Sering Muncul)

Mode adalah nilai yang paling sering muncul dalam dataset.

```python
# Sample data with repeated values
repeated_scores = [85, 92, 78, 85, 96, 73, 85, 88, 91, 82, 85, 95]

# Using scipy.stats for mode
mode_result = stats.mode(repeated_scores)
print(f"Mode: {mode_result.mode[0]} (appears {mode_result.count[0]} times)")

# Using pandas value_counts
df_repeated = pd.DataFrame({'scores': repeated_scores})
mode_pandas = df_repeated['scores'].mode()
print(f"Mode using pandas: {mode_pandas.iloc[0]}")

# Value counts to see frequency
print("\nFrequency distribution:")
print(df_repeated['scores'].value_counts().sort_index())
```

#### Types of Modality:

```python
# Different types of distributions
unimodal = [1, 2, 2, 2, 3, 4, 5]  # One mode
bimodal = [1, 2, 2, 2, 3, 4, 5, 5, 5, 6]  # Two modes  
no_mode = [1, 2, 3, 4, 5, 6, 7]  # No repeating values

print("Unimodal data:", unimodal)
print("Mode:", stats.mode(unimodal).mode[0])

print("\nBimodal data:", bimodal)
# For bimodal, we need to check manually
from collections import Counter
counter = Counter(bimodal)
max_count = max(counter.values())
modes = [k for k, v in counter.items() if v == max_count]
print("Modes:", modes)

print("\nNo mode data:", no_mode)
print("All values appear once - no mode")
```

## Understanding Variability and Spread

---

Measures of variability menunjukkan seberapa tersebar data di sekitar central tendency.

### Range (Rentang)

Range adalah perbedaan antara nilai maksimum dan minimum.

```python
# Calculate range
data_range = max(exam_scores) - min(exam_scores)
print(f"Range of exam scores: {data_range}")
print(f"Min: {min(exam_scores)}, Max: {max(exam_scores)}")

# Using numpy
np_range = np.ptp(exam_scores)  # Peak to peak
print(f"Range using numpy: {np_range}")
```

### Variance (Varians)

Variance mengukur rata-rata kuadrat deviasi dari mean.

```python
# Calculate variance
variance = np.var(exam_scores, ddof=1)  # Sample variance (ddof=1)
print(f"Sample variance: {variance:.2f}")

# Manual calculation for understanding
mean = np.mean(exam_scores)
squared_deviations = [(x - mean)**2 for x in exam_scores]
manual_variance = sum(squared_deviations) / (len(exam_scores) - 1)
print(f"Manual variance calculation: {manual_variance:.2f}")

# Population vs Sample variance
pop_variance = np.var(exam_scores, ddof=0)  # Population variance
sample_variance = np.var(exam_scores, ddof=1)  # Sample variance

print(f"Population variance (ddof=0): {pop_variance:.2f}")
print(f"Sample variance (ddof=1): {sample_variance:.2f}")
```

### Standard Deviation (Standar Deviasi)

Standard deviation adalah akar kuadrat dari variance, dalam unit yang sama dengan data asli.

```python
# Calculate standard deviation
std_dev = np.std(exam_scores, ddof=1)
print(f"Standard deviation: {std_dev:.2f}")

# Relationship with variance
print(f"Standard deviation = √variance: {np.sqrt(variance):.2f}")

# Interpretation
print(f"\nInterpretation:")
print(f"Mean ± 1 SD: {mean:.1f} ± {std_dev:.1f} = [{mean-std_dev:.1f}, {mean+std_dev:.1f}]")

# Count values within 1 standard deviation
within_1sd = sum(1 for x in exam_scores if abs(x - mean) <= std_dev)
percentage_1sd = (within_1sd / len(exam_scores)) * 100
print(f"Values within 1 SD: {within_1sd}/{len(exam_scores)} ({percentage_1sd:.1f}%)")
```

## Quartiles and Percentiles

---

### Understanding Quartiles

Quartiles membagi data menjadi empat bagian yang sama.

```python
# Calculate quartiles
Q1 = np.percentile(exam_scores, 25)  # First quartile
Q2 = np.percentile(exam_scores, 50)  # Second quartile (median)
Q3 = np.percentile(exam_scores, 75)  # Third quartile

print(f"First Quartile (Q1): {Q1:.2f}")
print(f"Second Quartile (Q2/Median): {Q2:.2f}")
print(f"Third Quartile (Q3): {Q3:.2f}")

# Interquartile Range (IQR)
IQR = Q3 - Q1
print(f"Interquartile Range (IQR): {IQR:.2f}")

# Using pandas describe for comprehensive stats
df_scores = pd.DataFrame({'scores': exam_scores})
print("\nComprehensive statistics:")
print(df_scores.describe())
```

### Percentiles for Detailed Analysis

```python
# Calculate various percentiles
percentiles = [10, 25, 50, 75, 90, 95, 99]
percentile_values = [np.percentile(exam_scores, p) for p in percentiles]

print("Percentile Analysis:")
for p, val in zip(percentiles, percentile_values):
    print(f"{p}th percentile: {val:.2f}")

# Interpretation
print(f"\nInterpretation:")
print(f"90% of students scored below {np.percentile(exam_scores, 90):.1f}")
print(f"Only 10% of students scored above {np.percentile(exam_scores, 90):.1f}")
```

## Detecting Outliers Using Statistical Methods

---

### IQR Method for Outlier Detection

```python
def detect_outliers_iqr(data):
    """Detect outliers using IQR method"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers
    }

# Test with data containing outliers
data_with_outliers = exam_scores + [45, 105]  # Adding outliers

outlier_analysis = detect_outliers_iqr(data_with_outliers)
print("Outlier Analysis using IQR method:")
print(f"Lower bound: {outlier_analysis['lower_bound']:.2f}")
print(f"Upper bound: {outlier_analysis['upper_bound']:.2f}")
print(f"Outliers found: {outlier_analysis['outliers']}")
```

### Z-Score Method for Outlier Detection

```python
def detect_outliers_zscore(data, threshold=2):
    """Detect outliers using Z-score method"""
    mean = np.mean(data)
    std = np.std(data)
    
    z_scores = [(x - mean) / std for x in data]
    outliers = [data[i] for i, z in enumerate(z_scores) if abs(z) > threshold]
    
    return {
        'mean': mean,
        'std': std,
        'z_scores': z_scores,
        'outliers': outliers,
        'outlier_indices': [i for i, z in enumerate(z_scores) if abs(z) > threshold]
    }

# Test Z-score method
zscore_analysis = detect_outliers_zscore(data_with_outliers)
print("\nOutlier Analysis using Z-score method:")
print(f"Mean: {zscore_analysis['mean']:.2f}")
print(f"Standard deviation: {zscore_analysis['std']:.2f}")
print(f"Outliers (|z| > 2): {zscore_analysis['outliers']}")
```

## Comparing Distributions

---

### Multiple Dataset Comparison

```python
# Simulate different class performance
class_a = [85, 87, 89, 92, 88, 86, 90, 91, 93, 87, 89, 88]
class_b = [75, 95, 70, 98, 72, 96, 74, 94, 76, 92, 78, 90]
class_c = [82, 83, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89]

classes_data = {
    'Class A': class_a,
    'Class B': class_b,
    'Class C': class_c
}

def compare_classes(classes_dict):
    """Compare statistical measures across classes"""
    comparison = pd.DataFrame()
    
    for class_name, scores in classes_dict.items():
        stats_dict = {
            'Mean': np.mean(scores),
            'Median': np.median(scores),
            'Mode': stats.mode(scores).mode[0],
            'Std Dev': np.std(scores, ddof=1),
            'Variance': np.var(scores, ddof=1),
            'Range': max(scores) - min(scores),
            'IQR': np.percentile(scores, 75) - np.percentile(scores, 25),
            'Min': min(scores),
            'Max': max(scores)
        }
        comparison[class_name] = stats_dict
    
    return comparison.round(2)

comparison_table = compare_classes(classes_data)
print("Class Performance Comparison:")
print(comparison_table)
```

### Interpreting the Differences

```python
# Analysis of the comparison
print("\n📊 Analysis:")
print("Class A: Consistent performance (low std dev)")
print("Class B: High variability (high std dev, wide range)")  
print("Class C: Very consistent (lowest std dev, tight range)")

# Calculate coefficient of variation for relative comparison
cv_comparison = {}
for class_name, scores in classes_data.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)
    cv = (std_score / mean_score) * 100
    cv_comparison[class_name] = cv

print("\nCoefficient of Variation (CV = std/mean * 100):")
for class_name, cv in cv_comparison.items():
    print(f"{class_name}: {cv:.2f}%")
```

## Skewness and Kurtosis

---

### Understanding Shape of Distribution

```python
from scipy.stats import skew, kurtosis

# Different types of distributions
normal_like = np.random.normal(85, 10, 1000)
right_skewed = np.random.exponential(2, 1000) * 10 + 70
left_skewed = 100 - np.random.exponential(2, 1000) * 5

distributions = {
    'Normal-like': normal_like,
    'Right-skewed': right_skewed,
    'Left-skewed': left_skewed
}

def analyze_shape(data_dict):
    """Analyze skewness and kurtosis of distributions"""
    shape_analysis = pd.DataFrame()
    
    for name, data in data_dict.items():
        analysis = {
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Std Dev': np.std(data),
            'Skewness': skew(data),
            'Kurtosis': kurtosis(data)
        }
        shape_analysis[name] = analysis
    
    return shape_analysis.round(3)

shape_comparison = analyze_shape(distributions)
print("Distribution Shape Analysis:")
print(shape_comparison)

# Interpretation of skewness
print("\n📈 Skewness Interpretation:")
for name, data in distributions.items():
    skewness_val = skew(data)
    if skewness_val > 0.5:
        interpretation = "Right-skewed (positive skew)"
    elif skewness_val < -0.5:
        interpretation = "Left-skewed (negative skew)"
    else:
        interpretation = "Approximately symmetric"
    
    print(f"{name}: {skewness_val:.3f} - {interpretation}")

# Interpretation of kurtosis
print("\n📊 Kurtosis Interpretation:")
for name, data in distributions.items():
    kurtosis_val = kurtosis(data)
    if kurtosis_val > 0:
        interpretation = "Leptokurtic (heavy tails, peaked)"
    elif kurtosis_val < 0:
        interpretation = "Platykurtic (light tails, flat)"
    else:
        interpretation = "Mesokurtic (normal-like)"
    
    print(f"{name}: {kurtosis_val:.3f} - {interpretation}")
```

## Working with Grouped Data

---

### Categorical Data Analysis

```python
# Sample dataset with categories
student_data = pd.DataFrame({
    'student_id': range(1, 101),
    'major': np.random.choice(['CS', 'Math', 'Physics', 'Chemistry'], 100),
    'year': np.random.choice([1, 2, 3, 4], 100),
    'gpa': np.random.normal(3.2, 0.5, 100).clip(0, 4),
    'gender': np.random.choice(['Male', 'Female'], 100)
})

# Ensure GPA is within realistic bounds
student_data['gpa'] = student_data['gpa'].clip(0, 4.0)

print("Sample of student data:")
print(student_data.head())

# Grouped statistics by major
grouped_stats = student_data.groupby('major')['gpa'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(3)

print("\nGPA Statistics by Major:")
print(grouped_stats)

# Multiple grouping
detailed_stats = student_data.groupby(['major', 'year'])['gpa'].agg([
    'mean', 'std', 'count'
]).round(3)

print("\nDetailed GPA Statistics by Major and Year:")
print(detailed_stats)
```

### Cross-tabulation Analysis

```python
# Cross-tabulation for categorical variables
cross_tab = pd.crosstab(student_data['major'], student_data['year'], 
                       margins=True, margins_name="Total")

print("Cross-tabulation: Major vs Year")
print(cross_tab)

# Percentage cross-tabulation
cross_tab_pct = pd.crosstab(student_data['major'], student_data['year'], 
                           normalize='index') * 100

print("\nPercentage distribution by Major:")
print(cross_tab_pct.round(1))
```

## Time Series Descriptive Statistics

---

### Temporal Data Analysis

```python
# Generate time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
np.random.seed(42)

# Simulate sales data with trend and seasonality
trend = np.linspace(1000, 1200, 365)
seasonality = 100 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly pattern
noise = np.random.normal(0, 50, 365)
sales = trend + seasonality + noise

time_series_data = pd.DataFrame({
    'date': dates,
    'sales': sales
})

time_series_data['month'] = time_series_data['date'].dt.month
time_series_data['quarter'] = time_series_data['date'].dt.quarter
time_series_data['day_of_week'] = time_series_data['date'].dt.dayofweek

print("Time series data sample:")
print(time_series_data.head())

# Monthly statistics
monthly_stats = time_series_data.groupby('month')['sales'].agg([
    'mean', 'std', 'min', 'max', 'count'
]).round(2)

print("\nMonthly Sales Statistics:")
print(monthly_stats)

# Rolling statistics (moving averages)
time_series_data['rolling_mean_7'] = time_series_data['sales'].rolling(window=7).mean()
time_series_data['rolling_std_7'] = time_series_data['sales'].rolling(window=7).std()

print("\nRolling statistics (7-day window) - last 5 days:")
print(time_series_data[['date', 'sales', 'rolling_mean_7', 'rolling_std_7']].tail())
```

## Correlation Analysis

---

### Understanding Relationships Between Variables

```python
# Generate correlated data
np.random.seed(42)
n_samples = 200

# Create correlated variables
study_hours = np.random.normal(5, 2, n_samples).clip(0, 12)
sleep_hours = 8 + np.random.normal(0, 1, n_samples) - 0.3 * (study_hours - 5)
stress_level = 3 + 0.4 * study_hours - 0.2 * sleep_hours + np.random.normal(0, 0.5, n_samples)
gpa = 2.5 + 0.2 * study_hours + 0.1 * sleep_hours - 0.15 * stress_level + np.random.normal(0, 0.3, n_samples)

# Create DataFrame
correlation_data = pd.DataFrame({
    'study_hours': study_hours.clip(0, 12),
    'sleep_hours': sleep_hours.clip(4, 12),
    'stress_level': stress_level.clip(1, 5),
    'gpa': gpa.clip(0, 4)
})

print("Correlation analysis data sample:")
print(correlation_data.head())

# Calculate correlation matrix
correlation_matrix = correlation_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# Interpretation of correlation strength
def interpret_correlation(corr_value):
    """Interpret correlation strength"""
    abs_corr = abs(corr_value)
    if abs_corr >= 0.8:
        strength = "Very Strong"
    elif abs_corr >= 0.6:
        strength = "Strong"
    elif abs_corr >= 0.4:
        strength = "Moderate"
    elif abs_corr >= 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    direction = "Positive" if corr_value > 0 else "Negative"
    return f"{strength} {direction}"

print("\nCorrelation Interpretations:")
variables = correlation_data.columns
for i in range(len(variables)):
    for j in range(i+1, len(variables)):
        var1, var2 = variables[i], variables[j]
        corr_val = correlation_matrix.loc[var1, var2]
        interpretation = interpret_correlation(corr_val)
        print(f"{var1} vs {var2}: {corr_val:.3f} ({interpretation})")
```

## Practical Data Profiling

---

### Comprehensive Data Profiling Function

```python
def comprehensive_data_profile(df, target_col=None):
    """
    Generate comprehensive data profile report
    """
    print("🔍 COMPREHENSIVE DATA PROFILE REPORT")
    print("=" * 60)
    
    # Basic information
    print(f"📊 Dataset Overview:")
    print(f"   • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"   • Duplicate rows: {df.duplicated().sum():,}")
    
    # Column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    
    print(f"\n📋 Column Types:")
    print(f"   • Numeric: {len(numeric_cols)} columns")
    print(f"   • Categorical: {len(categorical_cols)} columns") 
    print(f"   • Datetime: {len(datetime_cols)} columns")
    
    # Missing values analysis
    missing_analysis = df.isnull().sum()
    missing_pct = (missing_analysis / len(df)) * 100
    
    if missing_analysis.sum() > 0:
        print(f"\n❌ Missing Values:")
        for col in missing_analysis[missing_analysis > 0].index:
            print(f"   • {col}: {missing_analysis[col]:,} ({missing_pct[col]:.1f}%)")
    else:
        print(f"\n✅ No missing values found")
    
    # Numeric columns analysis
    if len(numeric_cols) > 0:
        print(f"\n📈 Numeric Columns Analysis:")
        numeric_stats = df[numeric_cols].describe().round(3)
        
        for col in numeric_cols:
            print(f"\n   {col}:")
            print(f"      Mean: {numeric_stats.loc['mean', col]:.3f}")
            print(f"      Median: {numeric_stats.loc['50%', col]:.3f}")
            print(f"      Std Dev: {numeric_stats.loc['std', col]:.3f}")
            
            # Skewness and kurtosis
            col_skew = skew(df[col].dropna())
            col_kurt = kurtosis(df[col].dropna())
            print(f"      Skewness: {col_skew:.3f}")
            print(f"      Kurtosis: {col_kurt:.3f}")
            
            # Outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            print(f"      Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    # Categorical columns analysis
    if len(categorical_cols) > 0:
        print(f"\n📊 Categorical Columns Analysis:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            most_common = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
            most_common_count = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            
            print(f"\n   {col}:")
            print(f"      Unique values: {unique_count}")
            print(f"      Most common: '{most_common}' ({most_common_count} times)")
            
            if unique_count <= 10:
                print(f"      Value counts:")
                value_counts = df[col].value_counts()
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    print(f"         '{val}': {count} ({pct:.1f}%)")
    
    # Correlation analysis for numeric columns
    if len(numeric_cols) > 1:
        print(f"\n🔗 Correlation Analysis:")
        corr_matrix = df[numeric_cols].corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Threshold for "high" correlation
                    high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
        
        if high_corr_pairs:
            print("   High correlations (|r| > 0.5):")
            for var1, var2, corr_val in high_corr_pairs:
                print(f"      {var1} ↔ {var2}: {corr_val:.3f}")
        else:
            print("   No high correlations found (|r| > 0.5)")
    
    return {
        'numeric_columns': numeric_cols.tolist(),
        'categorical_columns': categorical_cols.tolist(),
        'missing_values': missing_analysis.to_dict(),
        'basic_stats': df.describe().to_dict() if len(numeric_cols) > 0 else {}
    }

# Test the profiling function with our student data
profile_results = comprehensive_data_profile(student_data)
```

## Real-world Application: Customer Analysis

---

### E-commerce Customer Segmentation Analysis

```python
# Generate realistic e-commerce customer data
np.random.seed(42)
n_customers = 1000

# Customer demographics and behavior
customer_data = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(35, 12, n_customers).clip(18, 80).astype(int),
    'annual_income': np.random.lognormal(10.5, 0.5, n_customers).clip(20000, 200000).astype(int),
    'total_purchases': np.random.poisson(8, n_customers),
    'avg_order_value': np.random.gamma(2, 25, n_customers).clip(10, 500),
    'months_active': np.random.randint(1, 25, n_customers),
    'customer_segment': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                       n_customers, p=[0.4, 0.3, 0.2, 0.1])
})

# Calculate derived metrics
customer_data['total_spent'] = customer_data['total_purchases'] * customer_data['avg_order_value']
customer_data['monthly_spend'] = customer_data['total_spent'] / customer_data['months_active']

print("E-commerce Customer Data Analysis")
print("=" * 50)

# Segment-wise analysis
segment_analysis = customer_data.groupby('customer_segment').agg({
    'age': ['mean', 'std'],
    'annual_income': ['mean', 'median', 'std'],
    'total_purchases': ['mean', 'std'],
    'avg_order_value': ['mean', 'std'],
    'total_spent': ['mean', 'median', 'std'],
    'monthly_spend': ['mean', 'std']
}).round(2)

print("Customer Segment Analysis:")
print(segment_analysis)

# Key insights
print("\n💡 Key Insights:")
for segment in customer_data['customer_segment'].unique():
    seg_data = customer_data[customer_data['customer_segment'] == segment]
    avg_income = seg_data['annual_income'].mean()
    avg_spend = seg_data['total_spent'].mean()
    spend_ratio = (avg_spend / avg_income) * 100
    
    print(f"{segment} customers:")
    print(f"   • Average income: ${avg_income:,.0f}")
    print(f"   • Average total spent: ${avg_spend:,.0f}")
    print(f"   • Spend ratio: {spend_ratio:.2f}% of income")
    print()
```

## Summary and Best Practices

---

### 🎯 Key Takeaways:

```python
print("📚 DESCRIPTIVE STATISTICS SUMMARY")
print("=" * 50)

print("✅ Central Tendency Measures:")
print("   • Mean: Best for normal distributions, sensitive to outliers")
print("   • Median: Robust to outliers, better for skewed data")
print("   • Mode: Most frequent value, useful for categorical data")

print("\n✅ Variability Measures:")
print("   • Range: Simple but sensitive to outliers")
print("   • Standard Deviation: Most common, same units as data")
print("   • IQR: Robust to outliers, good for boxplots")

print("\n✅ Distribution Shape:")
print("   • Skewness: Measures asymmetry of distribution")
print("   • Kurtosis: Measures tail heaviness")

print("\n✅ Best Practices:")
print("   • Always visualize your data alongside statistics")
print("   • Consider outliers and their impact on measures")
print("   • Use appropriate measures for your data type")
print("   • Context matters - interpret statistics meaningfully")
print("   • Check assumptions before applying statistical methods")
```

### 🔧 Practical Tips:

```python
# Quick statistical summary function
def quick_stats(data, column_name="Data"):
    """Generate quick statistical summary"""
    stats = {
        'Count': len(data),
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std Dev': np.std(data, ddof=1),
        'Min': np.min(data),
        'Max': np.max(data),
        'Q1': np.percentile(data, 25),
        'Q3': np.percentile(data, 75),
        'Skewness': skew(data),
        'Outliers (IQR)': len([x for x in data if x < np.percentile(data, 25) - 1.5*(np.percentile(data, 75)-np.percentile(data, 25)) or x > np.percentile(data, 75) + 1.5*(np.percentile(data, 75)-np.percentile(data, 25))])
    }
    
    print(f"📊 Quick Stats for {column_name}:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    return stats

# Example usage
sample_data = [85, 92, 78, 96, 73, 88, 91, 82, 95, 87, 79, 93, 86, 90, 84]
quick_stats(sample_data, "Exam Scores")
```

### 🔜 Next Steps:

Sekarang setelah menguasai statistik deskriptif, kita memiliki tools yang solid untuk memahami karakteristik data. Di modul selanjutnya, kita akan belajar bagaimana memvisualisasikan insights ini menggunakan histogram, boxplot, dan scatter plot - karena "a picture is worth a thousand statistics!"

**Key Skills yang telah dikuasai:**

- ✅ Menghitung dan menginterpretasi measures of central tendency
- ✅ Memahami variability dan spread dalam data
- ✅ Mendeteksi outliers menggunakan metode statistik
- ✅ Menganalisis shape dan karakteristik distribusi
- ✅ Melakukan analisis korelasi antar variabel
- ✅ Membuat comprehensive data profiling

**Siap untuk visualisasi data!** 📊