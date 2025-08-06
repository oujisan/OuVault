# [data] #08 - Data Correlation & Covariance

![data](https://raw.githubusercontent.com/oujisan/OuVault/main/img/data.png)

## Introduction to Correlation & Covariance

---

Korelasi dan kovarians adalah konsep fundamental dalam analisis data yang membantu kita memahami hubungan antar variabel. Bayangkan seperti mendeteksi "chemistry" antara dua orang - apakah mereka cocok, bertentangan, atau tidak ada hubungan sama sekali.

Dalam machine learning, memahami hubungan ini sangat penting untuk:

- Memilih fitur yang relevan
- Mendeteksi multikolinearitas
- Memahami struktur data
- Membuat keputusan preprocessing
- Interpretasi model

Mari kita eksplorasi lebih dalam!

## Understanding Covariance

---

Kovarians mengukur bagaimana dua variabel berubah bersama-sama. Jika kedua variabel cenderung naik dan turun bersamaan, kovariansnya positif. Jika satu naik saat yang lain turun, kovariansnya negatif.

### Mathematical Foundation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create sample data
np.random.seed(42)
n_samples = 1000

# Generate correlated data
mean = [50, 100]
cov = [[100, 80],    # Positive covariance
       [80, 200]]

data_pos = np.random.multivariate_normal(mean, cov, n_samples)
df_pos = pd.DataFrame(data_pos, columns=['X_positive', 'Y_positive'])

# Generate negatively correlated data
cov_neg = [[100, -80],   # Negative covariance  
           [-80, 200]]

data_neg = np.random.multivariate_normal(mean, cov_neg, n_samples)
df_neg = pd.DataFrame(data_neg, columns=['X_negative', 'Y_negative'])

# Generate independent data
data_indep = np.random.multivariate_normal(mean, [[100, 0], [0, 200]], n_samples)
df_indep = pd.DataFrame(data_indep, columns=['X_independent', 'Y_independent'])

# Combine all data
df = pd.concat([df_pos, df_neg, df_indep], axis=1)

print("Sample data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
```

### Computing Covariance

```python
# Manual calculation of covariance
def calculate_covariance(x, y):
    """
    Manual calculation of covariance
    """
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    covariance = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)  # Sample covariance
    return covariance

# Using built-in functions
x = df['X_positive']
y = df['Y_positive']

manual_cov = calculate_covariance(x, y)
numpy_cov = np.cov(x, y)[0, 1]  # Off-diagonal element
pandas_cov = df[['X_positive', 'Y_positive']].cov().iloc[0, 1]

print(f"Manual calculation: {manual_cov:.2f}")
print(f"NumPy calculation: {numpy_cov:.2f}")  
print(f"Pandas calculation: {pandas_cov:.2f}")

# Covariance matrix for all variables
print("\nCovariance Matrix:")
cov_matrix = df.cov()
print(cov_matrix.round(2))
```

### Interpreting Covariance

```python
# Visualize different covariance relationships
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Positive covariance
axes[0].scatter(df['X_positive'], df['Y_positive'], alpha=0.6)
axes[0].set_title(f'Positive Covariance\nCov = {df[["X_positive", "Y_positive"]].cov().iloc[0,1]:.2f}')
axes[0].set_xlabel('X_positive')
axes[0].set_ylabel('Y_positive')

# Negative covariance
axes[1].scatter(df['X_negative'], df['Y_negative'], alpha=0.6, color='red')
axes[1].set_title(f'Negative Covariance\nCov = {df[["X_negative", "Y_negative"]].cov().iloc[0,1]:.2f}')
axes[1].set_xlabel('X_negative')
axes[1].set_ylabel('Y_negative')

# No covariance
axes[2].scatter(df['X_independent'], df['Y_independent'], alpha=0.6, color='green')
axes[2].set_title(f'No Covariance\nCov = {df[["X_independent", "Y_independent"]].cov().iloc[0,1]:.2f}')
axes[2].set_xlabel('X_independent')
axes[2].set_ylabel('Y_independent')

plt.tight_layout()
plt.show()
```

## Understanding Correlation

---

Korelasi adalah versi "normalized" dari kovarians yang selalu berada di rentang -1 hingga +1. Ini membuat korelasi lebih mudah diinterpretasi dan dibandingkan.

### Pearson Correlation

```python
# Calculate Pearson correlation
def calculate_pearson_correlation(x, y):
    """
    Manual calculation of Pearson correlation coefficient
    """
    n = len(x)
    
    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate numerator and denominators
    numerator = np.sum((x - mean_x) * (y - mean_y))
    sum_sq_x = np.sum((x - mean_x) ** 2)
    sum_sq_y = np.sum((y - mean_y) ** 2)
    
    denominator = np.sqrt(sum_sq_x * sum_sq_y)
    
    correlation = numerator / denominator
    return correlation

# Compare different methods
x = df['X_positive']
y = df['Y_positive']

manual_corr = calculate_pearson_correlation(x, y)
numpy_corr = np.corrcoef(x, y)[0, 1]
pandas_corr = df[['X_positive', 'Y_positive']].corr().iloc[0, 1]
scipy_corr, p_value = stats.pearsonr(x, y)

print("Pearson Correlation Calculations:")
print(f"Manual: {manual_corr:.4f}")
print(f"NumPy: {numpy_corr:.4f}")
print(f"Pandas: {pandas_corr:.4f}")
print(f"SciPy: {scipy_corr:.4f} (p-value: {p_value:.4f})")

# Full correlation matrix
print("\nCorrelation Matrix:")
corr_matrix = df.corr()
print(corr_matrix.round(3))
```

### Other Types of Correlation

```python
from scipy.stats import spearmanr, kendalltau

# Create non-linear relationship for demonstration
x_nonlinear = np.linspace(0, 10, 1000)
y_nonlinear = x_nonlinear ** 2 + np.random.normal(0, 5, 1000)

# Calculate different correlation types
pearson_corr, pearson_p = stats.pearsonr(x_nonlinear, y_nonlinear)
spearman_corr, spearman_p = spearmanr(x_nonlinear, y_nonlinear)
kendall_corr, kendall_p = kendalltau(x_nonlinear, y_nonlinear)

print("Non-linear Relationship Correlations:")
print(f"Pearson: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
print(f"Spearman: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
print(f"Kendall: {kendall_corr:.4f} (p-value: {kendall_p:.4f})")

# Visualize the difference
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x_nonlinear, y_nonlinear, alpha=0.6)
axes[0].set_title(f'Non-linear Relationship\nPearson: {pearson_corr:.3f}')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

# Show rank correlation is better for non-linear
axes[1].scatter(stats.rankdata(x_nonlinear), stats.rankdata(y_nonlinear), alpha=0.6, color='orange')
axes[1].set_title(f'Rank-based View\nSpearman: {spearman_corr:.3f}')
axes[1].set_xlabel('Rank of X')
axes[1].set_ylabel('Rank of Y')

plt.tight_layout()
plt.show()
```

## Advanced Correlation Analysis

---

### Partial Correlation

```python
from scipy.stats import linregress

def partial_correlation(df, x, y, control_vars):
    """
    Calculate partial correlation between x and y, controlling for other variables
    """
    # Regress x on control variables
    X_control = df[control_vars].values
    y_x = df[x].values
    
    # Simple linear regression if only one control variable
    if len(control_vars) == 1:
        slope, intercept, _, _, _ = linregress(df[control_vars[0]], y_x)
        x_residuals = y_x - (slope * df[control_vars[0]] + intercept)
        
        slope, intercept, _, _, _ = linregress(df[control_vars[0]], df[y])
        y_residuals = df[y] - (slope * df[control_vars[0]] + intercept)
    else:
        # Multiple regression using numpy
        X_with_intercept = np.column_stack([np.ones(len(X_control)), X_control])
        
        # For x
        beta_x = np.linalg.lstsq(X_with_intercept, y_x, rcond=None)[0]
        x_predicted = X_with_intercept @ beta_x
        x_residuals = y_x - x_predicted
        
        # For y
        beta_y = np.linalg.lstsq(X_with_intercept, df[y], rcond=None)[0]
        y_predicted = X_with_intercept @ beta_y
        y_residuals = df[y] - y_predicted
    
    # Correlation of residuals
    partial_corr = np.corrcoef(x_residuals, y_residuals)[0, 1]
    return partial_corr

# Example: partial correlation
# Add a confounding variable
df['Z'] = df['X_positive'] * 0.5 + df['Y_positive'] * 0.3 + np.random.normal(0, 10, len(df))

# Calculate correlations
simple_corr = df[['X_positive', 'Y_positive']].corr().iloc[0, 1]
partial_corr = partial_correlation(df, 'X_positive', 'Y_positive', ['Z'])

print(f"Simple correlation between X and Y: {simple_corr:.4f}")
print(f"Partial correlation (controlling for Z): {partial_corr:.4f}")
```

### Time-Lagged Correlation

```python
def lagged_correlation(series1, series2, max_lag=10):
    """
    Calculate correlation at different lags
    """
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            # series2 leads series1
            corr = np.corrcoef(series1[:lag], series2[-lag:])[0, 1]
        elif lag > 0:
            # series1 leads series2
            corr = np.corrcoef(series1[lag:], series2[:-lag])[0, 1]
        else:
            # No lag
            corr = np.corrcoef(series1, series2)[0, 1]
        
        correlations.append(corr)
    
    return lags, correlations

# Create time series with lag
np.random.seed(42)
t = np.arange(100)
series1 = np.sin(t * 0.1) + np.random.normal(0, 0.1, 100)
series2 = np.sin((t - 5) * 0.1) + np.random.normal(0, 0.1, 100)  # Lagged version

lags, correlations = lagged_correlation(series1, series2, max_lag=10)

# Plot lag correlation
plt.figure(figsize=(10, 6))
plt.plot(lags, correlations, marker='o')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title('Cross-Correlation Function')
plt.grid(True, alpha=0.3)
plt.show()

max_corr_idx = np.argmax(np.abs(correlations))
optimal_lag = lags[max_corr_idx]
max_correlation = correlations[max_corr_idx]
print(f"Maximum correlation: {max_correlation:.4f} at lag: {optimal_lag}")
```

## Detecting Patterns and Relationships

---

### Correlation Heatmap

```python
# Create a more comprehensive dataset for demonstration
np.random.seed(42)
n = 1000

# Generate realistic dataset
data = {
    'age': np.random.randint(18, 80, n),
    'income': np.random.gamma(2, 25000, n),
    'education_years': np.random.randint(12, 20, n),
    'experience_years': np.random.randint(0, 40, n),
    'satisfaction_score': np.random.randint(1, 11, n),
}

# Add some realistic relationships
data['income'] = data['income'] + data['education_years'] * 2000 + np.random.normal(0, 5000, n)
data['experience_years'] = np.maximum(0, data['age'] - data['education_years'] - 5 + np.random.randint(-5, 5, n))
data['satisfaction_score'] = np.clip(
    5 + (data['income'] / 10000) + np.random.normal(0, 2, n), 1, 10
).astype(int)

df_real = pd.DataFrame(data)

# Calculate correlation matrix
corr_matrix = df_real.corr()

# Create comprehensive heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()
```

### Finding Strong Correlations

```python
def find_strong_correlations(df, threshold=0.5):
    """
    Find variable pairs with correlation above threshold
    """
    corr_matrix = df.corr().abs()  # Use absolute values
    
    # Create mask to avoid duplicates and self-correlation
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Find correlations above threshold
    strong_correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                strong_correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': df.corr().iloc[i, j],
                    'abs_correlation': corr_matrix.iloc[i, j]
                })
    
    return pd.DataFrame(strong_correlations).sort_values('abs_correlation', ascending=False)

strong_corrs = find_strong_correlations(df_real, threshold=0.3)
print("Strong Correlations (|r| >= 0.3):")
print(strong_corrs)
```

### Multivariate Correlation Analysis

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def correlation_circle(df, n_components=2):
    """
    Create correlation circle using PCA
    """
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(df_scaled)
    
    # Get the loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Create correlation circle
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Draw the unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    ax.add_patch(circle)
    
    # Plot the variables
    feature_names = df.select_dtypes(include=[np.number]).columns
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                head_width=0.03, head_length=0.03, fc='red', ec='red')
        ax.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, 
                ha='center', va='center')
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('Correlation Circle')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return loadings, pca.explained_variance_ratio_

loadings, variance_ratio = correlation_circle(df_real)
print(f"Explained variance: PC1={variance_ratio[0]:.3f}, PC2={variance_ratio[1]:.3f}")
```

## Statistical Significance Testing

---

### Correlation Significance Tests

```python
def correlation_significance_test(x, y, alpha=0.05):
    """
    Test if correlation is significantly different from zero
    """
    n = len(x)
    
    # Pearson correlation and p-value
    r, p_value = stats.pearsonr(x, y)
    
    # Manual calculation of t-statistic
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    df = n - 2
    p_value_manual = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Confidence interval for correlation
    fisher_z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    z_lower = fisher_z - z_critical * se
    z_upper = fisher_z + z_critical * se
    
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    results = {
        'correlation': r,
        'p_value': p_value,
        't_statistic': t_stat,
        'degrees_of_freedom': df,
        'significant': p_value < alpha,
        'confidence_interval': (r_lower, r_upper)
    }
    
    return results

# Test significance for our variables
x = df_real['age']
y = df_real['income']

results = correlation_significance_test(x, y)
print("Correlation Significance Test Results:")
print(f"Correlation: {results['correlation']:.4f}")
print(f"P-value: {results['p_value']:.4f}")
print(f"Significant at α=0.05: {results['significant']}")
print(f"95% Confidence Interval: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
```

### Multiple Correlation Testing

```python
from statsmodels.stats.multitest import multipletests

def multiple_correlation_tests(df, alpha=0.05):
    """
    Perform multiple correlation tests with correction for multiple testing
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    n_vars = len(numerical_cols)
    
    correlations = []
    p_values = []
    var_pairs = []
    
    # Calculate all pairwise correlations
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            var1, var2 = numerical_cols[i], numerical_cols[j]
            r, p = stats.pearsonr(df[var1], df[var2])
            
            correlations.append(r)
            p_values.append(p)
            var_pairs.append((var1, var2))
    
    # Apply multiple testing correction
    rejected, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'var1': [pair[0] for pair in var_pairs],
        'var2': [pair[1] for pair in var_pairs], 
        'correlation': correlations,
        'p_value': p_values,
        'p_adjusted': p_adjusted,
        'significant_raw': np.array(p_values) < alpha,
        'significant_adjusted': rejected
    })
    
    return results_df.sort_values('p_adjusted')

# Perform multiple testing
multiple_test_results = multiple_correlation_tests(df_real)
print("Multiple Correlation Tests with Bonferroni Correction:")
print(multiple_test_results)
```

## Practical Applications in Machine Learning

---

### Feature Selection Based on Correlation

```python
def correlation_feature_selection(df, target_col, method='pearson', threshold=0.1):
    """
    Select features based on correlation with target variable
    """
    if method == 'pearson':
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    elif method == 'spearman':
        correlations = df.corr(method='spearman')[target_col].abs().sort_values(ascending=False)
    
    # Remove self-correlation
    correlations = correlations.drop(target_col)
    
    # Filter by threshold
    selected_features = correlations[correlations >= threshold]
    
    return selected_features

# Example usage
target = 'satisfaction_score'
selected_features = correlation_feature_selection(df_real, target, threshold=0.2)
print("Selected Features (correlation >= 0.2):")
print(selected_features)
```

### Multicollinearity Detection

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def detect_multicollinearity(df, threshold=5.0):
    """
    Detect multicollinearity using VIF (Variance Inflation Factor)
    """
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numerical_df.columns
    vif_data["VIF"] = [variance_inflation_factor(numerical_df.values, i) 
                       for i in range(numerical_df.shape[1])]
    
    # Sort by VIF
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    # Flag problematic features
    vif_data['Problematic'] = vif_data['VIF'] > threshold
    
    return vif_data

# Check for multicollinearity
vif_results = detect_multicollinearity(df_real)
print("Variance Inflation Factor (VIF) Analysis:")
print(vif_results)

# Visualize correlation matrix for multicollinearity
plt.figure(figsize=(8, 6))
corr_matrix = df_real.corr()
sns.heatmap(corr_matrix.abs(), annot=True, cmap='Reds', 
            square=True, fmt='.2f')
plt.title('Absolute Correlation Matrix (Multicollinearity Check)')
plt.tight_layout()
plt.show()
```

## Best Practices & Interpretation Guidelines

---

### Correlation Interpretation Guidelines

```python
def interpret_correlation(r):
    """
    Interpret correlation strength based on common guidelines
    """
    r_abs = abs(r)
    
    if r_abs >= 0.9:
        strength = "Very Strong"
    elif r_abs >= 0.7:
        strength = "Strong"
    elif r_abs >= 0.5:
        strength = "Moderate"
    elif r_abs >= 0.3:
        strength = "Weak"
    else:
        strength = "Very Weak/No"
    
    direction = "Positive" if r > 0 else "Negative" if r < 0 else "No"
    
    return f"{direction} {strength} correlation (r = {r:.3f})"

# Example interpretations
correlations_to_interpret = [0.95, 0.75, 0.45, 0.25, -0.65, 0.05]

print("Correlation Interpretations:")
for corr in correlations_to_interpret:
    print(f"r = {corr:5.2f}: {interpret_correlation(corr)}")
```

### Common Pitfalls and Warnings

````python
def correlation_warnings_demo():
    """
    Demonstrate common pitfalls in correlation analysis
    """
    np.random.seed(42)
    n = 1000
    
    # 1. Non-linear relationship
    x1 = np.linspace(-3, 3, n)
    y1 = x1**2 + np.random.normal(0, 1, n)
    
    # 2. Outlier effect
    x2 = np.random.normal(0, 1, n)
    y2 = x2 + np.random.normal(0, 0.5, n)
    # Add outliers
    x2 = np.append(x2, [5, -5])
    y2 = np.append(y2, [-5, 5])
    
    # 3. Restriction of range
    x3_full = np.random.normal(0, 2, n)
    y3_full = x3_full * 0.8 + np.random.normal(0, 1, n)
    # Restrict range
    mask = (x3_full > -1) & (x3_full < 1)
    x3_restricted = x3_full[mask]
    y3_restricted = y3_full[mask]
    
    # Calculate correlations
    corr1 = np.corrcoef(x1, y1)[0, 1]
    corr2_with_outliers = np.corrcoef(x2, y2)[0, 1]
    corr2_without_outliers = np.corrcoef(x2[:-2], y2[:-2])[0, 1]
    corr3_full = np.corrcoef(x3_full, y3_full)[0, 1]
    corr3_restricted = np.corrcoef(x3_restricted, y3_restricted)[0, 1]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Non-linear relationship
    axes[0, 0].scatter(x1, y1, alpha=0.6)
    axes[0, 0].set_title(f'Non-linear Relationship\nPearson r = {corr1:.3f}')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    # Outlier effect
    axes[0, 1].scatter(x2[:-2], y2[:-2], alpha=0.6, label='Normal data')
    axes[0, 1].scatter(x2[-2:], y2[-2:], color='red', s=100, label='Outliers')
    axes[0, 1].set_title(f'Outlier Effect\nWith outliers: r = {corr2_with_outliers:.3f}\nWithout: r = {corr2_without_outliers:.3f}')
    axes[0, 1].legend()
    
    # Range restriction
    axes[1, 0].scatter(x3_full, y3_full, alpha=0.3, label=f'Full range (r = {corr3_full:.3f})')
    axes[1, 0].scatter(x3_restricted, y3_restricted, alpha=0.8, label=f'Restricted (r = {corr3_restricted:.3f})')
    axes[1, 0].axvline(-1, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(1, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Range Restriction Effect')
    axes[1, 0].legend()
    
    # Correlation vs Causation reminder
    axes[1, 1].text(0.5, 0.5, 'Remember:\n\nCorrelation ≠ Causation\n\n• Third variable problem\n• Spurious correlations\n• Bidirectional causality\n• Confounding variables', 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Key Warnings:")
    print("1. Non-linear relationships may show low Pearson correlation")
    print("2. Outliers can dramatically affect correlation values")
    print("3. Range restriction reduces apparent correlation")
    print("4. Always remember: Correlation ≠ Causation")

correlation_warnings_demo()

## Summary & Key Takeaways
---

### Quick Reference Guide

```python
def correlation_analysis_checklist():
    """
    Checklist for proper correlation analysis
    """
    checklist = {
        "Data Preparation": [
            "Check for missing values",
            "Handle outliers appropriately", 
            "Ensure adequate sample size",
            "Check data distributions"
        ],
        "Analysis Steps": [
            "Choose appropriate correlation measure",
            "Test statistical significance",
            "Consider multiple testing correction",
            "Check for non-linear relationships"
        ],
        "Interpretation": [
            "Consider correlation strength guidelines",
            "Don't confuse correlation with causation",
            "Account for potential confounding variables",
            "Validate findings with domain knowledge"
        ],
        "Practical Applications": [
            "Feature selection for ML models",
            "Multicollinearity detection",
            "Data quality assessment",
            "Relationship discovery"
        ]
    }
    
    print("CORRELATION ANALYSIS CHECKLIST")
    print("=" * 40)
    for category, items in checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  ☐ {item}")

correlation_analysis_checklist()
````

### When to Use Which Correlation Method

```python
def correlation_method_guide():
    """
    Guide for choosing the right correlation method
    """
    methods = {
        "Pearson": {
            "Use when": "Both variables are continuous and normally distributed",
            "Measures": "Linear relationships",
            "Range": "-1 to +1",
            "Example": "Height vs Weight"
        },
        "Spearman": {
            "Use when": "Variables are ordinal or non-normally distributed",
            "Measures": "Monotonic relationships (including non-linear)",
            "Range": "-1 to +1", 
            "Example": "Education level vs Job satisfaction rank"
        },
        "Kendall": {
            "Use when": "Small sample sizes or many tied values",
            "Measures": "Concordance of rankings",
            "Range": "-1 to +1",
            "Example": "Ranking of preferences"
        },
        "Partial": {
            "Use when": "Need to control for confounding variables",
            "Measures": "Relationship after removing effect of other variables",
            "Range": "-1 to +1",
            "Example": "Age vs Income, controlling for Education"
        }
    }
    
    print("CORRELATION METHOD SELECTION GUIDE")
    print("=" * 45)
    for method, details in methods.items():
        print(f"\n{method} Correlation:")
        for key, value in details.items():
            print(f"  {key}: {value}")

correlation_method_guide()
```

Memahami korelasi dan kovarians adalah kunci untuk analisis data yang efektif. Ingat bahwa ini adalah alat eksploratori yang powerful, tapi selalu harus diinterpretasi dengan hati-hati dan dikombinasikan dengan domain knowledge. Selamat mengeksplorasi pola-pola menarik dalam data Anda!