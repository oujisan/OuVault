# [data] #05 - Data Distributions

![data](https://raw.githubusercontent.com/oujisan/OuVault/main/img/data.png)

## Data Distributions: Normal, Uniform, Skewed, and Outliers

---

Understanding distribusi data adalah kunci untuk memilih algoritma machine learning yang tepat, melakukan preprocessing yang efektif, dan menginterpretasi hasil model dengan benar. Mari kita pelajari berbagai jenis distribusi yang sering ditemui dalam data science dan implikasinya terhadap machine learning.

## Why Distribution Matters in Machine Learning

---

Distribusi data mempengaruhi:

- **Algorithm performance**: Beberapa algoritma bekerja optimal pada data normal
- **Feature engineering**: Transformasi yang diperlukan untuk improve model
- **Outlier detection**: Metode deteksi yang sesuai dengan distribusi
- **Statistical inference**: Validitas dari statistical tests dan confidence intervals
- **Model assumptions**: Banyak model mengasumsikan distribusi tertentu

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, uniform, expon, chi2, t, gamma, beta
from sklearn.preprocessing import StandardScaler, PowerTransformer
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📊 Understanding Data Distributions for Machine Learning")
print("=" * 60)
```

## Normal Distribution (Gaussian)

---

Normal distribution adalah "gold standard" dalam statistik dan machine learning karena banyak algoritma mengasumsikan data terdistribusi normal.

### Characteristics of Normal Distribution

```python
# Generate normal distribution data
np.random.seed(42)
n_samples = 10000

# Different normal distributions
normal_std = np.random.normal(0, 1, n_samples)  # Standard normal (μ=0, σ=1)
normal_custom = np.random.normal(100, 15, n_samples)  # Custom normal (μ=100, σ=15)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Normal Distribution Analysis', fontsize=16)

# Standard normal distribution
axes[0,0].hist(normal_std, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
x = np.linspace(-4, 4, 100)
y = norm.pdf(x, 0, 1)
axes[0,0].plot(x, y, 'r-', linewidth=2, label='Theoretical PDF')
axes[0,0].axvline(0, color='green', linestyle='--', label='Mean=0')
axes[0,0].set_title('Standard Normal Distribution (μ=0, σ=1)')
axes[0,0].set_xlabel('Value')
axes[0,0].set_ylabel('Density')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Custom normal distribution
axes[0,1].hist(normal_custom, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
x = np.linspace(40, 160, 100)
y = norm.pdf(x, 100, 15)
axes[0,1].plot(x, y, 'r-', linewidth=2, label='Theoretical PDF')
axes[0,1].axvline(100, color='green', linestyle='--', label='Mean=100')
axes[0,1].set_title('Custom Normal Distribution (μ=100, σ=15)')
axes[0,1].set_xlabel('Value')
axes[0,1].set_ylabel('Density')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Q-Q plot for normality testing
stats.probplot(normal_std, dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot: Standard Normal')
axes[1,0].grid(True, alpha=0.3)

# Multiple normal distributions comparison
means = [0, 0, 0]
stds = [0.5, 1, 2]
colors = ['blue', 'red', 'green']
x = np.linspace(-6, 6, 100)

for mean, std, color in zip(means, stds, colors):
    y = norm.pdf(x, mean, std)
    axes[1,1].plot(x, y, color=color, linewidth=2, label=f'μ={mean}, σ={std}')

axes[1,1].set_title('Normal Distributions with Different Standard Deviations')
axes[1,1].set_xlabel('Value')
axes[1,1].set_ylabel('Density')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical properties
print("📈 Normal Distribution Properties:")
print(f"Standard Normal:")
print(f"  Mean: {np.mean(normal_std):.3f} (theoretical: 0)")
print(f"  Std Dev: {np.std(normal_std, ddof=1):.3f} (theoretical: 1)")
print(f"  Skewness: {stats.skew(normal_std):.3f} (theoretical: 0)")
print(f"  Kurtosis: {stats.kurtosis(normal_std):.3f} (theoretical: 0)")

print(f"\nCustom Normal:")
print(f"  Mean: {np.mean(normal_custom):.3f} (theoretical: 100)")
print(f"  Std Dev: {np.std(normal_custom, ddof=1):.3f} (theoretical: 15)")
print(f"  Skewness: {stats.skew(normal_custom):.3f} (theoretical: 0)")
print(f"  Kurtosis: {stats.kurtosis(normal_custom):.3f} (theoretical: 0)")
```

### Empirical Rule (68-95-99.7 Rule)

```python
def empirical_rule_analysis(data, name="Data"):
    """Analyze data using empirical rule"""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # Calculate percentages within standard deviations
    within_1sd = np.sum(np.abs(data - mean) <= std) / len(data) * 100
    within_2sd = np.sum(np.abs(data - mean) <= 2*std) / len(data) * 100
    within_3sd = np.sum(np.abs(data - mean) <= 3*std) / len(data) * 100
    
    print(f"📊 Empirical Rule Analysis for {name}:")
    print(f"  Within 1 SD (μ±σ): {within_1sd:.1f}% (expected: 68.3%)")
    print(f"  Within 2 SD (μ±2σ): {within_2sd:.1f}% (expected: 95.4%)")
    print(f"  Within 3 SD (μ±3σ): {within_3sd:.1f}% (expected: 99.7%)")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    
    # Add vertical lines for standard deviations
    colors = ['red', 'orange', 'green']
    for i, color in enumerate(colors, 1):
        plt.axvline(mean - i*std, color=color, linestyle='--', alpha=0.7, 
                   label=f'μ-{i}σ = {mean - i*std:.2f}')
        plt.axvline(mean + i*std, color=color, linestyle='--', alpha=0.7,
                   label=f'μ+{i}σ = {mean + i*std:.2f}')
    
    plt.axvline(mean, color='black', linestyle='-', linewidth=2, label=f'Mean = {mean:.2f}')
    plt.title(f'Empirical Rule Visualization - {name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return within_1sd, within_2sd, within_3sd

# Test with normal data
empirical_rule_analysis(normal_custom, "Normal Distribution")
```

### Testing for Normality

```python
def test_normality(data, name="Data"):
    """Comprehensive normality testing"""
    print(f"🔍 Normality Tests for {name}:")
    print("=" * 40)
    
    # Shapiro-Wilk test (best for n < 5000)
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
        print(f"Shapiro-Wilk Test:")
        print(f"  Statistic: {shapiro_stat:.4f}")
        print(f"  p-value: {shapiro_p:.4f}")
        print(f"  Result: {'Normal' if shapiro_p > 0.05 else 'Not Normal'} (α=0.05)")
    else:
        print("Shapiro-Wilk Test: Skipped (sample too large)")
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_p:.4f}")
    print(f"  Result: {'Normal' if ks_p > 0.05 else 'Not Normal'} (α=0.05)")
    
    # Anderson-Darling test
    ad_stat, ad_critical, ad_significance = stats.anderson(data, dist='norm')
    print(f"\nAnderson-Darling Test:")
    print(f"  Statistic: {ad_stat:.4f}")
    for i, (sig_level, crit_val) in enumerate(zip(ad_significance, ad_critical)):
        result = "Normal" if ad_stat < crit_val else "Not Normal"
        print(f"  At {sig_level:4.1f}% significance: {result} (critical value: {crit_val:.4f})")
    
    # Visual assessment
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram with normal overlay
    axes[0].hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    mean, std = np.mean(data), np.std(data, ddof=1)
    x = np.linspace(data.min(), data.max(), 100)
    axes[0].plot(x, norm.pdf(x, mean, std), 'r-', linewidth=2, label='Normal PDF')
    axes[0].set_title(f'Histogram vs Normal PDF - {name}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot - {name}')
    axes[1].grid(True, alpha=0.3)
    
    # Box plot
    axes[2].boxplot(data)
    axes[2].set_title(f'Box Plot - {name}')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Test different distributions
# Normal data
test_normality(normal_custom, "Normal Distribution")

# Generate non-normal data for comparison
non_normal = np.random.exponential(2, 1000)
test_normality(non_normal, "Exponential Distribution")
```

## Uniform Distribution

---

Uniform distribution memberikan equal probability untuk semua nilai dalam range tertentu.

### Understanding Uniform Distribution

```python
# Generate uniform distribution data
uniform_data = np.random.uniform(0, 10, 10000)  # Uniform between 0 and 10
discrete_uniform = np.random.randint(1, 7, 10000)  # Dice rolls

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Uniform Distribution Analysis', fontsize=16)

# Continuous uniform distribution
axes[0,0].hist(uniform_data, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
x = np.linspace(0, 10, 100)
y = uniform.pdf(x, 0, 10)  # uniform.pdf(x, loc, scale)
axes[0,0].plot(x, y, 'r-', linewidth=3, label='Theoretical PDF')
axes[0,0].set_title('Continuous Uniform Distribution [0, 10]')
axes[0,0].set_xlabel('Value')
axes[0,0].set_ylabel('Density')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Discrete uniform distribution (dice)
dice_counts = np.bincount(discrete_uniform)[1:]  # Remove 0 count
dice_probs = dice_counts / len(discrete_uniform)
axes[0,1].bar(range(1, 7), dice_probs, alpha=0.7, color='orange', edgecolor='black')
axes[0,1].axhline(1/6, color='red', linestyle='--', linewidth=2, label='Expected probability (1/6)')
axes[0,1].set_title('Discrete Uniform Distribution (Dice Rolls)')
axes[0,1].set_xlabel('Dice Value')
axes[0,1].set_ylabel('Probability')
axes[0,1].set_xticks(range(1, 7))
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Comparison of different uniform ranges
ranges = [(0, 1), (0, 5), (0, 10)]
colors = ['blue', 'green', 'red']
x = np.linspace(-1, 11, 1000)

for (a, b), color in zip(ranges, colors):
    y = uniform.pdf(x, a, b-a)  # scale = b-a for uniform distribution
    axes[1,0].plot(x, y, color=color, linewidth=2, label=f'Uniform[{a}, {b}]')

axes[1,0].set_title('Uniform Distributions with Different Ranges')
axes[1,0].set_xlabel('Value')
axes[1,0].set_ylabel('Density')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_xlim(-1, 11)

# Statistical properties comparison
uniform_stats = pd.DataFrame({
    'Distribution': ['Uniform[0,10]', 'Normal(5,2)', 'Exponential(λ=2)'],
    'Mean': [np.mean(uniform_data), 5, 2],
    'Variance': [np.var(uniform_data, ddof=1), 4, 4],
    'Skewness': [stats.skew(uniform_data), 0, 2],
    'Kurtosis': [stats.kurtosis(uniform_data), 0, 6]
})

# Create table visualization
axes[1,1].axis('tight')
axes[1,1].axis('off')
table = axes[1,1].table(cellText=uniform_stats.round(3).values,
                       colLabels=uniform_stats.columns,
                       cellLoc='center',
                       loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
axes[1,1].set_title('Statistical Properties Comparison')

plt.tight_layout()
plt.show()

print("📊 Uniform Distribution Properties:")
print(f"Continuous Uniform [0, 10]:")
print(f"  Mean: {np.mean(uniform_data):.3f} (theoretical: 5.0)")
print(f"  Variance: {np.var(uniform_data, ddof=1):.3f} (theoretical: 8.333)")
print(f"  Skewness: {stats.skew(uniform_data):.3f} (theoretical: 0)")
print(f"  Kurtosis: {stats.kurtosis(uniform_data):.3f} (theoretical: -1.2)")

print(f"\nDiscrete Uniform (Dice):")
print(f"  Mean: {np.mean(discrete_uniform):.3f} (theoretical: 3.5)")
print(f"  Variance: {np.var(discrete_uniform, ddof=1):.3f} (theoretical: 2.917)")
```

### Applications of Uniform Distribution

```python
# Real-world applications of uniform distribution
print("🎲 Uniform Distribution Applications:")
applications = [
    "Random number generation and simulation",
    "Monte Carlo methods and sampling",
    "A/B testing with random assignment",
    "Random initialization in neural networks",
    "Cryptographic key generation",
    "Fair dice and lottery systems",
    "Bootstrap resampling techniques"
]

for i, app in enumerate(applications, 1):
    print(f"  {i}. {app}")

# Monte Carlo simulation example
def monte_carlo_pi_estimation(n_samples):
    """Estimate π using Monte Carlo method with uniform distribution"""
    # Generate random points in unit square
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # Check if points are inside unit circle
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    
    return pi_estimate, x, y, inside_circle

# Run simulation
n_points = 5000
pi_est, x_points, y_points, inside = monte_carlo_pi_estimation(n_points)

# Visualize Monte Carlo simulation
plt.figure(figsize=(10, 8))
colors = ['red' if not inside else 'blue' for inside in inside]
plt.scatter(x_points, y_points, c=colors, alpha=0.6, s=1)

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
plt.plot(circle_x, circle_y, 'black', linewidth=2)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect('equal')
plt.title(f'Monte Carlo π Estimation\nEstimate: {pi_est:.4f}, Actual π: {np.pi:.4f}, Error: {abs(pi_est - np.pi):.4f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Monte Carlo π estimation with {n_points:,} points:")
print(f"  Estimated π: {pi_est:.6f}")
print(f"  Actual π: {np.pi:.6f}")
print(f"  Absolute error: {abs(pi_est - np.pi):.6f}")
print(f"  Relative error: {abs(pi_est - np.pi)/np.pi*100:.3f}%")
```

## Skewed Distributions

---

Real-world data sering menunjukkan skewness - asymmetry dalam distribusi.

### Right-Skewed (Positive Skew) Distribution

```python
# Generate skewed distributions
np.random.seed(42)
n_samples = 10000

# Right-skewed distributions
exponential_data = np.random.exponential(2, n_samples)
lognormal_data = np.random.lognormal(2, 0.5, n_samples)
gamma_data = np.random.gamma(2, 2, n_samples)

# Left-skewed distribution (negative skew)
# Create by reflecting and shifting right-skewed data
left_skewed = 100 - np.random.exponential(3, n_samples)

# Create comprehensive skewness analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Skewed Distributions Analysis', fontsize=16)

# Right-skewed distributions
skewed_data = [
    (exponential_data, 'Exponential (λ=2)', 'lightcoral'),
    (lognormal_data, 'Log-Normal (μ=2, σ=0.5)', 'lightblue'),
    (gamma_data, 'Gamma (α=2, β=2)', 'lightgreen')
]

for i, (data, title, color) in enumerate(skewed_data):
    axes[0, i].hist(data, bins=50, density=True, alpha=0.7, color=color, edgecolor='black')
    
    # Add statistical lines
    mean_val = np.mean(data)
    median_val = np.median(data)
    mode_val = stats.mode(data, keepdims=True).mode[0]
    
    axes[0, i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    axes[0, i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    axes[0, i].axvline(mode_val, color='blue', linestyle='--', label=f'Mode: {mode_val:.2f}')
    
    skewness = stats.skew(data)
    axes[0, i].set_title(f'{title}\nSkewness: {skewness:.3f}')
    axes[0, i].set_xlabel('Value')
    axes[0, i].set_ylabel('Density')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)

# Left-skewed distribution
axes[1, 0].hist(left_skewed, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
mean_val = np.mean(left_skewed)
median_val = np.median(left_skewed)
skewness = stats.skew(left_skewed)

axes[1, 0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
axes[1, 0].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
axes[1, 0].set_title(f'Left-Skewed Distribution\nSkewness: {skewness:.3f}')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Box plots comparison
box_data = [exponential_data, lognormal_data, gamma_data, left_skewed]
box_labels = ['Exponential', 'Log-Normal', 'Gamma', 'Left-Skewed']
axes[1, 1].boxplot(box_data, labels=box_labels)
axes[1, 1].set_title('Box Plot Comparison of Skewed Distributions')
axes[1, 1].set_ylabel('Value')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

# Skewness interpretation guide
skew_guide = {
    'Distribution': ['Exponential', 'Log-Normal', 'Gamma', 'Left-Skewed'],
    'Skewness': [stats.skew(data) for data, _, _ in skewed_data] + [stats.skew(left_skewed)],
    'Interpretation': []
}

for skew_val in skew_guide['Skewness']:
    if skew_val > 1:
        interp = 'Highly Right-Skewed'
    elif skew_val > 0.5:
        interp = 'Moderately Right-Skewed'
    elif skew_val > -0.5:
        interp = 'Approximately Symmetric'
    elif skew_val > -1:
        interp = 'Moderately Left-Skewed'
    else:
        interp = 'Highly Left-Skewed'
    skew_guide['Interpretation'].append(interp)

skew_df = pd.DataFrame(skew_guide)
skew_df['Skewness'] = skew_df['Skewness'].round(3)

# Create table
axes[1, 2].axis('tight')
axes[1, 2].axis('off')
table = axes[1, 2].table(cellText=skew_df.values,
                        colLabels=skew_df.columns,
                        cellLoc='center',
                        loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
axes[1, 2].set_title('Skewness Interpretation Guide')

plt.tight_layout()
plt.show()

# Statistical analysis of skewed data
print("📈 Skewness Analysis:")
print("=" * 40)

distributions_analysis = {
    'Exponential': exponential_data,
    'Log-Normal': lognormal_data,
    'Gamma': gamma_data,
    'Left-Skewed': left_skewed
}

for name, data in distributions_analysis.items():
    print(f"\n{name} Distribution:")
    print(f"  Mean: {np.mean(data):.3f}")
    print(f"  Median: {np.median(data):.3f}")
    print(f"  Mode: {stats.mode(data, keepdims=True).mode[0]:.3f}")
    print(f"  Std Dev: {np.std(data, ddof=1):.3f}")
    print(f"  Skewness: {stats.skew(data):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(data):.3f}")
    
    # Relationship between mean, median, and mode for skewed data
    mean_val = np.mean(data)
    median_val = np.median(data)
    if mean_val > median_val:
        print(f"  → Right-skewed: Mean ({mean_val:.2f}) > Median ({median_val:.2f})")
    elif mean_val < median_val:
        print(f"  → Left-skewed: Mean ({mean_val:.2f}) < Median ({median_val:.2f})")
    else:
        print(f"  → Symmetric: Mean ≈ Median")
```

### Handling Skewed Data for Machine Learning

```python
def handle_skewed_data(data, method='log'):
    """
    Transform skewed data using various methods
    """
    transformations = {}
    
    # Original data
    transformations['Original'] = data
    
    # Log transformation (for positive skewed data)
    if method == 'log' and np.all(data > 0):
        transformations['Log'] = np.log(data)
    
    # Square root transformation
    if np.all(data >= 0):
        transformations['Square Root'] = np.sqrt(data)
    
    # Box-Cox transformation
    if np.all(data > 0):
        transformed_data, lambda_val = stats.boxcox(data)
        transformations[f'Box-Cox (λ={lambda_val:.3f})'] = transformed_data
    
    # Yeo-Johnson transformation (can handle negative values)
    pt = PowerTransformer(method='yeo-johnson')
    transformations['Yeo-Johnson'] = pt.fit_transform(data.reshape(-1, 1)).flatten()
    
    return transformations

# Apply transformations to right-skewed data
original_skewed = exponential_data
transformations = handle_skewed_data(original_skewed)

# Visualize transformations
n_transforms = len(transformations)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Skewness Correction Transformations', fontsize=16)
axes = axes.flatten()

for i, (name, data) in enumerate(transformations.items()):
    if i < len(axes):
        axes[i].hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        skewness = stats.skew(data)
        axes[i].set_title(f'{name}\nSkewness: {skewness:.3f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].grid(True, alpha=0.3)
        
        # Color code based on skewness improvement
        if abs(skewness) < 0.5:
            axes[i].set_facecolor('lightgreen')
            axes[i].set_alpha(0.1)

# Hide unused subplot
if len(transformations) < len(axes):
    for j in range(len(transformations), len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# Compare transformation effectiveness
print("🔧 Transformation Effectiveness:")
print("=" * 50)

for name, data in transformations.items():
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    # Normality test (Shapiro-Wilk for smaller samples)
    if len(data) <= 5000:
        _, p_value = stats.shapiro(data)
        normal_test = f"p={p_value:.4f} ({'Normal' if p_value > 0.05 else 'Not Normal'})"
    else:
        normal_test = "Sample too large for Shapiro-Wilk"
    
    print(f"{name}:")
    print(f"  Skewness: {skewness:7.3f} ({'Good' if abs(skewness) < 0.5 else 'Poor'})")
    print(f"  Kurtosis: {kurtosis:7.3f}")
    print(f"  Normality: {normal_test}")
    print()
```

## Outlier Detection and Treatment

---

Outliers dapat significantly mempengaruhi performance machine learning models.

### Statistical Methods for Outlier Detection

```python
def comprehensive_outlier_detection(data, name="Data"):
    """
    Comprehensive outlier detection using multiple methods
    """
    print(f"🔍 Outlier Detection Analysis for {name}")
    print("=" * 50)
    
    outlier_methods = {}
    
    # 1. Z-Score Method
    z_scores = np.abs(stats.zscore(data))
    z_outliers = data[z_scores > 3]  # 3-sigma rule
    outlier_methods['Z-Score (>3σ)'] = z_outliers
    
    # 2. Modified Z-Score (using median)
    median = np.median(data)
    mad = np.median(np.abs(data - median))  # Median Absolute Deviation
    modified_z_scores = 0.6745 * (data - median) / mad
    modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
    outlier_methods['Modified Z-Score'] = modified_z_outliers
    
    # 3. IQR Method
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_methods['IQR Method'] = iqr_outliers
    
    # 4. Isolation Forest (for multivariate data simulation)
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = iso_forest.fit_predict(data.reshape(-1, 1))
    iso_outliers = data[outlier_labels == -1]
    outlier_methods['Isolation Forest'] = iso_outliers
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Outlier Detection Methods - {name}', fontsize=16)
    
    # Original data histogram
    axes[0, 0].hist(data, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_title(f'Original Data Distribution\nn={len(data)}')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot with outliers
    bp = axes[0, 1].boxplot(data, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[0, 1].set_title('Box Plot with Outliers')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Z-Score visualization
    axes[0, 2].scatter(range(len(data)), z_scores, alpha=0.6, s=1)
    axes[0, 2].axhline(y=3, color='red', linestyle='--', label='3σ threshold')
    axes[0, 2].set_title('Z-Scores')
    axes[0, 2].set_xlabel('Data Point Index')
    axes[0, 2].set_ylabel('|Z-Score|')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Outlier counts comparison
    method_names = list(outlier_methods.keys())
    outlier_counts = [len(outliers) for outliers in outlier_methods.values()]
    
    bars = axes[1, 0].bar(method_names, outlier_counts, alpha=0.7, color=['red', 'orange', 'yellow', 'green'])
    axes[1, 0].set_title('Outlier Count by Method')
    axes[1, 0].set_ylabel('Number of Outliers')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, outlier_counts):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom', fontweight='bold')
    
    # Data without outliers (IQR method)
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    axes[1, 1].hist(clean_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title(f'Data After Outlier Removal (IQR)\nn={len(clean_data)}')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistical comparison
    stats_comparison = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
        'Original': [len(data), np.mean(data), np.median(data), 
                    np.std(data, ddof=1), stats.skew(data), stats.kurtosis(data)],
        'After Cleaning': [len(clean_data), np.mean(clean_data), np.median(clean_data),
                          np.std(clean_data, ddof=1), stats.skew(clean_data), stats.kurtosis(clean_data)]
    })
    
    # Display table
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')
    table = axes[1, 2].table(cellText=stats_comparison.round(3).values,
                            colLabels=stats_comparison.columns,
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 2].set_title('Statistical Comparison')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    for method, outliers in outlier_methods.items():
        percentage = (len(outliers) / len(data)) * 100
        print(f"{method}:")
        print(f"  Outliers found: {len(outliers)} ({percentage:.2f}%)")
        if len(outliers) > 0 and len(outliers) <= 10:
            print(f"  Outlier values: {np.sort(outliers)}")
        elif len(outliers) > 10:
            print(f"  Sample outliers: {np.sort(outliers)[:5]} ... {np.sort(outliers)[-5:]}")
        print()
    
    return outlier_methods, clean_data

# Test with contaminated normal data
np.random.seed(42)
normal_with_outliers = np.concatenate([
    np.random.normal(50, 10, 1000),  # Normal data
    np.array([100, 105, 110, 0, -5, 120])  # Artificial outliers
])

outlier_results, cleaned_data = comprehensive_outlier_detection(normal_with_outliers, "Normal Data with Outliers")
```

### Outlier Treatment Strategies

```python
def outlier_treatment_strategies(data, outliers_mask):
    """
    Demonstrate different outlier treatment approaches
    """
    treatment_results = {}
    
    # 1. Removal (deletion)
    clean_data = data[~outliers_mask]
    treatment_results['Removal'] = clean_data
    
    # 2. Capping/Winsorizing
    from scipy.stats.mstats import winsorize
    winsorized_data = winsorize(data, limits=[0.05, 0.05])  # Cap at 5th and 95th percentiles
    treatment_results['Winsorizing'] = winsorized_data
    
    # 3. Log transformation (if all positive)
    if np.all(data > 0):
        log_transformed = np.log(data)
        treatment_results['Log Transform'] = log_transformed
    
    # 4. Median imputation for outliers
    median_imputed = data.copy()
    median_val = np.median(data[~outliers_mask])
    median_imputed[outliers_mask] = median_val
    treatment_results['Median Imputation'] = median_imputed
    
    # 5. Mean imputation for outliers
    mean_imputed = data.copy()
    mean_val = np.mean(data[~outliers_mask])
    mean_imputed[outliers_mask] = mean_val
    treatment_results['Mean Imputation'] = mean_imputed
    
    return treatment_results

# Apply outlier treatments
# Use IQR method to identify outliers
Q1 = np.percentile(normal_with_outliers, 25)
Q3 = np.percentile(normal_with_outliers, 75)
IQR = Q3 - Q1
outlier_mask = (normal_with_outliers < Q1 - 1.5*IQR) | (normal_with_outliers > Q3 + 1.5*IQR)

treatments = outlier_treatment_strategies(normal_with_outliers, outlier_mask)

# Visualize treatment results
n_treatments = len(treatments)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Outlier Treatment Strategies Comparison', fontsize=16)
axes = axes.flatten()

# Add original data
all_treatments = {'Original': normal_with_outliers}
all_treatments.update(treatments)

for i, (name, treated_data) in enumerate(all_treatments.items()):
    if i < len(axes):
        axes[i].hist(treated_data, bins=30, alpha=0.7, edgecolor='black')
        
        # Statistical summary
        mean_val = np.mean(treated_data)
        std_val = np.std(treated_data, ddof=1)
        skew_val = stats.skew(treated_data)
        
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
        
        axes[i].set_title(f'{name}\nMean={mean_val:.2f}, σ={std_val:.2f}, Skew={skew_val:.3f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Color-code based on improvement
        if name != 'Original' and abs(skew_val) < abs(stats.skew(normal_with_outliers)):
            axes[i].patch.set_facecolor('lightgreen')
            axes[i].patch.set_alpha(0.3)

# Hide unused subplots
for j in range(len(all_treatments), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# Treatment effectiveness summary
print("🔧 Outlier Treatment Effectiveness:")
print("=" * 50)

original_stats = {
    'Mean': np.mean(normal_with_outliers),
    'Std Dev': np.std(normal_with_outliers, ddof=1),
    'Skewness': stats.skew(normal_with_outliers),
    'Sample Size': len(normal_with_outliers)
}

print(f"Original Data Statistics:")
for key, value in original_stats.items():
    print(f"  {key}: {value:.3f}")

print(f"\nTreatment Comparisons:")
for name, treated_data in treatments.items():
    mean_change = np.mean(treated_data) - original_stats['Mean']
    std_change = np.std(treated_data, ddof=1) - original_stats['Std Dev']
    skew_change = stats.skew(treated_data) - original_stats['Skewness']
    size_change = len(treated_data) - original_stats['Sample Size']
    
    print(f"\n{name}:")
    print(f"  Mean change: {mean_change:+.3f}")
    print(f"  Std Dev change: {std_change:+.3f}")
    print(f"  Skewness change: {skew_change:+.3f}")
    print(f"  Sample size change: {size_change:+d}")
    
    # Recommendation
    if abs(skew_change) > 0.1:
        improvement = "improves" if abs(stats.skew(treated_data)) < abs(original_stats['Skewness']) else "worsens"
        print(f"  → {improvement.capitalize()} distribution symmetry")
```

## Real-world Distribution Analysis

---

### E-commerce Sales Data Case Study

```python
# Generate realistic e-commerce sales data with different distributions
np.random.seed(42)

# Simulate different product categories with different distribution patterns
categories = {
    'Electronics': {
        'size': 2000,
        'distribution': 'lognormal',
        'params': {'mean': np.log(500), 'sigma': 0.8},
        'outliers': [2500, 3000, 3500]  # High-value items
    },
    'Clothing': {
        'size': 3000,
        'distribution': 'gamma',
        'params': {'a': 2, 'scale': 25},
        'outliers': [200, 250, 300]  # Designer items
    },
    'Books': {
        'size': 1500,
        'distribution': 'normal',
        'params': {'loc': 25, 'scale': 8},
        'outliers': [80, 90, 100]  # Textbooks/specialty books
    },
    'Home_Garden': {
        'size': 1000,
        'distribution': 'exponential',
        'params': {'scale': 45},
        'outliers': [500, 600, 750]  # Large furniture/appliances
    }
}

# Generate sales data
sales_data = []
for category, config in categories.items():
    size = config['size']
    dist_type = config['distribution']
    params = config['params']
    outliers = config['outliers']
    
    # Generate base data
    if dist_type == 'lognormal':
        prices = np.random.lognormal(params['mean'], params['sigma'], size)
    elif dist_type == 'gamma':
        prices = np.random.gamma(params['a'], scale=params['scale'], size)
    elif dist_type == 'normal':
        prices = np.random.normal(params['loc'], params['scale'], size)
    elif dist_type == 'exponential':
        prices = np.random.exponential(params['scale'], size)
    
    # Add outliers (5% of the data)
    n_outliers = len(outliers)
    outlier_indices = np.random.choice(size, n_outliers, replace=False)
    prices[outlier_indices] = outliers
    
    # Create DataFrame
    category_df = pd.DataFrame({
        'category': category,
        'price': np.maximum(prices, 1),  # Ensure positive prices
        'month': np.random.randint(1, 13, size),
        'quantity_sold': np.random.poisson(5, size)
    })
    
    sales_data.append(category_df)

# Combine all categories
ecommerce_sales = pd.concat(sales_data, ignore_index=True)
ecommerce_sales['revenue'] = ecommerce_sales['price'] * ecommerce_sales['quantity_sold']

print("🛒 E-commerce Sales Distribution Analysis")
print("=" * 60)
print(f"Total transactions: {len(ecommerce_sales):,}")
print(f"Categories: {ecommerce_sales['category'].value_counts().to_dict()}")

# Comprehensive distribution analysis
fig, axes = plt.subplots(3, 4, figsize=(20, 18))
fig.suptitle('E-commerce Sales Distribution Analysis', fontsize=16)

# Price distributions by category
for i, category in enumerate(categories.keys()):
    cat_data = ecommerce_sales[ecommerce_sales['category'] == category]['price']
    
    # Histogram
    axes[0, i].hist(cat_data, bins=50, alpha=0.7, edgecolor='black')
    axes[0, i].set_title(f'{category.replace("_", " ")} - Price Distribution')
    axes[0, i].set_xlabel('Price ($)')
    axes[0, i].set_ylabel('Frequency')
    axes[0, i].grid(True, alpha=0.3)
    
    # Add statistics
    mean_price = cat_data.mean()
    median_price = cat_data.median()
    skewness = stats.skew(cat_data)
    
    axes[0, i].axvline(mean_price, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: ${mean_price:.0f}')
    axes[0, i].axvline(median_price, color='green', linestyle='--', alpha=0.8,
                      label=f'Median: ${median_price:.0f}')
    axes[0, i].legend(fontsize=8)
    
    # Add skewness info
    axes[0, i].text(0.65, 0.95, f'Skew: {skewness:.2f}', 
                   transform=axes[0, i].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Box plots for price comparison
price_data = [ecommerce_sales[ecommerce_sales['category']==cat]['price'] 
              for cat in categories.keys()]
bp = axes[1, 0].boxplot(price_data, labels=[cat.replace('_', ' ') for cat in categories.keys()],
                        patch_artist=True)
axes[1, 0].set_title('Price Distribution Comparison')
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Color the boxes
colors = ['lightcoral', 'lightblue', 'lightgreen', 'orange']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Revenue distribution (typically more skewed)
axes[1, 1].hist(ecommerce_sales['revenue'], bins=50, alpha=0.7, 
               color='purple', edgecolor='black')
axes[1, 1].set_title('Revenue Distribution (All Categories)')
axes[1, 1].set_xlabel('Revenue ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

revenue_skewness = stats.skew(ecommerce_sales['revenue'])
axes[1, 1].text(0.65, 0.95, f'Skew: {revenue_skewness:.2f}', 
               transform=axes[1, 1].transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Log-transformed revenue
log_revenue = np.log(ecommerce_sales['revenue'])
axes[1, 2].hist(log_revenue, bins=50, alpha=0.7, 
               color='green', edgecolor='black')
axes[1, 2].set_title('Log-Transformed Revenue Distribution')
axes[1, 2].set_xlabel('Log(Revenue)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].grid(True, alpha=0.3)

log_skewness = stats.skew(log_revenue)
axes[1, 2].text(0.65, 0.95, f'Skew: {log_skewness:.2f}', 
               transform=axes[1, 2].transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Q-Q plot for normality assessment
stats.probplot(ecommerce_sales['price'], dist="norm", plot=axes[1, 3])
axes[1, 3].set_title('Q-Q Plot: All Prices vs Normal')
axes[1, 3].grid(True, alpha=0.3)

# Monthly sales trends by category
monthly_sales = ecommerce_sales.groupby(['month', 'category'])['revenue'].sum().unstack()
monthly_sales.plot(kind='line', ax=axes[2, 0], marker='o')
axes[2, 0].set_title('Monthly Revenue Trends by Category')
axes[2, 0].set_xlabel('Month')
axes[2, 0].set_ylabel('Total Revenue ($)')
axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[2, 0].grid(True, alpha=0.3)

# Price vs Quantity relationship
for i, category in enumerate(categories.keys()):
    cat_data = ecommerce_sales[ecommerce_sales['category'] == category]
    axes[2, 1].scatter(cat_data['price'], cat_data['quantity_sold'], 
                      alpha=0.6, label=category.replace('_', ' '), s=10)

axes[2, 1].set_title('Price vs Quantity Sold')
axes[2, 1].set_xlabel('Price ($)')
axes[2, 1].set_ylabel('Quantity Sold')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# Distribution statistics table
dist_stats = []
for category in categories.keys():
    cat_data = ecommerce_sales[ecommerce_sales['category'] == category]['price']
    stats_row = {
        'Category': category.replace('_', ' '),
        'Mean': f"${cat_data.mean():.0f}",
        'Median': f"${cat_data.median():.0f}",
        'Std Dev': f"${cat_data.std():.0f}",
        'Skewness': f"{stats.skew(cat_data):.3f}",
        'Outliers (IQR)': len(detect_outliers_iqr(cat_data)['outliers'])
    }
    dist_stats.append(stats_row)

stats_df = pd.DataFrame(dist_stats)

# Display statistics table
axes[2, 2].axis('tight')
axes[2, 2].axis('off')
table = axes[2, 2].table(cellText=stats_df.values,
                        colLabels=stats_df.columns,
                        cellLoc='center',
                        loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.5)
axes[2, 2].set_title('Category Distribution Statistics')

# Correlation heatmap
numeric_cols = ['price', 'quantity_sold', 'revenue', 'month']
correlation_matrix = ecommerce_sales[numeric_cols].corr()

im = axes[2, 3].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[2, 3].set_xticks(range(len(numeric_cols)))
axes[2, 3].set_yticks(range(len(numeric_cols)))
axes[2, 3].set_xticklabels(numeric_cols, rotation=45)
axes[2, 3].set_yticklabels(numeric_cols)
axes[2, 3].set_title('Variable Correlation Heatmap')

# Add correlation values
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        text = axes[2, 3].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')

plt.tight_layout()
plt.show()

# Detailed analysis
print("📊 Detailed Distribution Analysis:")
print("=" * 50)

for category in categories.keys():
    cat_data = ecommerce_sales[ecommerce_sales['category'] == category]
    price_data = cat_data['price']
    
    print(f"\n{category.replace('_', ' ')} Category:")
    print(f"  Transactions: {len(cat_data):,}")
    print(f"  Price Range: ${price_data.min():.0f} - ${price_data.max():.0f}")
    print(f"  Average Price: ${price_data.mean():.0f}")
    print(f"  Median Price: ${price_data.median():.0f}")
    print(f"  Price Std Dev: ${price_data.std():.0f}")
    print(f"  Skewness: {stats.skew(price_data):.3f}")
    
    # Identify distribution type
    skewness = stats.skew(price_data)
    if abs(skewness) < 0.5:
        dist_type = "Approximately Normal"
    elif skewness > 0.5:
        dist_type = "Right-Skewed"
    else:
        dist_type = "Left-Skewed"
    
    print(f"  Distribution Type: {dist_type}")
    
    # Business insights
    if price_data.mean() > price_data.median() * 1.2:
        print(f"  → High-value items significantly affect average price")
    
    if stats.skew(price_data) > 1:
        print(f"  → Consider log transformation for modeling")
    
    outliers = detect_outliers_iqr(price_data)['outliers']
    if len(outliers) > len(price_data) * 0.05:
        print(f"  → High outlier percentage ({len(outliers)/len(price_data)*100:.1f}%) - investigate data quality")

# Business recommendations
print(f"\n💡 BUSINESS INSIGHTS AND RECOMMENDATIONS:")
print("=" * 60)

recommendations = [
    "Electronics show log-normal distribution - typical for tech products with wide price ranges",
    "Books have most normal distribution - stable pricing with few premium items",
    "Home & Garden items show exponential pattern - many low-cost, few high-cost items", 
    "Revenue is highly right-skewed - few high-revenue transactions drive total sales",
    "Log transformation improves revenue distribution for statistical modeling",
    "Consider separate pricing strategies for each category based on their distributions",
    "Monitor outliers in electronics category - may indicate data quality issues or premium products",
    "Seasonal patterns visible in monthly trends - adjust inventory accordingly"
]

for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")
```

## Summary and Best Practices

---

### 🎯 Distribution Mastery Summary

```python
print("📚 DATA DISTRIBUTIONS MASTERY SUMMARY")
print("=" * 60)

print("✅ NORMAL DISTRIBUTION:")
print("  • Gold standard for many ML algorithms")
print("  • Symmetric, bell-shaped curve")
print("  • Defined by mean (μ) and standard deviation (σ)")
print("  • Empirical rule: 68-95-99.7% within 1-2-3 standard deviations")
print("  • Many statistical tests assume normality")
print("  • Key insight: Check normality before choosing algorithms!")

print("\n✅ UNIFORM DISTRIBUTION:")
print("  • Equal probability across all values in range")
print("  • Flat, rectangular shape")
print("  • Common in simulation and random sampling")
print("  • Zero skewness and negative kurtosis")
print("  • Key insight: Rare in real-world data but important for modeling!")

print("\n✅ SKEWED DISTRIBUTIONS:")
print("  • Right-skewed: Long tail to the right (mean > median)")
print("  • Left-skewed: Long tail to the left (mean < median)")
print("  • Common in real-world data (income, prices, etc.)")
print("  • Transformations can help: log, square root, Box-Cox")
print("  • Key insight: Skewness affects algorithm performance!")

print("\n✅ OUTLIER DETECTION:")
print("  • Statistical methods: Z-score, IQR, Modified Z-score")
print("  • Machine learning methods: Isolation Forest, LOF")
print("  • Treatment options: Remove, cap, transform, impute")
print("  • Context matters: Are outliers errors or valuable information?")
print("  • Key insight: Always investigate outliers before removing!")

print("\n🎯 DISTRIBUTION SELECTION GUIDE FOR ML:")
guide = {
    'Linear Regression': 'Assumes normal distribution of residuals',
    'Logistic Regression': 'Robust to distribution assumptions',
    'Decision Trees': 'Distribution-free (non-parametric)',
    'Random Forest': 'Distribution-free, handles skewness well',
    'SVM': 'Benefits from normalized/standardized data',
    'Neural Networks': 'Often require standardization',
    'Naive Bayes': 'Assumes specific distributions by feature',
    'K-Means': 'Works best with spherical (normal-like) clusters'
}

print("\nAlgorithm Distribution Requirements:")
for algorithm, requirement in guide.items():
    print(f"  • {algorithm}: {requirement}")

print("\n⚠️ COMMON DISTRIBUTION MISTAKES:")
mistakes = [
    "Assuming data is normal without testing",
    "Ignoring skewness in features before modeling",
    "Removing all outliers without investigation",
    "Using parametric tests on non-normal data",
    "Not considering data transformations",
    "Overlooking the impact of sample size on distribution shape",
    "Forgetting to check distribution assumptions in algorithms",
    "Using mean instead of median for highly skewed data"
]

for i, mistake in enumerate(mistakes, 1):
    print(f"  {i}. {mistake}")

print("\n🔧 PRACTICAL WORKFLOW:")
workflow_steps = [
    "Visualize distributions with histograms and Q-Q plots",
    "Calculate descriptive statistics (mean, median, skewness, kurtosis)",
    "Test for normality using appropriate statistical tests",
    "Identify and investigate outliers using multiple methods",
    "Consider transformations for skewed data",
    "Choose ML algorithms appropriate for your data distribution",
    "Validate assumptions throughout your analysis",
    "Document distribution characteristics for reproducibility"
]

for i, step in enumerate(workflow_steps, 1):
    print(f"  {i}. {step}")

# Quick reference functions
print("\n🛠️ QUICK REFERENCE TOOLKIT:")
```

### Practical Distribution Analysis Toolkit

```python
def distribution_analysis_toolkit():
    """Complete toolkit for distribution analysis"""
    
    def quick_distribution_check(data, name="Data"):
        """Quick distribution analysis"""
        print(f"📊 Quick Distribution Check: {name}")
        print("-" * 40)
        
        # Basic statistics
        stats_dict = {
            'Count': len(data),
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Mode': stats.mode(data, keepdims=True).mode[0],
            'Std Dev': np.std(data, ddof=1),
            'Variance': np.var(data, ddof=1),
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data),
            'Range': np.max(data) - np.min(data),
            'IQR': np.percentile(data, 75) - np.percentile(data, 25)
        }
        
        for key, value in stats_dict.items():
            if key == 'Count':
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value:.3f}")
        
        # Distribution classification
        skew_val = stats.skew(data)
        if abs(skew_val) < 0.5:
            dist_type = "Approximately Symmetric"
        elif skew_val > 0.5:
            dist_type = "Right-Skewed"
        else:
            dist_type = "Left-Skewed"
        
        print(f"\n  Distribution Type: {dist_type}")
        
        # Normality assessment
        if len(data) <= 5000:
            _, p_val = stats.shapiro(data)
            print(f"  Normality Test (Shapiro): {'Normal' if p_val > 0.05 else 'Not Normal'} (p={p_val:.4f})")
        
        # Outlier count
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        print(f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
        
        return stats_dict
    
    def recommend_transformation(data):
        """Recommend appropriate transformation"""
        skew_val = stats.skew(data)
        
        print(f"🔧 Transformation Recommendations:")
        print(f"  Current skewness: {skew_val:.3f}")
        
        if abs(skew_val) < 0.5:
            print("  ✅ No transformation needed - data is approximately symmetric")
        elif skew_val > 1:
            print("  📈 Highly right-skewed - consider:")
            print("    • Log transformation (if all values > 0)")
            print("    • Square root transformation")
            print("    • Box-Cox transformation")
        elif skew_val > 0.5:
            print("  📈 Moderately right-skewed - consider:")
            print("    • Square root transformation")
            print("    • Box-Cox transformation")
        elif skew_val < -1:
            print("  📉 Highly left-skewed - consider:")
            print("    • Reflect and transform")
            print("    • Yeo-Johnson transformation")
        else:
            print("  📉 Moderately left-skewed - consider:")
            print("    • Yeo-Johnson transformation")
    
    def ml_algorithm_recommendations(data):
        """Recommend ML algorithms based on distribution"""
        skew_val = abs(stats.skew(data))
        
        print(f"🤖 ML Algorithm Recommendations:")
        
        if skew_val < 0.5:
            print("  ✅ Distribution-friendly algorithms:")
            print("    • Linear/Logistic Regression")
            print("    • Neural Networks")
            print("    • SVM")
            print("    • Gaussian Naive Bayes")
        else:
            print("  ⚠️  Consider robust algorithms:")
            print("    • Decision Trees")
            print("    • Random Forest")
            print("    • Gradient Boosting")
            print("    • Non-parametric methods")
        
        print("  📊 Always consider:")
        print("    • Data transformation before parametric methods")
        print("    • Cross-validation for algorithm selection")
        print("    • Feature scaling/normalization")
    
    return quick_distribution_check, recommend_transformation, ml_algorithm_recommendations

# Make functions available
quick_check, recommend_transform, ml_recommend = distribution_analysis_toolkit()

print("🎉 DISTRIBUTION ANALYSIS TOOLKIT READY!")
print("\nAvailable functions:")
print("  • quick_check(data, name) - Comprehensive distribution summary")
print("  • recommend_transform(data) - Transformation suggestions")
print("  • ml_recommend(data) - Algorithm recommendations based on distribution")

# Example usage with our e-commerce data
print("\n" + "="*60)
print("EXAMPLE: Analyzing Electronics Category")
print("="*60)

electronics_prices = ecommerce_sales[ecommerce_sales['category'] == 'Electronics']['price']
stats_summary = quick_check(electronics_prices, "Electronics Prices")
recommend_transform(electronics_prices)
ml_recommend(electronics_prices)
```

### 🔜 Next Steps: Ready for Advanced Analytics

```python
print("\n🚀 CONGRATULATIONS! DISTRIBUTION MASTERY ACHIEVED!")
print("=" * 60)

print("🎓 Skills You've Mastered:")
skills = [
    "✅ Understanding normal, uniform, and skewed distributions",
    "✅ Statistical testing for distribution properties",
    "✅ Comprehensive outlier detection and treatment",
    "✅ Data transformation techniques for distribution improvement", 
    "✅ Choosing appropriate ML algorithms based on data distribution",
    "✅ Real-world distribution analysis workflows",
    "✅ Business insight generation from distribution patterns"
]

for skill in skills:
    print(f"  {skill}")

print(f"\n🎯 You're Now Ready For:")
next_steps = [
    "Advanced Feature Engineering techniques",
    "Machine Learning Algorithm Selection and Tuning",
    "Statistical Modeling and Inference",
    "Advanced Anomaly Detection methods",
    "Time Series Analysis and Forecasting",
    "Multivariate Statistical Analysis",
    "Deep Learning with proper data preprocessing"
]

for i, step in enumerate(next_steps, 1):
    print(f"  {i}. {step}")

print(f"\n💡 Key Takeaway:")
print("Understanding your data's distribution is the foundation of successful")
print("machine learning. Every algorithm has assumptions about data distribution,")
print("and knowing these patterns helps you choose the right approach and")
print("transformations to achieve optimal model performance!")

print(f"\n🏆 Well done! You now have a solid foundation in data distributions")
print("and are ready to tackle more advanced machine learning challenges!")
```

**Fundamental Skills Achieved:**

- ✅ Normal distribution analysis and empirical rule application
- ✅ Uniform distribution understanding and applications
- ✅ Skewed distribution identification and transformation techniques
- ✅ Comprehensive outlier detection using multiple methods
- ✅ Distribution-based algorithm selection for machine learning
- ✅ Real-world business case analysis using distribution insights
- ✅ Complete data distribution analysis workflows

**Ready for advanced machine learning topics!** 🚀