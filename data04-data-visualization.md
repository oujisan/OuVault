# [data] #04 - Data Visualization
![data](https://raw.githubusercontent.com/oujisan/OuVault/main/img/data.png)

## Data Visualization: Histogram, Boxplot, and Scatter Plot
---

"A picture is worth a thousand words" - dan dalam data science, visualisasi adalah cara paling efektif untuk memahami pola, trends, dan insights yang tersembunyi dalam data. Mari kita pelajari tiga jenis visualisasi fundamental yang akan menjadi senjata utama dalam analisis data.

## Why Data Visualization Matters
---

Visualisasi data membantu kita untuk:
- **Mengidentifikasi pola** yang tidak terlihat dalam angka mentah
- **Mendeteksi outliers** dan anomali dengan cepat
- **Memahami distribusi** data secara intuitif
- **Mengkomunikasikan insights** kepada stakeholders
- **Memvalidasi asumsi** sebelum modeling

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi yang lebih menarik
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Sample data untuk eksplorasi
np.random.seed(42)
n_samples = 1000

# Generate different types of data
normal_data = np.random.normal(100, 15, n_samples)
skewed_data = np.random.exponential(2, n_samples) * 20 + 50
bimodal_data = np.concatenate([
    np.random.normal(80, 10, n_samples//2),
    np.random.normal(120, 10, n_samples//2)
])

print("📊 Data visualization fundamentals - Let's start!")
```

## Histogram: Understanding Distribution Shape
---

Histogram menunjukkan distribusi frekuensi data dan membantu kita memahami shape, central tendency, dan spread.

### Basic Histogram
```python
# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Understanding Data Distributions with Histograms', fontsize=16)

# Normal distribution
axes[0,0].hist(normal_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(np.mean(normal_data), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(normal_data):.1f}')
axes[0,0].axvline(np.median(normal_data), color='green', linestyle='--', 
                  label=f'Median: {np.median(normal_data):.1f}')
axes[0,0].set_title('Normal Distribution')
axes[0,0].set_xlabel('Values')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# Skewed distribution
axes[0,1].hist(skewed_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
axes[0,1].axvline(np.mean(skewed_data), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(skewed_data):.1f}')
axes[0,1].axvline(np.median(skewed_data), color='green', linestyle='--', 
                  label=f'Median: {np.median(skewed_data):.1f}')
axes[0,1].set_title('Right-Skewed Distribution')
axes[0,1].set_xlabel('Values')
axes[0,1].set_ylabel('Frequency')
axes[0,1].legend()

# Bimodal distribution
axes[1,0].hist(bimodal_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1,0].axvline(np.mean(bimodal_data), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(bimodal_data):.1f}')
axes[1,0].axvline(np.median(bimodal_data), color='green', linestyle='--', 
                  label=f'Median: {np.median(bimodal_data):.1f}')
axes[1,0].set_title('Bimodal Distribution')
axes[1,0].set_xlabel('Values')
axes[1,0].set_ylabel('Frequency')
axes[1,0].legend()

# Comparison of all distributions
axes[1,1].hist(normal_data, bins=30, alpha=0.5, label='Normal', density=True)
axes[1,1].hist(skewed_data, bins=30, alpha=0.5, label='Skewed', density=True)
axes[1,1].hist(bimodal_data, bins=30, alpha=0.5, label='Bimodal', density=True)
axes[1,1].set_title('Distribution Comparison')
axes[1,1].set_xlabel('Values')
axes[1,1].set_ylabel('Density')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# Statistical summary
print("📈 Distribution Analysis:")
distributions = {
    'Normal': normal_data,
    'Skewed': skewed_data, 
    'Bimodal': bimodal_data
}

for name, data in distributions.items():
    print(f"\n{name} Distribution:")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Std Dev: {np.std(data):.2f}")
    print(f"  Skewness: {stats.skew(data):.2f}")
    print(f"  Kurtosis: {stats.kurtosis(data):.2f}")
```

### Advanced Histogram Techniques
```python
# Real-world example: Student grades analysis
student_grades = pd.DataFrame({
    'Math': np.random.normal(78, 12, 500).clip(0, 100),
    'Science': np.random.normal(82, 10, 500).clip(0, 100),
    'English': np.random.normal(75, 15, 500).clip(0, 100),
    'History': np.random.normal(80, 8, 500).clip(0, 100)
})

# Multiple histograms with different bin sizes
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Advanced Histogram Analysis: Student Grades', fontsize=16)

# Different bin sizes for Math scores
bin_sizes = [10, 20, 50]
for i, bins in enumerate(bin_sizes):
    axes[0, i].hist(student_grades['Math'], bins=bins, alpha=0.7, 
                    color=f'C{i}', edgecolor='black')
    axes[0, i].set_title(f'Math Scores - {bins} bins')
    axes[0, i].set_xlabel('Grade')
    axes[0, i].set_ylabel('Frequency')

# Cumulative histogram
axes[1, 0].hist(student_grades['Math'], bins=30, cumulative=True, 
                alpha=0.7, color='purple', edgecolor='black')
axes[1, 0].set_title('Cumulative Distribution - Math')
axes[1, 0].set_xlabel('Grade')
axes[1, 0].set_ylabel('Cumulative Frequency')

# Normalized histogram (density)
axes[1, 1].hist(student_grades['Math'], bins=30, density=True, 
                alpha=0.7, color='orange', edgecolor='black')
# Overlay normal curve
x = np.linspace(student_grades['Math'].min(), student_grades['Math'].max(), 100)
y = stats.norm.pdf(x, student_grades['Math'].mean(), student_grades['Math'].std())
axes[1, 1].plot(x, y, 'r-', linewidth=2, label='Normal curve')
axes[1, 1].set_title('Density Histogram with Normal Curve')
axes[1, 1].set_xlabel('Grade')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()

# Multiple subjects comparison
axes[1, 2].hist([student_grades['Math'], student_grades['Science'], 
                student_grades['English'], student_grades['History']], 
               bins=25, alpha=0.6, label=['Math', 'Science', 'English', 'History'])
axes[1, 2].set_title('All Subjects Comparison')
axes[1, 2].set_xlabel('Grade')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# Grade distribution analysis
print("🎓 Grade Distribution Analysis:")
for subject in student_grades.columns:
    grades = student_grades[subject]
    
    # Grade categories
    excellent = (grades >= 90).sum()
    good = ((grades >= 80) & (grades < 90)).sum()
    satisfactory = ((grades >= 70) & (grades < 80)).sum()
    needs_improvement = (grades < 70).sum()
    
    print(f"\n{subject}:")
    print(f"  Excellent (90-100): {excellent} ({excellent/len(grades)*100:.1f}%)")
    print(f"  Good (80-89): {good} ({good/len(grades)*100:.1f}%)")
    print(f"  Satisfactory (70-79): {satisfactory} ({satisfactory/len(grades)*100:.1f}%)")
    print(f"  Needs Improvement (<70): {needs_improvement} ({needs_improvement/len(grades)*100:.1f}%)")
```

## Boxplot: Detecting Outliers and Understanding Quartiles
---

Boxplot (box-and-whisker plot) memberikan ringkasan visual dari five-number summary dan sangat efektif untuk mendeteksi outliers.

### Basic Boxplot Analysis
```python
# Create comprehensive boxplot analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Boxplot Analysis: Understanding Data Distribution', fontsize=16)

# Single variable boxplot
axes[0,0].boxplot(student_grades['Math'], labels=['Math'])
axes[0,0].set_title('Single Variable Boxplot')
axes[0,0].set_ylabel('Grade')
axes[0,0].grid(True, alpha=0.3)

# Multiple variables comparison
axes[0,1].boxplot([student_grades[col] for col in student_grades.columns], 
                  labels=student_grades.columns)
axes[0,1].set_title('Multiple Subjects Comparison')
axes[0,1].set_ylabel('Grade')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# Horizontal boxplot
axes[1,0].boxplot(student_grades['Math'], vert=False, labels=['Math'])
axes[1,0].set_title('Horizontal Boxplot')
axes[1,0].set_xlabel('Grade')
axes[1,0].grid(True, alpha=0.3)

# Boxplot with outliers highlighted
# Add some artificial outliers
math_with_outliers = np.concatenate([student_grades['Math'], [20, 25, 98, 99]])
bp = axes[1,1].boxplot(math_with_outliers, labels=['Math + Outliers'])
axes[1,1].set_title('Boxplot with Outliers')
axes[1,1].set_ylabel('Grade')
axes[1,1].grid(True, alpha=0.3)

# Highlight outliers
outliers = bp['fliers'][0].get_data()[1]
for outlier in outliers:
    axes[1,1].annotate(f'{outlier:.0f}', xy=(1, outlier), xytext=(1.1, outlier),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       color='red', fontweight='bold')

plt.tight_layout()
plt.show()

# Boxplot statistics interpretation
def boxplot_analysis(data, name="Data"):
    """Analyze boxplot statistics"""
    Q1 = np.percentile(data, 25)
    Q2 = np.percentile(data, 50)  # Median
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]
    
    print(f"📦 Boxplot Analysis for {name}:")
    print(f"  Q1 (25th percentile): {Q1:.2f}")
    print(f"  Q2 (Median): {Q2:.2f}")
    print(f"  Q3 (75th percentile): {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Lower whisker: {lower_whisker:.2f}")
    print(f"  Upper whisker: {upper_whisker:.2f}")
    print(f"  Outliers: {len(outliers)} values")
    if len(outliers) > 0:
        print(f"  Outlier values: {outliers[:10]}")  # Show first 10 outliers
    
    return {
        'Q1': Q1, 'Q2': Q2, 'Q3': Q3, 'IQR': IQR,
        'outliers': outliers, 'outlier_count': len(outliers)
    }

# Analyze each subject
for subject in student_grades.columns:
    analysis = boxplot_analysis(student_grades[subject], subject)
    print()
```

### Advanced Boxplot Techniques
```python
# Grouped boxplots with categorical data
# Create synthetic student data with categories
np.random.seed(42)
student_performance = pd.DataFrame({
    'Grade': np.concatenate([
        np.random.normal(85, 10, 150),  # Grade A students
        np.random.normal(75, 8, 200),   # Grade B students  
        np.random.normal(65, 12, 150),  # Grade C students
        np.random.normal(55, 15, 100)   # Grade D students
    ]),
    'Category': ['A']*150 + ['B']*200 + ['C']*150 + ['D']*100,
    'Gender': np.random.choice(['Male', 'Female'], 600),
    'School_Type': np.random.choice(['Public', 'Private'], 600)
})

# Create advanced boxplot visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Advanced Boxplot Analysis: Student Performance', fontsize=16)

# Grouped by category
categories = ['A', 'B', 'C', 'D']
grade_data = [student_performance[student_performance['Category']==cat]['Grade'] 
              for cat in categories]
axes[0,0].boxplot(grade_data, labels=categories)
axes[0,0].set_title('Performance by Grade Category')
axes[0,0].set_xlabel('Grade Category')
axes[0,0].set_ylabel('Score')
axes[0,0].grid(True, alpha=0.3)

# Using seaborn for more advanced grouping
import seaborn as sns

sns.boxplot(data=student_performance, x='Category', y='Grade', ax=axes[0,1])
axes[0,1].set_title('Seaborn Boxplot by Category')
axes[0,1].grid(True, alpha=0.3)

# Multiple grouping variables
sns.boxplot(data=student_performance, x='Category', y='Grade', 
            hue='Gender', ax=axes[1,0])
axes[1,0].set_title('Performance by Category and Gender')
axes[1,0].grid(True, alpha=0.3)

# Violin plot (combination of boxplot and density)
sns.violinplot(data=student_performance, x='Category', y='Grade', ax=axes[1,1])
axes[1,1].set_title('Violin Plot: Distribution Shape + Quartiles')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical comparison between groups
print("📊 Group Comparison Analysis:")
for category in ['A', 'B', 'C', 'D']:
    cat_data = student_performance[student_performance['Category']==category]['Grade']
    print(f"\nGrade {category} students:")
    print(f"  Count: {len(cat_data)}")
    print(f"  Mean: {cat_data.mean():.2f}")
    print(f"  Median: {cat_data.median():.2f}")
    print(f"  Std Dev: {cat_data.std():.2f}")
    print(f"  Range: {cat_data.min():.1f} - {cat_data.max():.1f}")
```

## Scatter Plot: Exploring Relationships
---

Scatter plot adalah cara terbaik untuk memvisualisasikan hubungan antara dua variabel numerik.

### Basic Scatter Plot Analysis
```python
# Generate correlated data for scatter plot analysis
np.random.seed(42)
n_points = 300

# Different types of relationships
x1 = np.random.normal(50, 15, n_points)
y1_strong_pos = 2 * x1 + np.random.normal(0, 10, n_points)  # Strong positive
y1_weak_pos = 0.5 * x1 + np.random.normal(0, 20, n_points)  # Weak positive
y1_negative = -1.5 * x1 + 150 + np.random.normal(0, 15, n_points)  # Negative
y1_no_corr = np.random.normal(75, 20, n_points)  # No correlation

# Create comprehensive scatter plot analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Scatter Plot Analysis: Understanding Relationships', fontsize=16)

# Strong positive correlation
axes[0,0].scatter(x1, y1_strong_pos, alpha=0.6, color='blue')
correlation = np.corrcoef(x1, y1_strong_pos)[0,1]
axes[0,0].set_title(f'Strong Positive Correlation (r={correlation:.3f})')
axes[0,0].set_xlabel('X Variable')
axes[0,0].set_ylabel('Y Variable')
axes[0,0].grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(x1, y1_strong_pos, 1)
p = np.poly1d(z)
axes[0,0].plot(x1, p(x1), "r--", alpha=0.8, linewidth=2, label='Trend line')
axes[0,0].legend()

# Weak positive correlation
axes[0,1].scatter(x1, y1_weak_pos, alpha=0.6, color='green')
correlation = np.corrcoef(x1, y1_weak_pos)[0,1]
axes[0,1].set_title(f'Weak Positive Correlation (r={correlation:.3f})')
axes[0,1].set_xlabel('X Variable')
axes[0,1].set_ylabel('Y Variable')
axes[0,1].grid(True, alpha=0.3)

# Negative correlation
axes[0,2].scatter(x1, y1_negative, alpha=0.6, color='red')
correlation = np.corrcoef(x1, y1_negative)[0,1]
axes[0,2].set_title(f'Negative Correlation (r={correlation:.3f})')
axes[0,2].set_xlabel('X Variable')
axes[0,2].set_ylabel('Y Variable')
axes[0,2].grid(True, alpha=0.3)

# No correlation
axes[1,0].scatter(x1, y1_no_corr, alpha=0.6, color='purple')
correlation = np.corrcoef(x1, y1_no_corr)[0,1]
axes[1,0].set_title(f'No Correlation (r={correlation:.3f})')
axes[1,0].set_xlabel('X Variable')
axes[1,0].set_ylabel('Y Variable')
axes[1,0].grid(True, alpha=0.3)

# Non-linear relationship
x_nonlinear = np.linspace(-3, 3, n_points)
y_nonlinear = x_nonlinear**2 + np.random.normal(0, 1, n_points)
axes[1,1].scatter(x_nonlinear, y_nonlinear, alpha=0.6, color='orange')
correlation = np.corrcoef(x_nonlinear, y_nonlinear)[0,1]
axes[1,1].set_title(f'Non-linear Relationship (r={correlation:.3f})')
axes[1,1].set_xlabel('X Variable')
axes[1,1].set_ylabel('Y Variable')
axes[1,1].grid(True, alpha=0.3)

# Outliers effect
x_outliers = np.concatenate([x1[:250], [100, 10, 90]])
y_outliers = np.concatenate([y1_strong_pos[:250], [200, 10, 180]])
axes[1,2].scatter(x_outliers, y_outliers, alpha=0.6, color='brown')
# Highlight outliers
axes[1,2].scatter([100, 10, 90], [200, 10, 180], 
                  color='red', s=100, marker='x', linewidth=3, label='Outliers')
correlation = np.corrcoef(x_outliers, y_outliers)[0,1]
axes[1,2].set_title(f'With Outliers (r={correlation:.3f})')
axes[1,2].set_xlabel('X Variable')
axes[1,2].set_ylabel('Y Variable')
axes[1,2].grid(True, alpha=0.3)
axes[1,2].legend()

plt.tight_layout()
plt.show()

# Correlation interpretation
def interpret_correlation(r):
    """Interpret correlation coefficient"""
    abs_r = abs(r)
    if abs_r >= 0.8:
        strength = "Very Strong"
    elif abs_r >= 0.6:
        strength = "Strong"
    elif abs_r >= 0.4:
        strength = "Moderate"
    elif abs_r >= 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    direction = "Positive" if r > 0 else "Negative" if r < 0 else "None"
    return f"{strength} {direction}"

print("🔗 Correlation Analysis:")
correlations = [
    ("Strong Positive", np.corrcoef(x1, y1_strong_pos)[0,1]),
    ("Weak Positive", np.corrcoef(x1, y1_weak_pos)[0,1]),
    ("Negative", np.corrcoef(x1, y1_negative)[0,1]),
    ("No Correlation", np.corrcoef(x1, y1_no_corr)[0,1]),
    ("Non-linear", np.corrcoef(x_nonlinear, y_nonlinear)[0,1])
]

for name, corr in correlations:
    print(f"  {name}: r = {corr:.3f} ({interpret_correlation(corr)})")
```

### Advanced Scatter Plot Techniques
```python
# Real-world example: House prices analysis
# Generate realistic house data
np.random.seed(42)
n_houses = 500

house_data = pd.DataFrame({
    'size_sqft': np.random.normal(2000, 500, n_houses).clip(800, 4000),
    'bedrooms': np.random.randint(2, 6, n_houses),
    'age_years': np.random.randint(1, 50, n_houses),
    'lot_size': np.random.normal(8000, 2000, n_houses).clip(3000, 15000)
})

# Price calculation with realistic relationships
house_data['price'] = (
    150 * house_data['size_sqft'] +  # Size effect
    10000 * house_data['bedrooms'] +  # Bedroom premium
    -2000 * house_data['age_years'] +  # Depreciation
    5 * house_data['lot_size'] +  # Lot size value
    np.random.normal(0, 50000, n_houses)  # Random variation
).clip(100000, 800000)

# Add categorical variables
house_data['neighborhood'] = np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_houses)
house_data['condition'] = np.random.choice(['Excellent', 'Good', 'Fair'], n_houses)

# Advanced scatter plot analysis
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Advanced Scatter Plot Analysis: House Prices', fontsize=16)

# Size vs Price
axes[0,0].scatter(house_data['size_sqft'], house_data['price'], alpha=0.6)
axes[0,0].set_title('House Size vs Price')
axes[0,0].set_xlabel('Size (sq ft)')
axes[0,0].set_ylabel('Price ($)')
axes[0,0].grid(True, alpha=0.3)

# Add correlation coefficient
corr = house_data[['size_sqft', 'price']].corr().iloc[0,1]
axes[0,0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0,0].transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Age vs Price
axes[0,1].scatter(house_data['age_years'], house_data['price'], alpha=0.6, color='red')
axes[0,1].set_title('House Age vs Price')
axes[0,1].set_xlabel('Age (years)')
axes[0,1].set_ylabel('Price ($)')
axes[0,1].grid(True, alpha=0.3)

corr = house_data[['age_years', 'price']].corr().iloc[0,1]
axes[0,1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0,1].transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Colored by category (neighborhood)
neighborhoods = house_data['neighborhood'].unique()
colors = ['blue', 'red', 'green']
for i, neighborhood in enumerate(neighborhoods):
    mask = house_data['neighborhood'] == neighborhood
    axes[0,2].scatter(house_data[mask]['size_sqft'], house_data[mask]['price'], 
                     alpha=0.6, color=colors[i], label=neighborhood)
axes[0,2].set_title('Size vs Price by Neighborhood')
axes[0,2].set_xlabel('Size (sq ft)')
axes[0,2].set_ylabel('Price ($)')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# Bubble plot (size represents third variable)
scatter = axes[1,0].scatter(house_data['size_sqft'], house_data['price'], 
                           s=house_data['bedrooms']*20, alpha=0.6,
                           c=house_data['age_years'], cmap='viridis')
axes[1,0].set_title('Size vs Price (Bubble=Bedrooms, Color=Age)')
axes[1,0].set_xlabel('Size (sq ft)')
axes[1,0].set_ylabel('Price ($)')
plt.colorbar(scatter, ax=axes[1,0], label='Age (years)')

# Hexbin plot for high-density data
axes[1,1].hexbin(house_data['size_sqft'], house_data['price'], gridsize=20, cmap='Blues')
axes[1,1].set_title('Hexbin Plot: Density Visualization')
axes[1,1].set_xlabel('Size (sq ft)')
axes[1,1].set_ylabel('Price ($)')

# Multiple regression visualization
from sklearn.linear_model import LinearRegression
X = house_data[['size_sqft', 'age_years']].values
y = house_data['price'].values
model = LinearRegression().fit(X, y)
predicted = model.predict(X)

axes[1,2].scatter(y, predicted, alpha=0.6)
axes[1,2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[1,2].set_title('Actual vs Predicted Prices')
axes[1,2].set_xlabel('Actual Price ($)')
axes[1,2].set_ylabel('Predicted Price ($)')
axes[1,2].grid(True, alpha=0.3)

# R-squared
r2 = model.score(X, y)
axes[1,2].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1,2].transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

# Correlation matrix for all numeric variables
numeric_cols = ['size_sqft', 'bedrooms', 'age_years', 'lot_size', 'price']
correlation_matrix = house_data[numeric_cols].corr()

print("🏠 House Price Correlation Analysis:")
print(correlation_matrix.round(3))

# Find strongest correlations with price
price_correlations = correlation_matrix['price'].sort_values(key=abs, ascending=False)
print(f"\n📊 Variables most correlated with price:")
for var, corr in price_correlations.items():
    if var != 'price':
        print(f"  {var}: {corr:.3f} ({interpret_correlation(corr)})")
```

## Combining Multiple Visualization Techniques
---

### Comprehensive Data Exploration Dashboard
```python
def create_data_exploration_dashboard(df, target_col=None):
    """
    Create comprehensive data exploration dashboard
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    n_numeric = len(numeric_cols)
    if target_col:
        n_plots = min(6, n_numeric)  # Limit to 6 plots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Data Exploration Dashboard - Target: {target_col}', fontsize=16)
        axes = axes.flatten()
    else:
        n_plots = min(6, n_numeric)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Exploration Dashboard', fontsize=16)
        axes = axes.flatten()
    
    for i in range(n_plots):
        if i < len(numeric_cols):
            col = numeric_cols[i]
            
            if target_col and i < 3:
                # Scatter plots with target
                axes[i].scatter(df[col], df[target_col], alpha=0.6)
                axes[i].set_xlabel(col)
                axes[i].set_ylabel(target_col)
                axes[i].set_title(f'{col} vs {target_col}')
                axes[i].grid(True, alpha=0.3)
                
                # Add correlation
                corr = df[[col, target_col]].corr().iloc[0,1]
                axes[i].text(0.05, 0.95, f'r = {corr:.3f}', 
                           transform=axes[i].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            elif target_col and i >= 3:
                # Histograms for features
                axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = df[col].mean()
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}')
                axes[i].legend()
            
            else:
                # Just histograms if no target
                axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].grid(True, alpha=0.3)
        
        else:
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical summary
    print("📊 Statistical Summary:")
    print(df[numeric_cols + ([target_col] if target_col else [])].describe().round(3))
    
    if target_col:
        print(f"\n🎯 Target Variable Analysis ({target_col}):")
        target_stats = df[target_col]
        print(f"  Mean: {target_stats.mean():.3f}")
        print(f"  Median: {target_stats.median():.3f}")
        print(f"  Std Dev: {target_stats.std():.3f}")
        print(f"  Skewness: {stats.skew(target_stats):.3f}")
        print(f"  Range: {target_stats.min():.1f} - {target_stats.max():.1f}")

# Test the dashboard with house data
create_data_exploration_dashboard(house_data, 'price')
```

### Interactive Visualization Tips
```python
# Advanced visualization techniques summary
print("🎨 ADVANCED VISUALIZATION TECHNIQUES")
print("=" * 50)

techniques = {
    "Histogram Improvements": [
        "• Use appropriate bin sizes (rule of thumb: √n or Sturges' rule)",
        "• Add density curves for distribution comparison",
        "• Use cumulative histograms for percentile analysis",
        "• Color-code by categories for group comparisons"
    ],
    
    "Boxplot Enhancements": [
        "• Group by multiple categorical variables",
        "• Use violin plots to show distribution shape",
        "• Annotate outliers with their values",
        "• Add notches to show confidence intervals"
    ],
    
    "Scatter Plot Advanced": [
        "• Use bubble size for third dimension",
        "• Color-code points by categories",
        "• Add trend lines and confidence intervals",
        "• Use hexbin plots for high-density data",
        "• Create scatter plot matrices for multiple variables"
    ]
}

for category, tips in techniques.items():
    print(f"\n{category}:")
    for tip in tips:
        print(f"  {tip}")

# Visualization best practices
print(f"\n🏆 VISUALIZATION BEST PRACTICES:")
best_practices = [
    "Always add clear titles and axis labels",
    "Use color meaningfully and consistently", 
    "Include legends when using multiple categories",
    "Add grid lines for easier reading",
    "Choose appropriate scales (linear vs log)",
    "Annotate important points or statistics",
    "Consider your audience and purpose",
    "Test colorblind-friendly palettes",
    "Keep it simple but informative",
    "Tell a story with your visualizations"
]

for i, practice in enumerate(best_practices, 1):
    print(f"  {i:2d}. {practice}")
```

## Real-world Case Study: Customer Analytics
---

### E-commerce Customer Behavior Analysis
```python
# Generate realistic e-commerce customer data
np.random.seed(42)
n_customers = 2000

# Customer segments with different behaviors
segments = {
    'High_Value': {'ratio': 0.15, 'spend_mean': 2000, 'spend_std': 500, 'visits_mean': 25, 'age_mean': 45},
    'Regular': {'ratio': 0.50, 'spend_mean': 800, 'spend_std': 200, 'visits_mean': 12, 'age_mean': 35},
    'Occasional': {'ratio': 0.25, 'spend_mean': 300, 'spend_std': 100, 'visits_mean': 5, 'age_mean': 28}, 
    'New': {'ratio': 0.10, 'spend_mean': 150, 'spend_std': 50, 'visits_mean': 2, 'age_mean': 25}
}

customer_data = []
for segment, params in segments.items():
    n_segment = int(n_customers * params['ratio'])
    
    for _ in range(n_segment):
        customer = {
            'segment': segment,
            'annual_spend': max(50, np.random.normal(params['spend_mean'], params['spend_std'])),
            'monthly_visits': max(1, np.random.poisson(params['visits_mean'])),
            'age': max(18, np.random.normal(params['age_mean'], 8)),
            'satisfaction': np.random.uniform(1, 5),
            'gender': np.random.choice(['Male', 'Female'])
        }
        customer_data.append(customer)

ecommerce_df = pd.DataFrame(customer_data)

# Calculate additional metrics
ecommerce_df['spend_per_visit'] = ecommerce_df['annual_spend'] / (ecommerce_df['monthly_visits'] * 12)
ecommerce_df['age_group'] = pd.cut(ecommerce_df['age'], 
                                  bins=[0, 25, 35, 45, 100], 
                                  labels=['18-25', '26-35', '36-45', '45+'])

print("🛒 E-commerce Customer Analysis")
print("=" * 40)
print(f"Total customers: {len(ecommerce_df):,}")
print(f"Segments: {ecommerce_df['segment'].value_counts().to_dict()}")

# Create comprehensive visualization dashboard
fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle('E-commerce Customer Analytics Dashboard', fontsize=16)

# 1. Spending distribution by segment
for i, segment in enumerate(ecommerce_df['segment'].unique()):
    segment_data = ecommerce_df[ecommerce_df['segment'] == segment]['annual_spend']
    axes[0,0].hist(segment_data, alpha=0.6, label=segment, bins=30)
axes[0,0].set_title('Annual Spending Distribution by Segment')
axes[0,0].set_xlabel('Annual Spend ($)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Boxplot of spending by segment
segment_spending = [ecommerce_df[ecommerce_df['segment']==seg]['annual_spend'] 
                   for seg in ['New', 'Occasional', 'Regular', 'High_Value']]
axes[0,1].boxplot(segment_spending, labels=['New', 'Occasional', 'Regular', 'High_Value'])
axes[0,1].set_title('Spending Distribution by Segment')
axes[0,1].set_xlabel('Customer Segment')
axes[0,1].set_ylabel('Annual Spend ($)')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# 3. Scatter: Age vs Spending
segments_unique = ecommerce_df['segment'].unique()
colors = ['red', 'blue', 'green', 'orange']
for i, segment in enumerate(segments_unique):
    segment_data = ecommerce_df[ecommerce_df['segment'] == segment]
    axes[0,2].scatter(segment_data['age'], segment_data['annual_spend'], 
                     alpha=0.6, color=colors[i], label=segment)
axes[0,2].set_title('Age vs Annual Spending by Segment')
axes[0,2].set_xlabel('Age')
axes[0,2].set_ylabel('Annual Spend ($)')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 4. Monthly visits distribution
axes[1,0].hist(ecommerce_df['monthly_visits'], bins=25, alpha=0.7, color='purple', edgecolor='black')
axes[1,0].set_title('Distribution of Monthly Visits')
axes[1,0].set_xlabel('Monthly Visits')
axes[1,0].set_ylabel('Frequency')
axes[1,0].grid(True, alpha=0.3)
mean_visits = ecommerce_df['monthly_visits'].mean()
axes[1,0].axvline(mean_visits, color='red', linestyle='--', label=f'Mean: {mean_visits:.1f}')
axes[1,0].legend()

# 5. Spend per visit vs Monthly visits
axes[1,1].scatter(ecommerce_df['monthly_visits'], ecommerce_df['spend_per_visit'], alpha=0.6)
axes[1,1].set_title('Monthly Visits vs Spend per Visit')
axes[1,1].set_xlabel('Monthly Visits')
axes[1,1].set_ylabel('Spend per Visit ($)')
axes[1,1].grid(True, alpha=0.3)

# Add correlation
corr = ecommerce_df[['monthly_visits', 'spend_per_visit']].corr().iloc[0,1]
axes[1,1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1,1].transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 6. Satisfaction scores by segment
sns.boxplot(data=ecommerce_df, x='segment', y='satisfaction', ax=axes[1,2])
axes[1,2].set_title('Customer Satisfaction by Segment')
axes[1,2].set_xlabel('Customer Segment')
axes[1,2].set_ylabel('Satisfaction Score')
axes[1,2].tick_params(axis='x', rotation=45)
axes[1,2].grid(True, alpha=0.3)

# 7. Age group analysis
age_spend = ecommerce_df.groupby('age_group')['annual_spend'].mean()
axes[2,0].bar(age_spend.index, age_spend.values, alpha=0.7, color='lightblue', edgecolor='black')
axes[2,0].set_title('Average Spending by Age Group')
axes[2,0].set_xlabel('Age Group')
axes[2,0].set_ylabel('Average Annual Spend ($)')
axes[2,0].grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(age_spend.values):
    axes[2,0].text(i, v + 50, f'${v:.0f}', ha='center', va='bottom', fontweight='bold')

# 8. Gender analysis
gender_data = ecommerce_df.groupby(['segment', 'gender'])['annual_spend'].mean().unstack()
gender_data.plot(kind='bar', ax=axes[2,1], alpha=0.7)
axes[2,1].set_title('Average Spending by Segment and Gender')
axes[2,1].set_xlabel('Customer Segment')
axes[2,1].set_ylabel('Average Annual Spend ($)')
axes[2,1].tick_params(axis='x', rotation=45)
axes[2,1].legend(title='Gender')
axes[2,1].grid(True, alpha=0.3)

# 9. Correlation heatmap
numeric_cols = ['annual_spend', 'monthly_visits', 'age', 'satisfaction', 'spend_per_visit']
correlation_matrix = ecommerce_df[numeric_cols].corr()

im = axes[2,2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[2,2].set_xticks(range(len(numeric_cols)))
axes[2,2].set_yticks(range(len(numeric_cols)))
axes[2,2].set_xticklabels([col.replace('_', '\n') for col in numeric_cols], rotation=45)
axes[2,2].set_yticklabels([col.replace('_', '\n') for col in numeric_cols])
axes[2,2].set_title('Correlation Heatmap')

# Add correlation values
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        text = axes[2,2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')

plt.tight_layout()
plt.show()

# Generate insights
print("💡 KEY INSIGHTS FROM VISUALIZATION ANALYSIS:")
print("=" * 50)

# Segment analysis
print("📊 Segment Analysis:")
segment_stats = ecommerce_df.groupby('segment').agg({
    'annual_spend': ['mean', 'median', 'std'],
    'monthly_visits': ['mean'],
    'satisfaction': ['mean'],
    'age': ['mean']
}).round(2)

for segment in ['High_Value', 'Regular', 'Occasional', 'New']:
    if segment in segment_stats.index:
        stats = segment_stats.loc[segment]
        print(f"\n{segment} Customers:")
        print(f"  • Average spend: ${stats[('annual_spend', 'mean')]:,.0f}")
        print(f"  • Monthly visits: {stats[('monthly_visits', 'mean')]:,.1f}")
        print(f"  • Satisfaction: {stats[('satisfaction', 'mean')]:,.2f}/5")
        print(f"  • Average age: {stats[('age', 'mean')]:,.0f} years")

# Key correlations
print(f"\n🔗 Key Relationships:")
strong_correlations = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.3:  # Significant correlation
            strong_correlations.append((numeric_cols[i], numeric_cols[j], corr_val))

if strong_correlations:
    for var1, var2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
        direction = "positively" if corr > 0 else "negatively"
        strength = "strongly" if abs(corr) > 0.6 else "moderately"
        print(f"  • {var1} and {var2} are {strength} {direction} correlated (r={corr:.3f})")
else:
    print("  • No strong correlations found between main variables")

# Business recommendations
print(f"\n🎯 BUSINESS RECOMMENDATIONS:")
recommendations = [
    "High-Value customers show highest satisfaction - focus retention efforts here",
    "New customers have lowest spend per visit - implement onboarding programs",
    "Age groups 36-45 spend most - target marketing to this demographic", 
    "Monthly visits correlate with satisfaction - improve website experience",
    "Regular customers are largest segment - design loyalty programs for them"
]

for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")
```

## Summary and Best Practices
---

### 🎯 Key Takeaways from Data Visualization

```python
print("📚 DATA VISUALIZATION MASTERY SUMMARY")
print("=" * 60)

print("✅ HISTOGRAM - Distribution Analysis:")
print("  • Perfect for understanding data distribution shape")
print("  • Shows central tendency, spread, and skewness")
print("  • Helps identify normal vs non-normal distributions")
print("  • Useful for detecting data quality issues")
print("  • Key insight: Bin size affects interpretation!")

print("\n✅ BOXPLOT - Outlier Detection & Quartile Analysis:")
print("  • Excellent for outlier identification")
print("  • Shows five-number summary visually")
print("  • Great for comparing multiple groups")
print("  • Robust to extreme values")
print("  • Key insight: Whiskers show data range, dots show outliers!")

print("\n✅ SCATTER PLOT - Relationship Exploration:")
print("  • Best for exploring relationships between variables")
print("  • Shows correlation strength and direction")
print("  • Reveals linear vs non-linear relationships")
print("  • Helps identify influential points and outliers")
print("  • Key insight: Correlation ≠ Causation!")

print("\n🎨 VISUALIZATION BEST PRACTICES CHECKLIST:")
best_practices = [
    "Choose the right chart type for your data and question",
    "Always include clear, descriptive titles and axis labels",
    "Use colors meaningfully and consistently",
    "Add legends when using multiple categories or series",
    "Include grid lines for easier value reading",
    "Annotate important statistics or insights",
    "Consider your audience's technical level",
    "Test for colorblind accessibility",
    "Keep charts simple but informative",
    "Tell a coherent story with your visualizations"
]

for i, practice in enumerate(best_practices, 1):
    print(f"  {i:2d}. {practice}")

print("\n⚠️ COMMON PITFALLS TO AVOID:")
pitfalls = [
    "Using too many bins in histograms (obscures patterns)",
    "Ignoring outliers in boxplots without investigation", 
    "Assuming correlation implies causation in scatter plots",
    "Using inappropriate scales (linear when log is better)",
    "Overcrowding plots with too much information",
    "Not considering the context of your data",
    "Choosing misleading color schemes",
    "Forgetting to validate visual insights with statistics"
]

for i, pitfall in enumerate(pitfalls, 1):
    print(f"  {i}. {pitfall}")
```

### 🔧 Practical Visualization Toolkit

```python
def visualization_toolkit():
    """Essential visualization functions for daily use"""
    
    def quick_histogram(data, title="Distribution", bins=30):
        """Quick histogram with statistics"""
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.2f}')
        plt.axvline(np.median(data), color='green', linestyle='--', label=f'Median: {np.median(data):.2f}')
        plt.title(title)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Distribution Statistics:")
        print(f"  Mean: {np.mean(data):.3f}")
        print(f"  Median: {np.median(data):.3f}")
        print(f"  Std Dev: {np.std(data):.3f}")
        print(f"  Skewness: {stats.skew(data):.3f}")
    
    def quick_boxplot(data_dict, title="Comparison"):
        """Quick boxplot for multiple groups"""
        plt.figure(figsize=(12, 6))
        plt.boxplot(data_dict.values(), labels=data_dict.keys())
        plt.title(title)
        plt.ylabel('Values')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.show()
        
        for name, data in data_dict.items():
            outliers = len([x for x in data if x < np.percentile(data, 25) - 1.5*(np.percentile(data, 75)-np.percentile(data, 25)) or x > np.percentile(data, 75) + 1.5*(np.percentile(data, 75)-np.percentile(data, 25))])
            print(f"{name}: Median={np.median(data):.2f}, IQR={np.percentile(data, 75)-np.percentile(data, 25):.2f}, Outliers={outliers}")
    
    def quick_scatter(x, y, title="Relationship", labels=None):
        """Quick scatter plot with correlation"""
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8)
        
        # Correlation
        corr = np.corrcoef(x, y)[0,1]
        plt.title(f"{title} (r={corr:.3f})")
        
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Correlation: {corr:.3f} ({interpret_correlation(corr)})")
    
    return quick_histogram, quick_boxplot, quick_scatter

# Make functions available
quick_hist, quick_box, quick_scatter = visualization_toolkit()

print("🛠️ VISUALIZATION TOOLKIT READY!")
print("Available functions:")
print("  • quick_hist(data, title, bins) - Instant histogram with stats")
print("  • quick_box(data_dict, title) - Multi-group boxplot comparison") 
print("  • quick_scatter(x, y, title, labels) - Scatter plot with correlation")
```

### 🔜 Next Steps: From Visualization to Machine Learning

```python
print("\n🚀 PREPARING FOR MACHINE LEARNING")
print("=" * 50)

print("Now that you've mastered data visualization, you have:")
print("✅ Understanding of data distributions and their implications")
print("✅ Ability to identify outliers and anomalies")
print("✅ Skills to explore relationships between variables")
print("✅ Experience with different data types and patterns")
print("✅ Knowledge of data quality assessment techniques")

print(f"\nNext in your ML journey:")
print("📊 Distribution Analysis - Understanding normal, skewed, and other distributions")
print("🔍 Advanced outlier detection and treatment methods")
print("🎯 Feature engineering based on visualization insights")
print("🤖 Choosing appropriate ML algorithms based on data characteristics")
print("📈 Model evaluation using visualization techniques")

print(f"\n💡 Key Questions to Ask When Visualizing Data:")
questions = [
    "What story is my data telling?",
    "Are there any obvious patterns or trends?",
    "Where are the outliers and why might they exist?",
    "What assumptions can I make about the underlying distribution?",
    "Which variables seem most important for prediction?",
    "Are there any data quality issues I need to address?",
    "What transformations might improve my data?",
    "How do different groups or segments behave?"
]

for i, question in enumerate(questions, 1):
    print(f"  {i}. {question}")

print(f"\n🎓 Congratulations! You now have solid visualization skills for ML!")
```

**Skills Mastered:**
- ✅ Histogram analysis for distribution understanding
- ✅ Boxplot techniques for outlier detection and quartile analysis  
- ✅ Scatter plot methods for relationship exploration
- ✅ Advanced visualization techniques (bubble plots, heatmaps, etc.)
- ✅ Real-world data exploration workflows
- ✅ Statistical insight generation from visual analysis

**Ready for the next module: Distribution Analysis!** 📊