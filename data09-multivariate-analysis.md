# [data] #09 - Data Multivariate Analysis

![data](https://raw.githubusercontent.com/oujisan/OuVault/main/img/data.png)

## Introduction to Multivariate Analysis

---

Analisis multivariat adalah teknik untuk memahami hubungan antara lebih dari dua variabel secara bersamaan. Bayangkan seperti melihat orkestra - alih-alih mendengarkan satu instrumen, kita mendengarkan harmoni dari semua instrumen yang bermain bersamaan.

Dalam machine learning, kita jarang berhadapan dengan hanya dua variabel. Data real-world biasanya memiliki puluhan atau bahkan ribuan fitur yang berinteraksi satu sama lain. Analisis multivariat membantu kita:

- Memahami struktur data yang kompleks
- Mengidentifikasi pola tersembunyi
- Mengurangi dimensionalitas data
- Memvisualisasikan data high-dimensional
- Mempersiapkan data untuk modeling

Mari kita jelajahi berbagai teknik untuk menganalisis data multivariat!

## Understanding Multivariate Data Structure

---

### Creating Comprehensive Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create realistic multivariate dataset
np.random.seed(42)
n_samples = 1000

# Generate correlated features
mean = [50, 75, 100, 25, 60]
cov = np.array([
    [100,  80,  -20,  40,  10],
    [ 80, 150,  -30,  50,  15], 
    [-20, -30,  200, -10, -25],
    [ 40,  50,  -10,  80,  20],
    [ 10,  15,  -25,  20, 120]
])

# Generate base data
data_numeric = np.random.multivariate_normal(mean, cov, n_samples)

# Create comprehensive dataset
df = pd.DataFrame({
    'age': np.clip(data_numeric[:, 0], 18, 80).astype(int),
    'income': np.clip(data_numeric[:, 1] * 1000, 20000, 200000),
    'education_years': np.clip(data_numeric[:, 2] / 10, 12, 20).astype(int),
    'experience_years': np.clip(data_numeric[:, 3], 0, 40).astype(int),
    'satisfaction_score': np.clip(data_numeric[:, 4] / 12, 1, 10).astype(int),
})

# Add categorical variables
df['department'] = np.random.choice(['IT', 'Finance', 'Marketing', 'Operations', 'HR'], n_samples)
df['city'] = np.random.choice(['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang'], n_samples)
df['gender'] = np.random.choice(['Male', 'Female'], n_samples)

# Add some derived features
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                        labels=['Young', 'Adult', 'Middle', 'Senior'])
df['income_per_education'] = df['income'] / df['education_years']
df['experience_efficiency'] = df['income'] / (df['experience_years'] + 1)

print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"\nData types:")
print(df.dtypes)
print(f"\nFirst few rows:")
print(df.head())
```

### Data Structure Exploration

```python
def explore_multivariate_structure(df):
    """
    Comprehensive exploration of multivariate data structure
    """
    print("MULTIVARIATE DATA STRUCTURE ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numerical variables: {len(numerical_cols)}")
    print(f"Categorical variables: {len(categorical_cols)}")
    
    # Correlation structure
    corr_matrix = df[numerical_cols].corr()
    
    # Find highest correlations
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = upper_tri.stack().abs().sort_values(ascending=False)
    
    print(f"\nHighest correlations:")
    print(high_corr.head(5))
    
    # Variance analysis
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    variances = np.var(scaled_data, axis=0)
    
    print(f"\nVariance of standardized features:")
    var_df = pd.DataFrame({'feature': numerical_cols, 'variance': variances})
    print(var_df.sort_values('variance', ascending=False))
    
    return corr_matrix, var_df

corr_matrix, variance_df = explore_multivariate_structure(df)
```

## Pairplot Analysis

---

Pairplot adalah visualisasi fundamental untuk memahami hubungan antar variabel dalam dataset multivariat.

### Basic Pairplot

```python
def create_comprehensive_pairplot(df, vars_to_plot=None, hue=None):
    """
    Create comprehensive pairplot with customizations
    """
    if vars_to_plot is None:
        # Select numerical variables with reasonable number for visualization
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        vars_to_plot = numerical_cols[:5]  # Limit to 5 for readability
    
    # Create the pairplot
    g = sns.pairplot(df, 
                     vars=vars_to_plot,
                     hue=hue,
                     diag_kind='hist',
                     plot_kws={'alpha': 0.6, 's': 20},
                     diag_kws={'alpha': 0.7})
    
    # Customize the plot
    g.fig.suptitle('Comprehensive Pairplot Analysis', y=1.02)
    
    # Add correlation values to upper triangle
    for i in range(len(vars_to_plot)):
        for j in range(len(vars_to_plot)):
            if i != j:
                # Get the axis
                ax = g.axes[i, j]
                
                # Calculate correlation
                corr = df[vars_to_plot[i]].corr(df[vars_to_plot[j]])
                
                # Add correlation text (only in upper triangle to avoid duplication)
                if i < j:
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                           transform=ax.transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return g

# Create basic pairplot
numerical_vars = ['age', 'income', 'education_years', 'experience_years', 'satisfaction_score']
g = create_comprehensive_pairplot(df, vars_to_plot=numerical_vars)
plt.show()
```

### Advanced Pairplot with Categorical Grouping

```python
# Pairplot with categorical grouping
plt.figure(figsize=(12, 10))
g = sns.pairplot(df, 
                 vars=['age', 'income', 'education_years', 'satisfaction_score'],
                 hue='age_group',
                 palette='Set1',
                 diag_kind='kde',
                 plot_kws={'alpha': 0.6},
                 diag_kws={'alpha': 0.7})

g.fig.suptitle('Pairplot by Age Group', y=1.02)
plt.show()

# Custom pairplot function for specific relationships
def custom_pairplot_analysis(df):
    """
    Create custom pairplot with specific focus areas
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Age vs Income colored by Department
    sns.scatterplot(data=df, x='age', y='income', hue='department', 
                   ax=axes[0, 0], alpha=0.7)
    axes[0, 0].set_title('Age vs Income by Department')
    
    # Education vs Income colored by Experience
    scatter = axes[0, 1].scatter(df['education_years'], df['income'], 
                                c=df['experience_years'], 
                                cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('Education Years')
    axes[0, 1].set_ylabel('Income')
    axes[0, 1].set_title('Education vs Income (colored by Experience)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Experience Years')
    
    # Satisfaction distribution by Gender
    sns.boxplot(data=df, x='gender', y='satisfaction_score', ax=axes[1, 0])
    axes[1, 0].set_title('Satisfaction Score by Gender')
    
    # Income distribution by City
    sns.violinplot(data=df, x='city', y='income', ax=axes[1, 1])
    axes[1, 1].set_title('Income Distribution by City')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

custom_pairplot_analysis(df)
```

## Heatmap Analysis

---

Heatmap adalah cara powerful untuk memvisualisasikan matriks korelasi dan pola dalam data multivariat.

### Advanced Correlation Heatmaps

```python
def create_advanced_heatmaps(df):
    """
    Create multiple types of heatmaps for comprehensive analysis
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Standard correlation heatmap
    corr_matrix = df[numerical_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=axes[0, 0], cbar_kws={'shrink': 0.8})
    axes[0, 0].set_title('Pearson Correlation Matrix')
    
    # 2. Spearman correlation (for non-linear relationships)
    spearman_corr = df[numerical_cols].corr(method='spearman')
    sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=axes[0, 1], cbar_kws={'shrink': 0.8})
    axes[0, 1].set_title('Spearman Correlation Matrix')
    
    # 3. Covariance matrix
    cov_matrix = df[numerical_cols].cov()
    sns.heatmap(cov_matrix, annot=True, cmap='viridis', 
                square=True, ax=axes[1, 0], cbar_kws={'shrink': 0.8})
    axes[1, 0].set_title('Covariance Matrix')
    
    # 4. Standardized data correlation
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), 
                              columns=numerical_cols)
    scaled_corr = scaled_data.corr()
    
    sns.heatmap(scaled_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
    axes[1, 1].set_title('Correlation Matrix (Standardized Data)')
    
    plt.tight_layout()
    plt.show()
    
    return corr_matrix, spearman_corr, cov_matrix

corr_matrix, spearman_corr, cov_matrix = create_advanced_heatmaps(df)
```

### Clustered Heatmap

```python
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def create_clustered_heatmap(df):
    """
    Create heatmap with hierarchical clustering
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr().abs()  # Use absolute correlation
    
    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    # Create clustered heatmap
    g = sns.clustermap(corr_matrix, 
                       method='ward',
                       cmap='RdBu_r', 
                       center=0,
                       annot=True,
                       figsize=(10, 8),
                       cbar_kws={'label': 'Absolute Correlation'})
    
    g.fig.suptitle('Clustered Correlation Heatmap', y=1.02)
    plt.show()
    
    # Print clustering results
    from scipy.cluster.hierarchy import fcluster
    clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
    
    cluster_df = pd.DataFrame({'variable': numerical_cols, 'cluster': clusters})
    print("Variable Clustering Results:")
    for cluster_id in sorted(cluster_df['cluster'].unique()):
        variables = cluster_df[cluster_df['cluster'] == cluster_id]['variable'].tolist()
        print(f"Cluster {cluster_id}: {variables}")
    
    return cluster_df

cluster_results = create_clustered_heatmap(df)
```

### Interactive Heatmap Analysis

```python
def correlation_strength_analysis(corr_matrix):
    """
    Analyze correlation strength patterns
    """
    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlations = upper_tri.stack().abs()
    
    # Categorize correlation strengths
    strength_categories = []
    for corr in correlations:
        if corr >= 0.8:
            strength_categories.append('Very Strong')
        elif corr >= 0.6:
            strength_categories.append('Strong')
        elif corr >= 0.4:
            strength_categories.append('Moderate')
        elif corr >= 0.2:
            strength_categories.append('Weak')
        else:
            strength_categories.append('Very Weak')
    
    # Create summary
    strength_summary = pd.Series(strength_categories).value_counts()
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot of correlation strengths
    strength_summary.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution of Correlation Strengths')
    axes[0].set_ylabel('Number of Variable Pairs')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Histogram of correlation values
    axes[1].hist(correlations, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_title('Distribution of Absolute Correlation Values')
    axes[1].set_xlabel('Absolute Correlation')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    print("Correlation Strength Summary:")
    print(strength_summary)
    
    return strength_summary

strength_analysis = correlation_strength_analysis(corr_matrix.abs())
```

## Dimensionality Reduction Techniques

---

Dimensionality reduction adalah teknik kunci dalam analisis multivariat untuk memahami struktur data high-dimensional.

### Principal Component Analysis (PCA)

```python
def comprehensive_pca_analysis(df):
    """
    Comprehensive PCA analysis with multiple visualizations
    """
    # Prepare data
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Scree plot
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    axes[0, 0].bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
    axes[0, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Variance Explained')
    axes[0, 0].set_title('Scree Plot')
    axes[0, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
    axes[0, 0].legend()
    
    # 2. First two PCs scatter plot
    scatter = axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=df['satisfaction_score'], cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
    axes[0, 1].set_title('PCA: First Two Components')
    plt.colorbar(scatter, ax=axes[0, 1], label='Satisfaction Score')
    
    # 3. Component loadings heatmap
    loadings = pca.components_[:4, :]  # First 4 components
    loadings_df = pd.DataFrame(loadings.T, 
                              columns=[f'PC{i+1}' for i in range(4)],
                              index=numerical_cols)
    
    sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, 
                ax=axes[0, 2], cbar_kws={'shrink': 0.8})
    axes[0, 2].set_title('Component Loadings')
    
    # 4. Biplot
    def biplot(ax, pca_result, loadings, labels):
        # Plot data points
        ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=10)
        
        # Plot loading vectors
        for i, (label, loading) in enumerate(zip(labels, loadings.T)):
            ax.arrow(0, 0, loading[0]*3, loading[1]*3, 
                    head_width=0.1, head_length=0.1, fc='red', ec='red')
            ax.text(loading[0]*3.2, loading[1]*3.2, label, fontsize=10,
                   ha='center', va='center')
        
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
        ax.set_title('PCA Biplot')
        ax.grid(True, alpha=0.3)
    
    biplot(axes[1, 0], pca_result, pca.components_[:2], numerical_cols)
    
    # 5. Contribution of variables to PCs
    contributions = np.abs(pca.components_[:2]) / np.abs(pca.components_[:2]).sum(axis=1)[:, np.newaxis]
    contrib_df = pd.DataFrame(contributions.T, 
                             columns=['PC1', 'PC2'], 
                             index=numerical_cols)
    
    contrib_df.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Variable Contributions to PCs')
    axes[1, 1].set_ylabel('Contribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    
    # 6. 3D visualization (first 3 components)
    ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
    scatter_3d = ax_3d.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                              c=df['age'], cmap='plasma', alpha=0.6)
    ax_3d.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
    ax_3d.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
    ax_3d.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
    ax_3d.set_title('3D PCA Visualization')
    plt.colorbar(scatter_3d, ax=ax_3d, shrink=0.6, label='Age')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("PCA Analysis Summary:")
    print(f"Total components: {len(explained_variance)}")
    print(f"Variance explained by first 2 components: {sum(explained_variance[:2]):.1%}")
    print(f"Components needed for 80% variance: {np.argmax(cumulative_variance >= 0.8) + 1}")
    print(f"Components needed for 95% variance: {np.argmax(cumulative_variance >= 0.95) + 1}")
    
    return pca, pca_result, loadings_df

pca, pca_result, loadings_df = comprehensive_pca_analysis(df)
```

### t-SNE Analysis

```python
def tsne_analysis(df, perplexity_values=[5, 30, 50]):
    """
    t-SNE analysis with different perplexity values
    """
    # Prepare data
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    fig, axes = plt.subplots(1, len(perplexity_values), figsize=(5*len(perplexity_values), 5))
    if len(perplexity_values) == 1:
        axes = [axes]
    
    tsne_results = {}
    
    for i, perplexity in enumerate(perplexity_values):
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                   learning_rate=200, n_iter=1000)
        tsne_result = tsne.fit_transform(scaled_data)
        tsne_results[perplexity] = tsne_result
        
        # Plot results
        scatter = axes[i].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                 c=df['department'].astype('category').cat.codes, 
                                 cmap='tab10', alpha=0.7, s=20)
        axes[i].set_title(f't-SNE (perplexity={perplexity})')
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')
        
        # Add department legend
        departments = df['department'].unique()
        for j, dept in enumerate(departments):
            axes[i].scatter([], [], c=plt.cm.tab10(j), label=dept)
        axes[i].legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return tsne_results

tsne_results = tsne_analysis(df)
```

### Comparison of Dimensionality Reduction Techniques

```python
def compare_dimensionality_reduction(df):
    """
    Compare PCA, t-SNE, and other dimensionality reduction techniques
    """
    from sklearn.manifold import MDS
    from sklearn.decomposition import FastICA
    
    # Prepare data
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    # Apply different techniques
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(scaled_data)
    
    mds = MDS(n_components=2, random_state=42)
    mds_result = mds.fit_transform(scaled_data)
    
    ica = FastICA(n_components=2, random_state=42)
    ica_result = ica.fit_transform(scaled_data)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define colors based on a categorical variable
    colors = df['age_group'].astype('category').cat.codes
    
    # PCA
    scatter1 = axes[0, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                 c=colors, cmap='viridis', alpha=0.7, s=20)
    axes[0, 0].set_title(f'PCA\n(Variance explained: {pca.explained_variance_ratio_.sum():.1%})')
    axes[0, 0].set_xlabel('Component 1')
    axes[0, 0].set_ylabel('Component 2')
    
    # t-SNE
    scatter2 = axes[0, 1].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                 c=colors, cmap='viridis', alpha=0.7, s=20)
    axes[0, 1].set_title('t-SNE')
    axes[0, 1].set_xlabel('Dimension 1')
    axes[0, 1].set_ylabel('Dimension 2')
    
    # MDS
    scatter3 = axes[1, 0].scatter(mds_result[:, 0], mds_result[:, 1], 
                                 c=colors, cmap='viridis', alpha=0.7, s=20)
    axes[1, 0].set_title('MDS')
    axes[1, 0].set_xlabel('Dimension 1')
    axes[1, 0].set_ylabel('Dimension 2')
    
    # ICA
    scatter4 = axes[1, 1].scatter(ica_result[:, 0], ica_result[:, 1], 
                                 c=colors, cmap='viridis', alpha=0.7, s=20)
    axes[1, 1].set_title('ICA')
    axes[1, 1].set_xlabel('Component 1')
    axes[1, 1].set_ylabel('Component 2')
    
    # Add colorbar
    cbar = plt.colorbar(scatter1, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Age Group')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Young', 'Adult', 'Middle', 'Senior'])
    
    plt.suptitle('Comparison of Dimensionality Reduction Techniques', y=1.02)
    plt.tight_layout()
    plt.show()
    
    return {
        'pca': pca_result,
        'tsne': tsne_result, 
        'mds': mds_result,
        'ica': ica_result
    }

reduction_results = compare_dimensionality_reduction(df)
```

## Advanced Multivariate Visualization Techniques

---

### Parallel Coordinates Plot

```python
from pandas.plotting import parallel_coordinates

def create_parallel_coordinates_analysis(df):
    """
    Create parallel coordinates plot for multivariate analysis
    """
    # Prepare data - select numerical variables and normalize
    numerical_cols = ['age', 'income', 'education_years', 'experience_years', 'satisfaction_score']
    df_normalized = df[numerical_cols + ['department']].copy()
    
    # Normalize numerical columns to 0-1 scale for better visualization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])
    
    # Create parallel coordinates plot
    plt.figure(figsize=(12, 8))
    parallel_coordinates(df_normalized, 'department', alpha=0.3, colormap='tab10')
    plt.title('Parallel Coordinates Plot by Department')
    plt.ylabel('Normalized Values')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Create grouped parallel coordinates
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # By Age Group
    df_age_normalized = df[numerical_cols + ['age_group']].copy()
    df_age_normalized[numerical_cols] = scaler.fit_transform(df_age_normalized[numerical_cols])
    
    parallel_coordinates(df_age_normalized, 'age_group', ax=axes[0], alpha=0.4, colormap='viridis')
    axes[0].set_title('Parallel Coordinates by Age Group')
    axes[0].set_ylabel('Normalized Values')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # By Gender
    df_gender_normalized = df[numerical_cols + ['gender']].copy()
    df_gender_normalized[numerical_cols] = scaler.fit_transform(df_gender_normalized[numerical_cols])
    
    parallel_coordinates(df_gender_normalized, 'gender', ax=axes[1], alpha=0.4, colormap='Set1')
    axes[1].set_title('Parallel Coordinates by Gender')
    axes[1].set_ylabel('Normalized Values')
    axes[1].set_xticklabels(numerical_cols, rotation=45)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

create_parallel_coordinates_analysis(df)
```

### Radar Chart Analysis

```python
def create_radar_chart_analysis(df):
    """
    Create radar charts for multivariate profile analysis
    """
    import math
    
    # Prepare data by groups
    numerical_cols = ['age', 'income', 'education_years', 'experience_years', 'satisfaction_score']
    
    # Calculate mean values by department
    dept_means = df.groupby('department')[numerical_cols].mean()
    
    # Normalize to 0-100 scale for radar chart
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))
    dept_means_normalized = pd.DataFrame(
        scaler.fit_transform(dept_means), 
        columns=dept_means.columns, 
        index=dept_means.index
    )
    
    # Create radar chart
    def create_radar_chart(data, title):
        categories = list(data.columns)
        N = len(categories)
        
        # Compute angle for each category
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot each department
        colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
        for idx, (dept, values) in enumerate(data.iterrows()):
            values_list = values.tolist()
            values_list += values_list[:1]  # Complete the circle
            
            ax.plot(angles, values_list, 'o-', linewidth=2, 
                   label=dept, color=colors[idx], alpha=0.8)
            ax.fill(angles, values_list, alpha=0.1, color=colors[idx])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title(title, size=16, weight='bold', pad=20)
        
        return fig, ax
    
    fig, ax = create_radar_chart(dept_means_normalized, 
                                'Department Profile Comparison (Radar Chart)')
    plt.tight_layout()
    plt.show()
    
    # Create individual radar charts for top departments
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    top_departments = dept_means_normalized.index[:4]  # Top 4 departments
    
    for idx, dept in enumerate(top_departments):
        values = dept_means_normalized.loc[dept].tolist()
        values += values[:1]
        
        categories = list(dept_means_normalized.columns)
        angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
        angles += angles[:1]
        
        axes[idx].plot(angles, values, 'o-', linewidth=2, alpha=0.8, color='red')
        axes[idx].fill(angles, values, alpha=0.3, color='red')
        axes[idx].set_xticks(angles[:-1])
        axes[idx].set_xticklabels(categories, size=8)
        axes[idx].set_ylim(0, 100)
        axes[idx].set_title(f'{dept} Profile', size=12, weight='bold')
        axes[idx].grid(True)
    
    plt.suptitle('Individual Department Profiles', size=16, weight='bold')
    plt.tight_layout()
    plt.show()
    
    return dept_means_normalized

radar_data = create_radar_chart_analysis(df)
```

### Andrews Curves

```python
def create_andrews_curves_analysis(df):
    """
    Create Andrews curves for multivariate data visualization
    """
    from pandas.plotting import andrews_curves
    
    # Prepare data
    numerical_cols = ['age', 'income', 'education_years', 'experience_years', 'satisfaction_score']
    df_andrews = df[numerical_cols + ['department', 'age_group']].copy()
    
    # Standardize numerical columns
    scaler = StandardScaler()
    df_andrews[numerical_cols] = scaler.fit_transform(df_andrews[numerical_cols])
    
    # Create Andrews curves
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # By department
    andrews_curves(df_andrews[numerical_cols + ['department']], 'department', 
                   ax=axes[0], alpha=0.3, colormap='tab10')
    axes[0].set_title('Andrews Curves by Department')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # By age group
    andrews_curves(df_andrews[numerical_cols + ['age_group']], 'age_group', 
                   ax=axes[1], alpha=0.4, colormap='viridis')
    axes[1].set_title('Andrews Curves by Age Group')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    print("Andrews Curves Interpretation:")
    print("- Similar curves indicate similar multivariate profiles")
    print("- Curves that separate well indicate good discrimination between groups")
    print("- Crossing patterns show complex relationships between groups")

create_andrews_curves_analysis(df)
```

## Multivariate Statistical Analysis

---

### MANOVA (Multivariate Analysis of Variance)

```python
def multivariate_anova_analysis(df):
    """
    Perform MANOVA to test group differences across multiple variables
    """
    from statsmodels.multivariate.manova import MANOVA
    
    # Prepare data
    numerical_cols = ['age', 'income', 'education_years', 'experience_years', 'satisfaction_score']
    
    # Create formula for MANOVA
    formula = f"{' + '.join(numerical_cols)} ~ department"
    
    # Perform MANOVA
    manova = MANOVA.from_formula(formula, data=df)
    manova_results = manova.mv_test()
    
    print("MANOVA Results - Testing Department Effects:")
    print("=" * 50)
    print(manova_results)
    
    # Follow-up univariate ANOVAs
    from scipy import stats
    
    print("\nFollow-up Univariate ANOVAs:")
    print("=" * 35)
    
    for col in numerical_cols:
        # Group data by department
        groups = [df[df['department'] == dept][col].values 
                 for dept in df['department'].unique()]
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        print(f"{col}:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        print()
    
    return manova_results

manova_results = multivariate_anova_analysis(df)
```

### Multivariate Normality Testing

```python
def test_multivariate_normality(df):
    """
    Test multivariate normality assumptions
    """
    from scipy.stats import jarque_bera, shapiro
    import scipy.stats as stats
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    print("MULTIVARIATE NORMALITY ASSESSMENT")
    print("=" * 40)
    
    # 1. Univariate normality tests
    print("1. Univariate Normality Tests:")
    print("-" * 30)
    
    normality_results = {}
    for col in numerical_cols:
        data = df[col].dropna()
        
        # Shapiro-Wilk test (for smaller samples)
        if len(data) <= 5000:
            sw_stat, sw_p = shapiro(data)
            normality_results[col] = {'shapiro_p': sw_p}
        
        # Jarque-Bera test
        jb_stat, jb_p = jarque_bera(data)
        normality_results[col]['jarque_bera_p'] = jb_p
        
        print(f"{col}:")
        if len(data) <= 5000:
            print(f"  Shapiro-Wilk p-value: {sw_p:.4f}")
        print(f"  Jarque-Bera p-value: {jb_p:.4f}")
        print(f"  Normal: {'Yes' if jb_p > 0.05 else 'No'}")
        print()
    
    # 2. Q-Q plots for visual assessment
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            stats.probplot(df[col].dropna(), dist="norm", plot=axes[i])
            axes[i].set_title(f'Q-Q Plot: {col}')
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Mahalanobis distance for multivariate outliers
    from scipy.spatial.distance import mahalanobis
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    # Calculate covariance matrix
    cov_matrix = np.cov(scaled_data.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean_vector = np.mean(scaled_data, axis=0)
    
    # Calculate Mahalanobis distances
    mahal_distances = []
    for i in range(scaled_data.shape[0]):
        mahal_dist = mahalanobis(scaled_data[i], mean_vector, inv_cov_matrix)
        mahal_distances.append(mahal_dist)
    
    mahal_distances = np.array(mahal_distances)
    
    # Chi-square critical value for outlier detection
    chi2_critical = stats.chi2.ppf(0.975, df=len(numerical_cols))
    outliers = mahal_distances > chi2_critical
    
    print(f"3. Multivariate Outlier Detection:")
    print(f"   Chi-square critical value (α=0.025): {chi2_critical:.4f}")
    print(f"   Number of multivariate outliers: {np.sum(outliers)}")
    print(f"   Percentage of outliers: {np.mean(outliers):.1%}")
    
    # Plot Mahalanobis distances
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(mahal_distances)), mahal_distances, alpha=0.6)
    plt.axhline(y=chi2_critical, color='red', linestyle='--', label='Critical value')
    plt.xlabel('Observation Index')
    plt.ylabel('Mahalanobis Distance')
    plt.title('Mahalanobis Distance Plot')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(mahal_distances, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=chi2_critical, color='red', linestyle='--', label='Critical value')
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Mahalanobis Distances')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return normality_results, mahal_distances, outliers

normality_results, mahal_distances, outliers = test_multivariate_normality(df)
```

## Best Practices & Interpretation Guidelines

---

### Multivariate Analysis Workflow

```python
def multivariate_analysis_workflow(df):
    """
    Comprehensive workflow for multivariate analysis
    """
    print("MULTIVARIATE ANALYSIS WORKFLOW")
    print("=" * 40)
    
    workflow_steps = {
        "Step 1: Data Exploration": [
            "✓ Check data dimensions and structure",
            "✓ Identify variable types (numerical, categorical)",
            "✓ Examine missing values and outliers",
            "✓ Assess data quality and distributions"
        ],
        "Step 2: Correlation Analysis": [
            "✓ Calculate correlation matrices",
            "✓ Identify multicollinearity issues",
            "✓ Examine linear and non-linear relationships",
            "✓ Test correlation significance"
        ],
        "Step 3: Visualization": [
            "✓ Create comprehensive pairplots",
            "✓ Generate correlation heatmaps",
            "✓ Use parallel coordinates for patterns",
            "✓ Apply radar charts for group comparisons"
        ],
        "Step 4: Dimensionality Reduction": [
            "✓ Apply PCA for linear relationships",
            "✓ Use t-SNE for non-linear patterns",
            "✓ Compare multiple reduction techniques",
            "✓ Interpret component meanings"
        ],
        "Step 5: Statistical Testing": [
            "✓ Test multivariate normality assumptions",
            "✓ Perform MANOVA for group differences",
            "✓ Identify multivariate outliers",
            "✓ Validate findings statistically"
        ],
        "Step 6: Interpretation & Action": [
            "✓ Synthesize findings across analyses",
            "✓ Identify key patterns and relationships",
            "✓ Make data-driven recommendations",
            "✓ Plan follow-up analyses or modeling"
        ]
    }
    
    for step, tasks in workflow_steps.items():
        print(f"\n{step}:")
        for task in tasks:
            print(f"  {task}")
    
    print(f"\nCurrent Dataset Assessment:")
    print(f"  Observations: {len(df)}")
    print(f"  Variables: {len(df.columns)}")
    print(f"  Numerical variables: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  Categorical variables: {len(df.select_dtypes(include=['object', 'category']).columns)}")

multivariate_analysis_workflow(df)
```

### Interpretation Guidelines

```python
def interpretation_guidelines():
    """
    Guidelines for interpreting multivariate analysis results
    """
    guidelines = {
        "Correlation Analysis": {
            "Strong correlations (|r| > 0.7)": "Consider for feature selection or multicollinearity",
            "Moderate correlations (0.3 < |r| < 0.7)": "Meaningful relationships worth investigating",
            "Weak correlations (|r| < 0.3)": "Limited linear relationship, check non-linear patterns",
            "Pattern clusters": "Groups of related variables, consider dimensionality reduction"
        },
        "PCA Interpretation": {
            "High loadings (|loading| > 0.7)": "Variable strongly contributes to component",
            "Scree plot elbow": "Optimal number of components to retain",
            "Cumulative variance > 80%": "Good representation of original data",
            "Component meanings": "Interpret based on highest loading variables"
        },
        "t-SNE Interpretation": {
            "Tight clusters": "Similar data points, potential natural groupings",
            "Scattered points": "Diverse or noisy data",
            "Perplexity effects": "Lower values = local structure, higher = global structure",
            "Multiple runs": "Check consistency across different random seeds"
        },
        "Visualization Insights": {
            "Pairplot patterns": "Linear, non-linear, or no relationships between variables",
            "Heatmap clusters": "Groups of correlated variables",
            "Parallel coordinates": "Multivariate profiles and group differences",
            "Outlier detection": "Unusual observations requiring investigation"
        }
    }
    
    print("MULTIVARIATE ANALYSIS INTERPRETATION GUIDE")
    print("=" * 50)
    
    for analysis_type, interpretations in guidelines.items():
        print(f"\n{analysis_type}:")
        print("-" * len(analysis_type))
        for pattern, meaning in interpretations.items():
            print(f"• {pattern}: {meaning}")
    
    print(f"\n📋 KEY REMINDERS:")
    print("• Always validate statistical assumptions before applying tests")
    print("• Consider both statistical and practical significance")
    print("• Use multiple visualization techniques for comprehensive understanding") 
    print("• Domain knowledge is crucial for proper interpretation")
    print("• Document findings and methodology for reproducibility")

interpretation_guidelines()
```

Analisis multivariat adalah toolkit yang sangat powerful untuk memahami struktur data kompleks. Dengan menggabungkan teknik visualisasi, statistik, dan dimensionality reduction, kita bisa mengungkap insights yang tersembunyi dalam data high-dimensional. Ingat bahwa setiap teknik memiliki kelebihan dan keterbatasan masing-masing, jadi gunakan pendekatan multi-faceted untuk analisis yang komprehensif!