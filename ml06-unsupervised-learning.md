# [ai] #06 - ML Unsupervised Learning 
![ml](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ml.png)

## Understanding Unsupervised Learning

---

Unsupervised learning adalah tipe ML dimana kita tidak punya target labels - kita cuma punya input data X tanpa output y. Tujuannya adalah menemukan hidden patterns, structures, atau relationships dalam data. Bayangkan seperti explorer yang menjelajahi wilayah baru tanpa peta - kita cari pola dan struktur yang menarik.

Main applications:

- **Clustering**: Grouping similar data points
- **Dimensionality Reduction**: Simplify data while preserving important info
- **Anomaly Detection**: Find outliers or unusual patterns
- **Association Rules**: Discover relationships between variables

## K-Means Clustering

---

K-Means adalah algoritma clustering paling populer yang membagi data menjadi k clusters berdasarkan similarity.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Generate sample data
X_blobs, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, 
                            random_state=42, n_features=2)

# Basic K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_blobs)

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true, cmap='viridis', alpha=0.7)
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 3, 2)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c='lightgray', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3)

# Draw circles around clusters
for i, center in enumerate(kmeans.cluster_centers_):
    # Calculate distances from center to all points in this cluster
    cluster_points = X_blobs[y_pred == i]
    distances = np.sqrt(np.sum((cluster_points - center)**2, axis=1))
    radius = np.max(distances)
    
    circle = plt.Circle(center, radius, fill=False, color='red', linestyle='--', alpha=0.7)
    plt.gca().add_patch(circle)

plt.title('Cluster Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')

plt.tight_layout()
plt.show()

print(f"Silhouette Score: {silhouette_score(X_blobs, y_pred):.3f}")
print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, y_pred):.3f}")
```

### Finding Optimal Number of Clusters

```python
# Elbow Method
inertias = []
silhouette_scores = []
k_range = range(1, 11)

for k in k_range:
    if k == 1:
        inertias.append(0)  # No clustering for k=1
        silhouette_scores.append(0)
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X_blobs)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_blobs, y_pred))

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)

# Highlight elbow point
optimal_k_elbow = 4  # Visual inspection
plt.axvline(x=optimal_k_elbow, color='red', linestyle='--', alpha=0.7, 
           label=f'Optimal k = {optimal_k_elbow}')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_range[1:], silhouette_scores[1:], 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.grid(True, alpha=0.3)

# Highlight best silhouette score
optimal_k_silhouette = k_range[1:][np.argmax(silhouette_scores[1:])]
plt.axvline(x=optimal_k_silhouette, color='red', linestyle='--', alpha=0.7,
           label=f'Best k = {optimal_k_silhouette}')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Optimal k by Elbow Method: {optimal_k_elbow}")
print(f"Optimal k by Silhouette Score: {optimal_k_silhouette}")
```

### K-Means Limitations

```python
# Show K-Means limitations with non-spherical clusters
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Different data types
datasets = [
    ("Circular Clusters", make_circles(n_samples=300, noise=0.1, factor=0.6, random_state=42)),
    ("Moon-shaped Clusters", make_moons(n_samples=300, noise=0.1, random_state=42)),
    ("Elongated Clusters", make_blobs(n_samples=300, centers=2, cluster_std=2.0, 
                                     random_state=42, n_features=2))
]

for i, (name, (X, y_true)) in enumerate(datasets):
    # Apply K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # True clusters
    axes[0, i].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    axes[0, i].set_title(f'{name} - True Clusters')
    axes[0, i].set_xlabel('Feature 1')
    axes[0, i].set_ylabel('Feature 2')
    
    # K-Means results
    axes[1, i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    axes[1, i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                      c='red', marker='x', s=200, linewidths=3)
    axes[1, i].set_title(f'{name} - K-Means Results')
    axes[1, i].set_xlabel('Feature 1')
    axes[1, i].set_ylabel('Feature 2')
    
    # Print performance
    sil_score = silhouette_score(X, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    axes[1, i].text(0.05, 0.95, f'Silhouette: {sil_score:.3f}\nARI: {ari:.3f}', 
                   transform=axes[1, i].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```

## Hierarchical Clustering

---

Hierarchical clustering membuat tree-like cluster structure yang bisa divisualize dengan dendrogram.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Generate sample data
X_hier, _ = make_blobs(n_samples=50, centers=3, n_features=2, 
                      cluster_std=1.0, random_state=42)

# Hierarchical clustering with different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']

plt.figure(figsize=(20, 15))

for i, method in enumerate(linkage_methods):
    # Agglomerative clustering
    agg_cluster = AgglomerativeClustering(n_clusters=3, linkage=method)
    y_pred = agg_cluster.fit_predict(X_hier)
    
    # Create linkage matrix for dendrogram
    linkage_matrix = linkage(X_hier, method=method)
    
    # Plot dendrogram
    plt.subplot(3, 4, i+1)
    dendrogram(linkage_matrix, truncate_mode='level', p=3)
    plt.title(f'Dendrogram - {method.title()} Linkage')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # Plot clusters
    plt.subplot(3, 4, i+5)
    plt.scatter(X_hier[:, 0], X_hier[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.title(f'Clusters - {method.title()} Linkage')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Silhouette score
    sil_score = silhouette_score(X_hier, y_pred)
    plt.text(0.05, 0.95, f'Silhouette: {sil_score:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

# Compare with K-Means
kmeans_hier = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans_hier.fit_predict(X_hier)

plt.subplot(3, 4, 9)
plt.scatter(X_hier[:, 0], X_hier[:, 1], c=y_kmeans, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_hier.cluster_centers_[:, 0], kmeans_hier.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Comparison')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

sil_kmeans = silhouette_score(X_hier, y_kmeans)
plt.text(0.05, 0.95, f'Silhouette: {sil_kmeans:.3f}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```

## DBSCAN - Density-Based Clustering

---

DBSCAN dapat menemukan clusters dengan arbitrary shapes dan mengidentifikasi noise points.

```python
from sklearn.cluster import DBSCAN

# Test DBSCAN on different datasets
datasets = [
    ("Blobs", make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)),
    ("Circles", make_circles(n_samples=300, noise=0.05, factor=0.6, random_state=42)),
    ("Moons", make_moons(n_samples=300, noise=0.1, random_state=42))
]

plt.figure(figsize=(18, 12))

for i, (name, (X, y_true)) in enumerate(datasets):
    # Standardize features (important for DBSCAN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    y_pred = dbscan.fit_predict(X_scaled)
    
    # Count clusters and noise points
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = list(y_pred).count(-1)
    
    # Plot true clusters
    plt.subplot(3, 3, i*3 + 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.title(f'{name} - True Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot DBSCAN results
    plt.subplot(3, 3, i*3 + 2)
    unique_labels = set(y_pred)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise
            col = 'black'
            marker = 'x'
            alpha = 0.5
        else:
            marker = 'o'
            alpha = 0.7
            
        class_member_mask = (y_pred == k)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, alpha=alpha, s=50)
    
    plt.title(f'{name} - DBSCAN\nClusters: {n_clusters}, Noise: {n_noise}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Compare with K-Means
    plt.subplot(3, 3, i*3 + 3)
    kmeans_comp = KMeans(n_clusters=2 if name != "Blobs" else 4, random_state=42)
    y_kmeans_comp = kmeans_comp.fit_predict(X)
    
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans_comp, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans_comp.cluster_centers_[:, 0], kmeans_comp.cluster_centers_[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    plt.title(f'{name} - K-Means')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Print comparison metrics
    if len(set(y_pred)) > 1:  # Only if DBSCAN found clusters
        dbscan_sil = silhouette_score(X, y_pred)
        kmeans_sil = silhouette_score(X, y_kmeans_comp)
        print(f"{name} - DBSCAN Silhouette: {dbscan_sil:.3f}, K-Means Silhouette: {kmeans_sil:.3f}")

plt.tight_layout()
plt.show()
```

### DBSCAN Parameter Tuning

```python
# Find optimal eps using k-distance graph
from sklearn.neighbors import NearestNeighbors

def plot_k_distance(X, k=4):
    """Plot k-distance graph to help choose eps parameter"""
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # Sort distances
    distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k}-NN Distance')
    plt.title(f'{k}-Distance Graph')
    plt.grid(True, alpha=0.3)
    
    # Find elbow point (simplified)
    knee_point = np.where(np.diff(distances) > np.percentile(np.diff(distances), 95))[0]
    if len(knee_point) > 0:
        suggested_eps = distances[knee_point[0]]
        plt.axhline(y=suggested_eps, color='red', linestyle='--', 
                   label=f'Suggested eps ≈ {suggested_eps:.3f}')
        plt.legend()
    
    plt.show()
    return distances

# Apply to circles dataset
X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.6, random_state=42)
X_circles_scaled = StandardScaler().fit_transform(X_circles)

distances = plot_k_distance(X_circles_scaled, k=4)

# Test different eps values
eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
plt.figure(figsize=(20, 4))

for i, eps in enumerate(eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    y_pred = dbscan.fit_predict(X_circles_scaled)
    
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = list(y_pred).count(-1)
    
    plt.subplot(1, len(eps_values), i+1)
    unique_labels = set(y_pred)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'black'
            marker = 'x'
        else:
            marker = 'o'
            
        class_member_mask = (y_pred == k)
        xy = X_circles[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, alpha=0.7)
    
    plt.title(f'eps={eps}\nClusters: {n_clusters}, Noise: {n_noise}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()  
plt.show()
```

## Principal Component Analysis (PCA)

---

PCA adalah teknik dimensionality reduction yang mencari directions (principal components) dengan variance terbesar dalam data.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_digits

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Standardize the data
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_iris_scaled)

# Explained variance ratio
explained_var_ratio = pca.explained_variance_ratio_
cumsum_var_ratio = np.cumsum(explained_var_ratio)

plt.figure(figsize=(15, 10))

# Plot explained variance
plt.subplot(2, 3, 1)
plt.bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Explained Variance')
plt.xticks(range(1, len(explained_var_ratio) + 1))

plt.subplot(2, 3, 2)
plt.plot(range(1, len(cumsum_var_ratio) + 1), cumsum_var_ratio, 'bo-', linewidth=2)
plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, len(cumsum_var_ratio) + 1))

# 2D visualization
plt.subplot(2, 3, 3)
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.scatter(X_pca[y_iris == i, 0], X_pca[y_iris == i, 1], 
               c=color, alpha=0.7, label=iris.target_names[i])
plt.xlabel(f'PC1 ({explained_var_ratio[0]:.1%} variance)')
plt.ylabel(f'PC2 ({explained_var_ratio[1]:.1%} variance)')
plt.title('PCA - First 2 Components')
plt.legend()

# 3D visualization
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(2, 3, 4, projection='3d')
for i, color in enumerate(colors):
    ax.scatter(X_pca[y_iris == i, 0], X_pca[y_iris == i, 1], X_pca[y_iris == i, 2],
              c=color, alpha=0.7, label=iris.target_names[i])
ax.set_xlabel(f'PC1 ({explained_var_ratio[0]:.1%})')
ax.set_ylabel(f'PC2 ({explained_var_ratio[1]:.1%})')
ax.set_zlabel(f'PC3 ({explained_var_ratio[2]:.1%})')
ax.set_title('PCA - First 3 Components')
ax.legend()

# Component loadings (feature contributions)
plt.subplot(2, 3, 5)
feature_names = iris.feature_names
components_df = pd.DataFrame(
    pca.components_[:2].T,
    columns=['PC1', 'PC2'],
    index=feature_names
)

components_df.plot(kind='bar', ax=plt.gca())
plt.title('Feature Loadings on PC1 and PC2')
plt.xlabel('Features')
plt.ylabel('Loading')
plt.xticks(rotation=45)
plt.legend()

# Biplot
plt.subplot(2, 3, 6)
# Plot data points
for i, color in enumerate(colors):
    plt.scatter(X_pca[y_iris == i, 0], X_pca[y_iris == i, 1], 
               c=color, alpha=0.6, label=iris.target_names[i])

# Plot loading vectors
scale = 3
for i, (feature, loading) in enumerate(zip(feature_names, pca.components_[:2].T)):
    plt.arrow(0, 0, loading[0]*scale, loading[1]*scale, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    plt.text(loading[0]*scale*1.1, loading[1]*scale*1.1, feature, 
            fontsize=10, ha='center', va='center')

plt.xlabel(f'PC1 ({explained_var_ratio[0]:.1%} variance)')
plt.ylabel(f'PC2 ({explained_var_ratio[1]:.1%} variance)')
plt.title('PCA Biplot')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("PCA Results:")
print(f"Original dimensions: {X_iris.shape[1]}")
print(f"Explained variance by each component: {explained_var_ratio}")
print(f"Cumulative explained variance: {cumsum_var_ratio}")
```

### PCA for High-Dimensional Data

```python
# Load digits dataset (8x8 images = 64 features)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"Original digits data shape: {X_digits.shape}")

# Apply PCA
pca_digits = PCA()
X_digits_pca = pca_digits.fit_transform(X_digits)

# Find number of components for different variance thresholds
thresholds = [0.80, 0.90, 0.95, 0.99]
n_components_needed = []

cumsum_var = np.cumsum(pca_digits.explained_variance_ratio_)

for threshold in thresholds:
    n_comp = np.argmax(cumsum_var >= threshold) + 1
    n_components_needed.append(n_comp)
    print(f"Components needed for {threshold:.0%} variance: {n_comp}")

# Visualize
plt.figure(figsize=(15, 10))

# Explained variance
plt.subplot(2, 3, 1)
plt.plot(range(1, 51), pca_digits.explained_variance_ratio_[:50], 'b-', linewidth=2)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance (First 50 Components)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(range(1, 51), cumsum_var[:50], 'r-', linewidth=2)
for threshold, n_comp in zip(thresholds, n_components_needed):
    if n_comp <= 50:
        plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7)
        plt.axvline(x=n_comp, color='gray', linestyle='--', alpha=0.7)
        plt.text(n_comp+1, threshold-0.02, f'{n_comp} components', fontsize=8)

plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid(True, alpha=0.3)

# Reconstruct digits with different number of components
components_to_test = [5, 10, 20, 50]
original_digit = X_digits[0].reshape(8, 8)

plt.subplot(2, 3, 3)
plt.imshow(original_digit, cmap='gray')
plt.title('Original Digit')
plt.axis('off')

for i, n_comp in enumerate(components_to_test):
    # Reconstruct with n components
    pca_temp = PCA(n_components=n_comp)
    X_transformed = pca_temp.fit_transform(X_digits)
    X_reconstructed = pca_temp.inverse_transform(X_transformed)
    
    reconstructed_digit = X_reconstructed[0].reshape(8, 8)
    
    plt.subplot(2, 3, 4 + i)
    plt.imshow(reconstructed_digit, cmap='gray')
    
    # Calculate reconstruction error
    mse = np.mean((original_digit - reconstructed_digit)**2)
    var_explained = np.sum(pca_temp.explained_variance_ratio_)
    
    plt.title(f'{n_comp} components\nMSE: {mse:.3f}\nVar: {var_explained:.1%}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 2D visualization of digits
pca_2d = PCA(n_components=2)
X_digits_2d = pca_2d.fit_transform(X_digits)

plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for digit in range(10):
    mask = y_digits == digit
    plt.scatter(X_digits_2d[mask, 0], X_digits_2d[mask, 1], 
               c=[colors[digit]], alpha=0.6, label=f'Digit {digit}')

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA Visualization')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# t-SNE 2D
plt.subplot(1, 2, 2)
for digit in range(10):
    mask = y_digits_subset == digit
    plt.scatter(X_digits_tsne[mask, 0], X_digits_tsne[mask, 1], 
               c=[colors[digit]], alpha=0.6, label=f'{digit}')

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Notice how t-SNE better separates the digit classes compared to PCA!")
```

### t-SNE Parameter Effects

```python
# Test different perplexity values
perplexity_values = [5, 30, 50, 100]

plt.figure(figsize=(20, 5))

for i, perp in enumerate(perplexity_values):
    print(f"Computing t-SNE with perplexity={perp}...")
    
    tsne_perp = TSNE(n_components=2, random_state=42, perplexity=perp, n_iter=1000)
    X_tsne_perp = tsne_perp.fit_transform(X_digits_pca_prep[:300])  # Smaller subset for speed
    y_subset = y_digits_subset[:300]
    
    plt.subplot(1, 4, i+1)
    
    for digit in range(10):
        mask = y_subset == digit
        plt.scatter(X_tsne_perp[mask, 0], X_tsne_perp[mask, 1], 
                   c=[colors[digit]], alpha=0.7, s=20)
    
    plt.title(f't-SNE (perplexity={perp})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

plt.tight_layout()
plt.show()
```

## Anomaly Detection

---

Anomaly detection menemukan data points yang significantly different dari mayoritas data.

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Generate data with outliers
np.random.seed(42)
X_normal = np.random.randn(200, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X_anomaly = np.vstack([X_normal, X_outliers])

# True labels (1 for normal, -1 for outlier)
y_true = np.ones(len(X_anomaly))
y_true[-20:] = -1

# Anomaly detection algorithms
anomaly_algorithms = [
    ("Isolation Forest", IsolationForest(contamination=0.1, random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=20, contamination=0.1)),
    ("One-Class SVM", OneClassSVM(gamma='scale', nu=0.1))
]

plt.figure(figsize=(18, 6))

for i, (name, algorithm) in enumerate(anomaly_algorithms):
    # Fit and predict
    if name == "Local Outlier Factor":
        y_pred = algorithm.fit_predict(X_anomaly)
    else:
        y_pred = algorithm.fit(X_anomaly).predict(X_anomaly)
    
    # Plot results
    plt.subplot(1, 3, i+1)
    
    # Plot normal points
    normal_mask = y_pred == 1
    outlier_mask = y_pred == -1
    
    plt.scatter(X_anomaly[normal_mask, 0], X_anomaly[normal_mask, 1], 
               c='blue', alpha=0.6, label='Normal')
    plt.scatter(X_anomaly[outlier_mask, 0], X_anomaly[outlier_mask, 1], 
               c='red', alpha=0.8, label='Anomaly')
    
    # Calculate performance
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=-1)
    recall = recall_score(y_true, y_pred, pos_label=-1)
    
    plt.title(f'{name}\nAcc: {accuracy:.2f}, Prec: {precision:.2f}, Rec: {recall:.2f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Real-world Anomaly Detection Example

```python
# Create more realistic anomaly detection scenario
from sklearn.datasets import make_classification

# Generate imbalanced dataset
X_real, y_real = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                    n_redundant=2, n_clusters_per_class=1, 
                                    weights=[0.95, 0.05], random_state=42)

# Convert to anomaly detection problem (minority class as anomaly)
y_anomaly = np.where(y_real == 1, -1, 1)  # 1 for normal, -1 for anomaly

print(f"Normal samples: {np.sum(y_anomaly == 1)}")
print(f"Anomaly samples: {np.sum(y_anomaly == -1)}")

# Split data
from sklearn.model_selection import train_test_split
X_train_anom, X_test_anom, y_train_anom, y_test_anom = train_test_split(
    X_real, y_anomaly, test_size=0.3, random_state=42, stratify=y_anomaly)

# Train only on normal data (common in anomaly detection)
X_train_normal = X_train_anom[y_train_anom == 1]

# Anomaly detection models
models = {
    'Isolation Forest': IsolationForest(contamination=0.05, random_state=42),
    'One-Class SVM': OneClassSVM(gamma='scale', nu=0.05),
}

results = []

for name, model in models.items():
    # Train on normal data only
    model.fit(X_train_normal)
    
    # Predict on test set
    y_pred_anom = model.predict(X_test_anom)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_anom, y_pred_anom)
    precision = precision_score(y_test_anom, y_pred_anom, pos_label=-1)
    recall = recall_score(y_test_anom, y_pred_anom, pos_label=-1)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Display results
results_df = pd.DataFrame(results)
print("\nAnomaly Detection Results:")
print(results_df.round(3))

# Visualize results using PCA
pca_anom = PCA(n_components=2)
X_test_pca = pca_anom.fit_transform(X_test_anom)

plt.figure(figsize=(15, 5))

# True anomalies
plt.subplot(1, 3, 1)
normal_mask = y_test_anom == 1
anomaly_mask = y_test_anom == -1

plt.scatter(X_test_pca[normal_mask, 0], X_test_pca[normal_mask, 1], 
           c='blue', alpha=0.6, label='Normal')
plt.scatter(X_test_pca[anomaly_mask, 0], X_test_pca[anomaly_mask, 1], 
           c='red', alpha=0.8, label='True Anomaly')
plt.title('True Labels')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# Predictions from each model
for i, (name, model) in enumerate(models.items()):
    y_pred_vis = model.predict(X_test_anom)
    
    plt.subplot(1, 3, i+2)
    normal_pred_mask = y_pred_vis == 1
    anomaly_pred_mask = y_pred_vis == -1
    
    plt.scatter(X_test_pca[normal_pred_mask, 0], X_test_pca[normal_pred_mask, 1], 
               c='blue', alpha=0.6, label='Predicted Normal')
    plt.scatter(X_test_pca[anomaly_pred_mask, 0], X_test_pca[anomaly_pred_mask, 1], 
               c='red', alpha=0.8, label='Predicted Anomaly')
    
    plt.title(f'{name} Predictions')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

plt.tight_layout()
plt.show()
```

## Gaussian Mixture Models (GMM)

---

GMM adalah probabilistic model yang assumes data comes from mixture of Gaussian distributions.

```python
from sklearn.mixture import GaussianMixture

# Generate data from multiple Gaussians
np.random.seed(42)
X_gmm1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 100)
X_gmm2 = np.random.multivariate_normal([6, 6], [[1, 0.5], [0.5, 1]], 150)
X_gmm3 = np.random.multivariate_normal([2, 6], [[1, -0.5], [-0.5, 1]], 120)

X_gmm = np.vstack([X_gmm1, X_gmm2, X_gmm3])
y_true_gmm = np.hstack([np.zeros(100), np.ones(150), np.full(120, 2)])

# Apply GMM with different number of components
n_components_range = range(1, 8)
aic_scores = []
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_gmm)
    aic_scores.append(gmm.aic(X_gmm))
    bic_scores.append(gmm.bic(X_gmm))

# Plot model selection criteria
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(n_components_range, aic_scores, 'o-', label='AIC', linewidth=2)
plt.plot(n_components_range, bic_scores, 's-', label='BIC', linewidth=2)
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion')
plt.title('Model Selection: AIC vs BIC')
plt.legend()
plt.grid(True, alpha=0.3)

# Optimal number of components
optimal_n_aic = n_components_range[np.argmin(aic_scores)]
optimal_n_bic = n_components_range[np.argmin(bic_scores)]

plt.axvline(x=optimal_n_aic, color='blue', linestyle='--', alpha=0.7, label=f'Optimal AIC: {optimal_n_aic}')
plt.axvline(x=optimal_n_bic, color='red', linestyle='--', alpha=0.7, label=f'Optimal BIC: {optimal_n_bic}')

print(f"Optimal components - AIC: {optimal_n_aic}, BIC: {optimal_n_bic}")

# Fit GMM with optimal components
gmm_optimal = GaussianMixture(n_components=3, random_state=42)
y_pred_gmm = gmm_optimal.fit_predict(X_gmm)

# Plot true vs predicted clusters
plt.subplot(2, 3, 2)
plt.scatter(X_gmm[:, 0], X_gmm[:, 1], c=y_true_gmm, cmap='viridis', alpha=0.7)
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(2, 3, 3)
plt.scatter(X_gmm[:, 0], X_gmm[:, 1], c=y_pred_gmm, cmap='viridis', alpha=0.7)

# Plot Gaussian ellipses
from matplotlib.patches import Ellipse

for i in range(gmm_optimal.n_components):
    mean = gmm_optimal.means_[i]
    covariance = gmm_optimal.covariances_[i]
    
    # Calculate ellipse parameters
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues)
    
    ellipse = Ellipse(mean, width, height, angle=angle, 
                     fill=False, color='red', linewidth=2)
    plt.gca().add_patch(ellipse)

plt.title('GMM Clusters with Gaussians')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Compare with K-Means
kmeans_gmm = KMeans(n_clusters=3, random_state=42)
y_kmeans_gmm = kmeans_gmm.fit_predict(X_gmm)

plt.subplot(2, 3, 4)
plt.scatter(X_gmm[:, 0], X_gmm[:, 1], c=y_kmeans_gmm, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_gmm.cluster_centers_[:, 0], kmeans_gmm.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Comparison')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Probability contours
plt.subplot(2, 3, 5)
x_min, x_max = X_gmm[:, 0].min() - 1, X_gmm[:, 0].max() + 1
y_min, y_max = X_gmm[:, 1].min() - 1, X_gmm[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100))

mesh_points = np.array([xx.ravel(), yy.ravel()]).T
Z = gmm_optimal.score_samples(mesh_points)
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=10, colors='black', alpha=0.5)
plt.scatter(X_gmm[:, 0], X_gmm[:, 1], c=y_pred_gmm, cmap='viridis', alpha=0.7)
plt.title('GMM Probability Contours')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Component probabilities
proba = gmm_optimal.predict_proba(X_gmm)
plt.subplot(2, 3, 6)
plt.hist(proba.max(axis=1), bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Maximum Component Probability')
plt.ylabel('Frequency')
plt.title('Certainty of Cluster Assignments')

plt.tight_layout()
plt.show()

# Print model details
print(f"\nGMM Model Details:")
print(f"Converged: {gmm_optimal.converged_}")
print(f"Number of iterations: {gmm_optimal.n_iter_}")
print(f"Log-likelihood: {gmm_optimal.score(X_gmm):.3f}")
print(f"AIC: {gmm_optimal.aic(X_gmm):.3f}")
print(f"BIC: {gmm_optimal.bic(X_gmm):.3f}")

print(f"\nComponent weights: {gmm_optimal.weights_}")
print(f"\nComponent means:")
for i, mean in enumerate(gmm_optimal.means_):
    print(f"  Component {i}: [{mean[0]:.2f}, {mean[1]:.2f}]")
```

## Clustering Evaluation and Comparison

---

```python
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score

# Comprehensive clustering comparison
def evaluate_clustering(X, y_true, y_pred, method_name):
    """Evaluate clustering performance"""
    
    # Internal metrics (don't need true labels)
    silhouette = silhouette_score(X, y_pred)
    
    # External metrics (need true labels)
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    
    return {
        'Method': method_name,
        'Silhouette': silhouette,
        'ARI': ari,
        'AMI': ami,
        'Homogeneity': homogeneity,
        'Completeness': completeness
    }

# Test on different datasets
test_datasets = [
    ("Blobs", make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)),
    ("Circles", make_circles(n_samples=300, noise=0.05, factor=0.6, random_state=42)),
    ("Moons", make_moons(n_samples=300, noise=0.1, random_state=42))
]

clustering_methods = {
    'K-Means': lambda X, n_clusters: KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
    'Hierarchical': lambda X, n_clusters: AgglomerativeClustering(n_clusters=n_clusters),
    'DBSCAN': lambda X, n_clusters: DBSCAN(eps=0.3, min_samples=5),
    'GMM': lambda X, n_clusters: GaussianMixture(n_components=n_clusters, random_state=42)
}

all_results = []

for dataset_name, (X, y_true) in test_datasets:
    print(f"\nEvaluating on {dataset_name} dataset:")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = len(np.unique(y_true))
    
    for method_name, method_func in clustering_methods.items():
        try:
            if method_name == 'DBSCAN':
                model = method_func(X_scaled, n_clusters)
                y_pred = model.fit_predict(X_scaled)
            elif method_name == 'GMM':
                model = method_func(X_scaled, n_clusters)
                y_pred = model.fit_predict(X_scaled)
            else:
                model = method_func(X_scaled, n_clusters)
                y_pred = model.fit_predict(X_scaled)
            
            # Skip if only one cluster found
            if len(np.unique(y_pred)) < 2:
                continue
                
            result = evaluate_clustering(X_scaled, y_true, y_pred, method_name)
            result['Dataset'] = dataset_name
            all_results.append(result)
            
        except Exception as e:
            print(f"Error with {method_name}: {e}")

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Display results by dataset
for dataset in ["Blobs", "Circles", "Moons"]:
    dataset_results = results_df[results_df['Dataset'] == dataset]
    if not dataset_results.empty:
        print(f"\n{dataset} Dataset Results:")
        print(dataset_results[['Method', 'Silhouette', 'ARI', 'AMI']].round(3))

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics = ['Silhouette', 'ARI', 'AMI', 'Homogeneity']

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    
    # Create pivot table for heatmap
    pivot_data = results_df.pivot(index='Method', columns='Dataset', values=metric)
    
    # Plot heatmap
    im = ax.imshow(pivot_data.values, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)
    
    # Add text annotations
    for j in range(len(pivot_data.index)):
        for k in range(len(pivot_data.columns)):
            text = ax.text(k, j, f'{pivot_data.iloc[j, k]:.2f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title(f'{metric} Scores')

# Add colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout()
plt.show()
```

## Summary and Best Practices

---

```python
# Create a comprehensive guide
clustering_guide = {
    'K-Means': {
        'Best for': 'Spherical clusters, known number of clusters',
        'Pros': ['Fast', 'Simple', 'Works well with globular clusters'],
        'Cons': ['Assumes spherical clusters', 'Sensitive to initialization', 'Need to specify k'],
        'When to use': 'Customer segmentation, image segmentation, general purpose clustering'
    },
    
    'Hierarchical': {
        'Best for': 'Understanding cluster hierarchy, don\'t know k beforehand',
        'Pros': ['No need to specify k', 'Deterministic', 'Creates hierarchy'],
        'Cons': ['O(n³) complexity', 'Sensitive to noise', 'Difficult to handle large datasets'],
        'When to use': 'Small to medium datasets, need cluster hierarchy, exploratory analysis'
    },
    
    'DBSCAN': {
        'Best for': 'Arbitrary shaped clusters, noisy data',
        'Pros': ['Finds arbitrary shapes', 'Identifies noise', 'No need to specify k'],
        'Cons': ['Sensitive to hyperparameters', 'Struggles with varying densities'],
        'When to use': 'Anomaly detection, irregular cluster shapes, varying cluster sizes'
    },
    
    'GMM': {
        'Best for': 'Overlapping clusters, probabilistic assignments',
        'Pros': ['Soft clustering', 'Model-based', 'Handles overlapping clusters'],
        'Cons': ['Assumes Gaussian distributions', 'Sensitive to initialization'],
        'When to use': 'Need probability of membership, overlapping clusters, generative modeling'
    }
}

dimensionality_reduction_guide = {
    'PCA': {
        'Best for': 'Linear dimensionality reduction, feature extraction',
        'Pros': ['Linear', 'Fast', 'Interpretable components', 'Preserves global structure'],
        'Cons': ['Only linear relationships', 'Components may not be interpretable'],
        'When to use': 'Preprocessing, visualization, feature reduction, noise reduction'
    },
    
    't-SNE': {
        'Best for': 'Non-linear visualization, exploring data structure',
        'Pros': ['Great for visualization', 'Preserves local structure', 'Handles non-linear'],
        'Cons': ['Slow', 'Not deterministic', 'Only for visualization', 'Sensitive to hyperparameters'],
        'When to use': 'Data exploration, visualization, understanding cluster structure'
    }
}

print("=== UNSUPERVISED LEARNING GUIDE ===\n")

print("CLUSTERING ALGORITHMS:")
for method, info in clustering_guide.items():
    print(f"\n{method}:")
    print(f"  Best for: {info['Best for']}")
    print(f"  Pros: {', '.join(info['Pros'])}")
    print(f"  Cons: {', '.join(info['Cons'])}")
    print(f"  When to use: {info['When to use']}")

print("\n" + "="*50)
print("DIMENSIONALITY REDUCTION:")
for method, info in dimensionality_reduction_guide.items():
    print(f"\n{method}:")
    print(f"  Best for: {info['Best for']}")
    print(f"  Pros: {', '.join(info['Pros'])}")
    print(f"  Cons: {', '.join(info['Cons'])}")
    print(f"  When to use: {info['When to use']}")

print("\n" + "="*50)
print("BEST PRACTICES:")
print("1. Always standardize/normalize your features before clustering")
print("2. Use multiple evaluation metrics (internal + external if possible)")
print("3. Try different algorithms - no single method works for everything")
print("4. Visualize your results (use PCA/t-SNE for high-dimensional data)")
print("5. Consider domain knowledge when interpreting clusters")
print("6. Use PCA before t-SNE for high-dimensional data (>50 features)")
print("7. For anomaly detection, train only on normal data when possible")
print("8. Always evaluate the stability of your clustering results")
```

## Key Takeaways

---

1. **Unsupervised Learning** explores data without target labels - find hidden patterns
2. **K-Means** is good default choice but assumes spherical clusters
3. **Hierarchical Clustering** shows cluster relationships and doesn't need predetermined k
4. **DBSCAN** handles arbitrary shapes and identifies noise/outliers
5. **GMM** provides probabilistic cluster assignments and handles overlapping clusters
6. **PCA** is essential for dimensionality reduction and preprocessing
7. **t-SNE** excellent for visualization but only for exploration, not feature reduction
8. **Evaluation** is tricky without true labels - use multiple metrics and domain knowledge
9. **Data preprocessing** crucial - always standardize features
10. **No single algorithm** works for everything - try multiple approaches

Unsupervised learning adalah powerful tool untuk data exploration dan pattern discovery. Skills ini akan sangat berguna untuk understanding data sebelum apply supervised learning, dan juga untuk real-world problems dimana labeled data tidak tersedia. Next, kita akan explore model evaluation dan selection!(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)') plt.title('Digits Dataset - PCA 2D Visualization') plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') plt.grid(True, alpha=0.3) plt.tight_layout() plt.show()

````

## t-SNE - t-Distributed Stochastic Neighbor Embedding
---

t-SNE adalah teknik non-linear dimensionality reduction yang excellent untuk visualization.

```python
from sklearn.manifold import TSNE

# Apply t-SNE to digits dataset
print("Applying t-SNE (this may take a while...)...")

# Use subset of data for faster computation
n_samples = 1000
indices = np.random.choice(len(X_digits), n_samples, replace=False)
X_digits_subset = X_digits[indices]
y_digits_subset = y_digits[indices]

# First reduce dimensions with PCA (common practice)
pca_preprocessing = PCA(n_components=50)
X_digits_pca_prep = pca_preprocessing.fit_transform(X_digits_subset)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_digits_tsne = tsne.fit_transform(X_digits_pca_prep)

# Compare PCA vs t-SNE visualization
plt.figure(figsize=(15, 6))

# PCA 2D
plt.subplot(1, 2, 1)
X_digits_pca_2d = pca_2d.transform(X_digits_subset)

for digit in range(10):
    mask = y_digits_subset == digit
    plt.scatter(X_digits_pca_2d[mask, 0], X_digits_pca_2d[mask, 1], 
               c=[colors[digit]], alpha=0.6, label=f'{digit}')

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel
````