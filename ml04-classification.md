# [ai] #04 - ML Supervised Learning - Classification 
![ml](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ml.png)

## Understanding Classification

---

Classification adalah tipe supervised learning yang memprediksi kategori atau kelas. Berbeda dengan regression yang prediksi nilai kontinyu, classification prediksi discrete labels. Contohnya: email spam/tidak spam, gambar kucing/anjing, diagnosis penyakit, sentimen positif/negatif.

Ada dua jenis utama:

- **Binary Classification**: 2 kelas (ya/tidak, spam/ham)
- **Multi-class Classification**: >2 kelas (setosa/versicolor/virginica)

## Logistic Regression

---

Meskipun namanya "regression", ini sebenarnya classification algorithm. Logistic regression menggunakan sigmoid function untuk map nilai apapun ke range 0-1, yang bisa diinterpretasi sebagai probabilitas.

### Binary Logistic Regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Generate binary classification data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Coefficients: {log_reg.coef_[0]}")
print(f"Intercept: {log_reg.intercept_[0]:.3f}")

# Visualize decision boundary
def plot_decision_boundary(X, y, model, scaler=None, title="Decision Boundary"):
    if scaler:
        X = scaler.transform(X)
    
    plt.figure(figsize=(10, 8))
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Probability of Class 1')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

plot_decision_boundary(X_test, y_test, log_reg, scaler, "Logistic Regression Decision Boundary")
```

### Sigmoid Function Visualization

```python
# Visualize sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid_values, 'b-', linewidth=2, label='Sigmoid Function')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
plt.xlabel('z (Linear Combination)')
plt.ylabel('Probability')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1)
plt.show()

# Show how changing coefficients affects the curve
plt.figure(figsize=(12, 8))

# Different slopes
for i, slope in enumerate([0.5, 1, 2, 5]):
    plt.subplot(2, 2, i+1)
    sigmoid_vals = sigmoid(slope * z)
    plt.plot(z, sigmoid_vals, linewidth=2, label=f'Slope = {slope}')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.title(f'Sigmoid with slope = {slope}')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()
```

### Multi-class Classification

```python
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Split data
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42)

# Scale features
scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

# Multi-class logistic regression
multi_log_reg = LogisticRegression(multi_class='ovr', random_state=42)
multi_log_reg.fit(X_train_iris_scaled, y_train_iris)

# Predictions
y_pred_iris = multi_log_reg.predict(X_test_iris_scaled)
y_pred_proba_iris = multi_log_reg.predict_proba(X_test_iris_scaled)

print("Multi-class Classification Results:")
print(f"Accuracy: {accuracy_score(y_test_iris, y_pred_iris):.3f}")
print("\nClassification Report:")
print(classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test_iris, y_pred_iris)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix - Iris Classification')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add text annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()
```

## Decision Trees

---

Decision trees membuat keputusan dengan serangkaian pertanyaan yes/no. Sangat interpretable dan bisa handle both numerical dan categorical features.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification

# Generate data for decision tree
X_tree, y_tree = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1, random_state=42)

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_tree, y_tree, test_size=0.2, random_state=42)

# Train decision tree with different max_depth
depths = [2, 3, 5, None]
plt.figure(figsize=(20, 12))

for i, depth in enumerate(depths):
    # Train model
    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_clf.fit(X_train_tree, y_train_tree)
    
    # Calculate accuracy
    y_pred_tree = tree_clf.predict(X_test_tree)
    accuracy = accuracy_score(y_test_tree, y_pred_tree)
    
    # Plot decision boundary
    plt.subplot(2, 4, i+1)
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_tree[:, 0].min() - 1, X_tree[:, 0].max() + 1
    y_min, y_max = X_tree[:, 1].min() - 1, X_tree[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = tree_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.6, cmap='RdYlBu')
    plt.scatter(X_tree[:, 0], X_tree[:, 1], c=y_tree, cmap='RdYlBu', edgecolors='black')
    plt.title(f'Decision Tree (depth={depth})\nAccuracy: {accuracy:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot tree structure
    plt.subplot(2, 4, i+5)
    plot_tree(tree_clf, max_depth=3, filled=True, fontsize=8)
    plt.title(f'Tree Structure (depth={depth})')

plt.tight_layout()
plt.show()

# Feature importance
tree_final = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_final.fit(X_train_tree, y_train_tree)

feature_importance = pd.DataFrame({
    'feature': [f'Feature_{i}' for i in range(X_tree.shape[1])],
    'importance': tree_final.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Feature Importance')
plt.title('Decision Tree Feature Importance')
plt.show()
```

## Support Vector Machine (SVM)

---

SVM mencari hyperplane yang paling optimal untuk memisahkan kelas dengan margin maksimal. Sangat powerful untuk high-dimensional data.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons

# Linear SVM
linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train_scaled, y_train)
y_pred_svm = linear_svm.predict(X_test_scaled)

print(f"Linear SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")

# Visualize linear SVM
plot_decision_boundary(X_test, y_test, linear_svm, scaler, "Linear SVM Decision Boundary")

# Non-linear data examples
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Create different non-linear datasets
datasets = [
    make_circles(n_samples=300, noise=0.1, factor=0.6, random_state=42),
    make_moons(n_samples=300, noise=0.1, random_state=42),
    make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                       n_clusters_per_class=2, random_state=42)
]

dataset_names = ['Circles', 'Moons', 'Multi-cluster']
kernels = ['linear', 'poly', 'rbf']

for i, (X_data, y_data) in enumerate(datasets):
    # Split and scale
    X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42)
    
    scaler_nl = StandardScaler()
    X_train_nl_scaled = scaler_nl.fit_transform(X_train_nl)
    X_test_nl_scaled = scaler_nl.transform(X_test_nl)
    
    for j, kernel in enumerate(kernels):
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train_nl_scaled, y_train_nl)
        
        accuracy = svm.score(X_test_nl_scaled, y_test_nl)
        
        # Plot only first row for different datasets, second row for different kernels on moons
        if i == 0:  # First dataset, show all kernels
            ax = axes[0, j]
            
            # Create mesh
            h = 0.02
            X_scaled = scaler_nl.transform(X_data)
            x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
            y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.6, cmap='RdYlBu')
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_data, cmap='RdYlBu', edgecolors='black')
            ax.set_title(f'{kernel.upper()} SVM on {dataset_names[i]}\nAccuracy: {accuracy:.3f}')
        
        elif i == 1 and j < 3:  # Second dataset, show comparison
            ax = axes[1, j]
            
            # Create mesh
            h = 0.02
            X_scaled = scaler_nl.transform(X_data)
            x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
            y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.6, cmap='RdYlBu')
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_data, cmap='RdYlBu', edgecolors='black')
            ax.set_title(f'{kernel.upper()} SVM on {dataset_names[i]}\nAccuracy: {accuracy:.3f}')

plt.tight_layout()
plt.show()
```

## K-Nearest Neighbors (KNN)

---

KNN adalah lazy learning algorithm yang mengklasifikasi berdasarkan mayoritas vote dari k tetangga terdekat.

```python
from sklearn.neighbors import KNeighborsClassifier

# Test different k values
k_values = [1, 3, 5, 7, 15, 30]
knn_scores = []

plt.figure(figsize=(18, 12))

for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    y_pred_knn = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_knn)
    knn_scores.append(accuracy)
    
    # Plot decision boundary
    plt.subplot(2, 3, i+1)
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
    y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.6, cmap='RdYlBu')
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black')
    plt.title(f'KNN (k={k})\nAccuracy: {accuracy:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Performance vs k Value')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)

# Highlight best k
best_k = k_values[np.argmax(knn_scores)]
best_score = max(knn_scores)
plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7)
plt.text(best_k + 0.5, best_score - 0.01, f'Best k={best_k}\nScore={best_score:.3f}', 
         bbox=dict(boxstyle="round", facecolor='wheat'))
plt.show()
```

## Naive Bayes

---

Naive Bayes assumes independence antara features (naive assumption) dan menggunakan Bayes theorem untuk classification.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Gaussian Naive Bayes for continuous features
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_pred_gnb = gnb.predict(X_test_scaled)

print(f"Gaussian Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_gnb):.3f}")

# Visualize Gaussian NB decision boundary
plot_decision_boundary(X_test, y_test, gnb, scaler, "Gaussian Naive Bayes Decision Boundary")

# Text classification example with Multinomial NB
# Load subset of 20newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_text = vectorizer.fit_transform(newsgroups_train.data)
X_test_text = vectorizer.transform(newsgroups_test.data)

# Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train_text, newsgroups_train.target)
y_pred_text = mnb.predict(X_test_text)

print(f"\nText Classification Results:")
print(f"Multinomial Naive Bayes Accuracy: {accuracy_score(newsgroups_test.target, y_pred_text):.3f}")
print("\nClassification Report:")
print(classification_report(newsgroups_test.target, y_pred_text, 
                          target_names=newsgroups_train.target_names))

# Show most informative features
feature_names = vectorizer.get_feature_names_out()
for i, category in enumerate(newsgroups_train.target_names):
    top_features = np.argsort(mnb.feature_log_prob_[i])[-10:]
    print(f"\nTop words for {category}:")
    print([feature_names[j] for j in top_features])
```

## Evaluation Metrics for Classification

---

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve, precision_recall_curve, average_precision_score)

def comprehensive_classification_evaluation(y_true, y_pred, y_pred_proba=None, class_names=None):
    """Comprehensive evaluation for classification models"""
    
    print("=== CLASSIFICATION EVALUATION ===")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision (weighted): {precision:.3f}")
    print(f"Recall (weighted): {recall:.3f}")
    print(f"F1-Score (weighted): {f1:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 5))
    
    # Plot confusion matrix
    plt.subplot(1, 3, 1)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        # ROC Curve (for binary classification)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.subplot(1, 3, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
        
        plt.subplot(1, 3, 3)
        plt.plot(recall_vals, precision_vals, color='blue', lw=2, 
                label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
    
    plt.tight_layout()
    plt.show()
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Example usage
log_reg_final = LogisticRegression(random_state=42)
log_reg_final.fit(X_train_scaled, y_train)
y_pred_final = log_reg_final.predict(X_test_scaled)
y_pred_proba_final = log_reg_final.predict_proba(X_test_scaled)

metrics = comprehensive_classification_evaluation(
    y_test, y_pred_final, y_pred_proba_final, 
    class_names=['Class 0', 'Class 1']
)
```

## Model Comparison

---

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Compare multiple classification algorithms
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

# Evaluate all classifiers
results = []
cv_results = {}

for name, clf in classifiers.items():
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_results[name] = cv_scores
    
    # Fit and predict
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'CV Accuracy Mean': cv_scores.mean(),
        'CV Accuracy Std': cv_scores.std(),
        'Test Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Display results
results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
print("Model Comparison:")
print(results_df.round(4))

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Test Accuracy
axes[0, 0].barh(results_df['Model'], results_df['Test Accuracy'])
axes[0, 0].set_xlabel('Test Accuracy')
axes[0, 0].set_title('Model Performance - Accuracy')

# F1-Score
axes[0, 1].barh(results_df['Model'], results_df['F1-Score'])
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_title('Model Performance - F1-Score')

# Cross-validation boxplot
axes[1, 0].boxplot([cv_results[model] for model in results_df['Model']], 
                   labels=results_df['Model'])
axes[1, 0].set_ylabel('CV Accuracy')
axes[1, 0].set_title('Cross-Validation Scores Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)

# Precision vs Recall scatter
axes[1, 1].scatter(results_df['Recall'], results_df['Precision'], s=100, alpha=0.7)
for i, model in enumerate(results_df['Model']):
    axes[1, 1].annotate(model, (results_df.iloc[i]['Recall'], results_df.iloc[i]['Precision']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_title('Precision vs Recall')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Handling Imbalanced Data

---

```python
from sklearn.utils import resample
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Create imbalanced dataset
X_imb, y_imb = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                  n_informative=2, n_clusters_per_class=1, 
                                  weights=[0.9, 0.1], random_state=42)

print("Original class distribution:")
unique, counts = np.unique(y_imb, return_counts=True)
print(dict(zip(unique, counts)))

# Split data
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb)

# Method 1: Class weights
clf_weighted = LogisticRegression(class_weight='balanced', random_state=42)
clf_weighted.fit(X_train_imb, y_train_imb)
y_pred_weighted = clf_weighted.predict(X_test_imb)

# Method 2: SMOTE (Synthetic Minority Oversampling)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_imb, y_train_imb)

clf_smote = LogisticRegression(random_state=42)
clf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = clf_smote.predict(X_test_imb)

# Method 3: Random undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train_imb, y_train_imb)

clf_under = LogisticRegression(random_state=42)
clf_under.fit(X_train_under, y_train_under)
y_pred_under = clf_under.predict(X_test_imb)

# Compare results
methods = ['Original', 'Weighted', 'SMOTE', 'Undersampling']
predictions = [LogisticRegression(random_state=42).fit(X_train_imb, y_train_imb).predict(X_test_imb),
              y_pred_weighted, y_pred_smote, y_pred_under]

print("\nComparison of methods for imbalanced data:")
for method, y_pred in zip(methods, predictions):
    print(f"\n{method} Results:")
    print(f"Accuracy: {accuracy_score(y_test_imb, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_test_imb, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test_imb, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test_imb, y_pred):.3f}")
```

## Key Takeaways

---

1. **Logistic Regression** - Simple, interpretable, good baseline untuk binary dan multi-class
2. **Decision Trees** - Highly interpretable, can handle non-linear relationships, prone to overfitting
3. **SVM** - Powerful untuk high-dimensional data, kernel trick untuk non-linear problems
4. **KNN** - Simple, non-parametric, sensitive to curse of dimensionality
5. **Naive Bayes** - Fast, works well dengan small datasets dan text classification
6. **Evaluation** - Accuracy tidak selalu cukup, perhatikan precision, recall, F1-score
7. **Imbalanced Data** - Gunakan appropriate metrics dan sampling techniques

Classification adalah core skill dalam supervised learning. Konsep-konsep seperti decision boundaries, probability estimation, dan evaluation metrics akan sangat berguna saat kamu lanjut ke ensemble methods dan deep learning!