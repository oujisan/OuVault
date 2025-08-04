# [ai] #07 - ML Model Evaluation and Selection

![ml](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ml.png)

## Cross-Validation

---

Cross-validation adalah teknik untuk mengevaluasi performa model dengan cara membagi data menjadi beberapa fold dan melakukan training-testing secara berulang. Ini membantu kita mendapatkan estimasi performa yang lebih reliable.

### K-Fold Cross-Validation

Teknik paling umum dimana data dibagi menjadi k bagian (fold). Model ditraining pada k-1 fold dan ditest pada 1 fold sisanya, diulang k kali.

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

# Load data
X, y = load_iris(return_X_y=True)

# Model
rf = RandomForestClassifier(random_state=42)

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Stratified K-Fold

Mempertahankan proporsi kelas di setiap fold, cocok untuk dataset yang tidak seimbang.

```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(rf, X, y, cv=skfold, scoring='accuracy')

print(f"Stratified CV Scores: {stratified_scores}")
print(f"Mean: {stratified_scores.mean():.3f}")
```

### Leave-One-Out Cross-Validation (LOOCV)

Setiap observasi menjadi test set sekali, cocok untuk dataset kecil tapi computationally expensive.

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
loo_scores = cross_val_score(rf, X, y, cv=loo, scoring='accuracy')
print(f"LOOCV Mean Score: {loo_scores.mean():.3f}")
```

## Model Evaluation Metrics

---

Pemilihan metric yang tepat sangat penting karena setiap metric mengukur aspek performa yang berbeda.

### Classification Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### ROC Curve dan AUC

Untuk binary classification, ROC curve menunjukkan trade-off antara true positive rate dan false positive rate.

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Binary classification data
X_binary, y_binary = make_classification(n_samples=1000, n_features=20, 
                                       n_classes=2, random_state=42)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42)

# Train model
lr = LogisticRegression()
lr.fit(X_train_bin, y_train_bin)

# Get probabilities
y_proba = lr.predict_proba(X_test_bin)[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print(f"AUC Score: {roc_auc_score(y_test_bin, y_proba):.3f}")
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

# Train model
lr_reg = LinearRegression()
lr_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = lr_reg.predict(X_test_reg)

# Calculate metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
```

## Hyperparameter Tuning

---

Hyperparameter adalah parameter yang tidak dipelajari dari data tapi harus diset sebelum training. Tuning yang tepat bisa significantly meningkatkan performa model.

### Grid Search

Mencoba semua kombinasi hyperparameter yang didefinisikan.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
rf_grid = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_grid,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Test with best model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test score with best model: {test_score:.3f}")
```

### Random Search

Lebih efisien untuk high-dimensional parameter space, sampling random combinations.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.3f}")
```

### Bayesian Optimization

Menggunakan probabilistic model untuk mencari hyperparameter optimal lebih efisien.

```python
# Note: Requires scikit-optimize: pip install scikit-optimize
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    
    # Define search space
    search_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Categorical([3, 5, 7, 10, None]),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['sqrt', 'log2', None])
    }
    
    # Bayesian optimization
    bayes_search = BayesSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        search_spaces=search_space,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    bayes_search.fit(X_train, y_train)
    
    print(f"Bayesian Best parameters: {bayes_search.best_params_}")
    print(f"Bayesian Best score: {bayes_search.best_score_:.3f}")
    
except ImportError:
    print("scikit-optimize not installed. Install with: pip install scikit-optimize")
```

## Bias-Variance Tradeoff

---

Understanding bias-variance tradeoff sangat penting untuk memahami mengapa model overfit atau underfit.

### Bias vs Variance Analysis

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt

def bias_variance_analysis(X, y, model, n_trials=100):
    """
    Analyze bias and variance of a model
    """
    n_samples = len(X)
    predictions = []
    
    for trial in range(n_trials):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        
        # Train model
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_bootstrap, y_bootstrap)
        
        # Predict on original data
        pred = model_copy.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate bias and variance
    mean_prediction = np.mean(predictions, axis=0)
    bias_squared = np.mean((mean_prediction - y) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias_squared, variance

# Generate synthetic data
np.random.seed(42)
X_bias_var = np.linspace(0, 1, 100).reshape(-1, 1)
y_true = 1.5 * X_bias_var.ravel() + 0.5 * np.sin(2 * np.pi * X_bias_var.ravel())
y_bias_var = y_true + 0.1 * np.random.randn(100)

# Compare different models
models = {
    'High Bias (Shallow Tree)': DecisionTreeRegressor(max_depth=2, random_state=42),
    'Balanced': DecisionTreeRegressor(max_depth=5, random_state=42),
    'High Variance (Deep Tree)': DecisionTreeRegressor(max_depth=None, random_state=42),
    'Bagging (Reduced Variance)': BaggingRegressor(
        DecisionTreeRegressor(max_depth=None, random_state=42), 
        n_estimators=50, random_state=42
    )
}

print("Bias-Variance Analysis:")
print("-" * 40)

for name, model in models.items():
    bias_sq, variance = bias_variance_analysis(X_bias_var, y_bias_var, model)
    print(f"{name}:")
    print(f"  Bias²: {bias_sq:.4f}")
    print(f"  Variance: {variance:.4f}")
    print(f"  Total: {bias_sq + variance:.4f}")
    print()
```

### Learning Curves

Membantu diagnose apakah model suffering from high bias atau high variance.

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """
    Plot learning curves untuk diagnose bias-variance
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Error')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Error')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot learning curves for different models
for name, model in models.items():
    if 'Bagging' not in name:  # Skip bagging for clarity
        plot_learning_curve(model, X_bias_var, y_bias_var, f"Learning Curve - {name}")
```

## Model Selection Strategies

---

### Validation Curves

Membantu memilih nilai hyperparameter optimal dengan melihat performa training dan validation.

```python
from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, X, y, param_name, param_range, title="Validation Curve"):
    """
    Plot validation curve untuk hyperparameter tuning
    """
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Example: Validation curve for max_depth
max_depth_range = range(1, 21)
plot_validation_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, 
    param_name='max_depth', 
    param_range=max_depth_range,
    title='Validation Curve - Max Depth'
)
```

### Model Comparison Framework

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def compare_models(X, y, models_dict, cv=5):
    """
    Compare multiple models using cross-validation
    """
    results = {}
    
    for name, model in models_dict.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    return results

# Define models to compare
models_to_compare = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'K-Neighbors': KNeighborsClassifier()
}

# Compare models
comparison_results = compare_models(X, y, models_to_compare)

# Display results
print("Model Comparison Results:")
print("-" * 50)
for name, result in comparison_results.items():
    print(f"{name}:")
    print(f"  Mean Accuracy: {result['mean']:.3f} (+/- {result['std'] * 2:.3f})")
    print()

# Visualize comparison
model_names = list(comparison_results.keys())
mean_scores = [comparison_results[name]['mean'] for name in model_names]
std_scores = [comparison_results[name]['std'] for name in model_names]

plt.figure(figsize=(12, 6))
plt.bar(model_names, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Best Practices dan Tips

---

### 1. Data Leakage Prevention

```python
# BAD: Feature scaling before split
from sklearn.preprocessing import StandardScaler

# Don't do this - causes data leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Using all data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# GOOD: Feature scaling after split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_test_scaled = scaler.transform(X_test)  # Transform test data
```

### 2. Nested Cross-Validation

Untuk unbiased estimate ketika melakukan hyperparameter tuning.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

def nested_cross_validation(X, y, model, param_grid, outer_cv=5, inner_cv=3):
    """
    Nested cross-validation untuk unbiased model evaluation
    """
    outer_scores = []
    
    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for train_idx, test_idx in outer_kfold.split(X):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Inner loop: hyperparameter tuning
        inner_kfold = KFold(n_splits=inner_cv, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=inner_kfold, scoring='accuracy'
        )
        grid_search.fit(X_train_outer, y_train_outer)
        
        # Outer loop: model evaluation
        best_model = grid_search.best_estimator_
        score = best_model.score(X_test_outer, y_test_outer)
        outer_scores.append(score)
    
    return np.array(outer_scores)

# Example usage
param_grid_simple = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

nested_scores = nested_cross_validation(
    X, y, 
    RandomForestClassifier(random_state=42), 
    param_grid_simple
)

print(f"Nested CV Scores: {nested_scores}")
print(f"Mean: {nested_scores.mean():.3f} (+/- {nested_scores.std() * 2:.3f})")
```

### 3. Model Selection Decision Tree

```python
def model_selection_guide(dataset_size, feature_count, problem_type):
    """
    Simple guide untuk model selection
    """
    print(f"Dataset: {dataset_size} samples, {feature_count} features")
    print(f"Problem: {problem_type}")
    print("-" * 40)
    
    if problem_type == "classification":
        if dataset_size < 1000:
            if feature_count < 10:
                print("Recommended: Logistic Regression, SVM")
            else:
                print("Recommended: Random Forest, Naive Bayes")
        else:
            if feature_count < 100:
                print("Recommended: Random Forest, Gradient Boosting")
            else:
                print("Recommended: Random Forest, Linear models with regularization")
    
    elif problem_type == "regression":
        if dataset_size < 1000:
            if feature_count < 10:
                print("Recommended: Linear Regression, SVR")
            else:
                print("Recommended: Random Forest, Ridge Regression")
        else:
            if feature_count < 100:
                print("Recommended: Random Forest, Gradient Boosting")
            else:
                print("Recommended: Linear models with regularization")

# Example usage
model_selection_guide(1500, 20, "classification")
```

Model evaluation dan selection adalah fondasi yang sangat penting dalam machine learning. Dengan memahami berbagai teknik cross-validation, metrics, dan hyperparameter tuning, kamu bisa memastikan model yang dibangun benar-benar reliable dan performant. Next, kita akan explore advanced topics yang akan melengkapi journey kamu di machine learning!