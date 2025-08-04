# [ai] #03 - ML Supervised Learning - Regression ![ml](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ml.png)

## Understanding Regression

---

Regression adalah tipe supervised learning yang memprediksi nilai kontinyu (angka). Bayangkan kamu mau prediksi harga rumah, gaji, atau suhu - semua ini adalah regression problems. Target variable-nya berupa angka yang bisa berapa saja dalam range tertentu, bukan kategori yang terbatas.

Bedanya dengan classification: regression prediksi "berapa banyak", sedangkan classification prediksi "kategori apa".

## Linear Regression

---

Linear regression adalah algoritma paling fundamental dalam ML. Konsepnya sederhana: cari garis lurus yang paling pas untuk menggambarkan hubungan antara input dan output.

### Simple Linear Regression

Satu input variable, satu output variable.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature: size rumah (dalam 100 m²)
y = 2.5 * X.ravel() + 1 + np.random.randn(100) * 0.5  # Target: harga (dalam juta)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('House Size (100 m²)')
plt.ylabel('Price (Million)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
```

### Multiple Linear Regression

Multiple input variables, satu output variable.

```python
# Load sample dataset
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Create multiple features
np.random.seed(42)
n_samples = 500

# Features: size, location_score, age, bedrooms
X_multi = np.column_stack([
    np.random.rand(n_samples) * 200,      # size
    np.random.rand(n_samples) * 10,       # location score
    np.random.rand(n_samples) * 50,       # age
    np.random.randint(1, 6, n_samples)    # bedrooms
])

# Target with multiple feature dependency
y_multi = (X_multi[:, 0] * 0.5 +          # size effect
           X_multi[:, 1] * 2 +             # location effect  
           X_multi[:, 2] * (-0.1) +        # age effect (negative)
           X_multi[:, 3] * 5 +             # bedroom effect
           10 + np.random.randn(n_samples) * 2)

# Convert to DataFrame for easier handling
feature_names = ['size', 'location_score', 'age', 'bedrooms']
df = pd.DataFrame(X_multi, columns=feature_names)
df['price'] = y_multi

# Split and scale data
X = df[feature_names]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
multi_model = LinearRegression()
multi_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_multi = multi_model.predict(X_test_scaled)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': multi_model.coef_,
    'abs_coefficient': np.abs(multi_model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("Feature Importance:")
print(feature_importance)
print(f"\nR² Score: {r2_score(y_test, y_pred_multi):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_multi)):.3f}")
```

## Polynomial Regression

---

Ketika hubungan antara input dan output tidak linear, kita bisa gunakan polynomial features untuk capture non-linearity.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Generate non-linear data
X_poly = np.linspace(0, 4, 100).reshape(-1, 1)
y_poly = 0.5 * X_poly.ravel()**3 - 2 * X_poly.ravel()**2 + X_poly.ravel() + np.random.randn(100) * 2

# Split data
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y_poly, test_size=0.2, random_state=42)

# Create polynomial models with different degrees
degrees = [1, 2, 3, 4, 8]
plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees):
    # Create polynomial pipeline
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Train
    poly_model.fit(X_train_poly, y_train_poly)
    
    # Predict
    y_pred_poly = poly_model.predict(X_test_poly)
    
    # Plot
    plt.subplot(2, 3, i+1)
    
    # Create smooth curve for visualization
    X_plot = np.linspace(0, 4, 100).reshape(-1, 1)
    y_plot = poly_model.predict(X_plot)
    
    plt.scatter(X_test_poly, y_test_poly, alpha=0.6, label='Actual')
    plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}\nR² = {r2_score(y_test_poly, y_pred_poly):.3f}')
    plt.legend()

plt.tight_layout()
plt.show()
```

## Regularization

---

Regularization adalah teknik untuk prevent overfitting dengan menambahkan penalty term pada cost function. Ada dua jenis utama:

### Ridge Regression (L2 Regularization)

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Generate data with many features (some irrelevant)
np.random.seed(42)
n_samples, n_features = 100, 20
X_ridge = np.random.randn(n_samples, n_features)
true_coef = np.zeros(n_features)
true_coef[:5] = [1.5, -2, 0.8, -1.2, 0.5]  # Only first 5 features are relevant
y_ridge = X_ridge @ true_coef + np.random.randn(n_samples) * 0.1

X_train_ridge, X_test_ridge, y_train_ridge, y_test_ridge = train_test_split(
    X_ridge, y_ridge, test_size=0.2, random_state=42)

# Compare different alpha values
alphas = [0.01, 0.1, 1, 10, 100]
ridge_scores = []

plt.figure(figsize=(12, 8))

for i, alpha in enumerate(alphas):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_ridge, y_train_ridge)
    
    score = ridge.score(X_test_ridge, y_test_ridge)
    ridge_scores.append(score)
    
    # Plot coefficients
    plt.subplot(2, 3, i+1)
    plt.bar(range(len(ridge.coef_)), ridge.coef_)
    plt.title(f'Ridge α={alpha}, R²={score:.3f}')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    
    # Highlight true non-zero coefficients
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Find optimal alpha using cross-validation
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': np.logspace(-3, 2, 50)}
ridge_cv = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
ridge_cv.fit(X_train_ridge, y_train_ridge)

print(f"Best alpha: {ridge_cv.best_params_['alpha']:.4f}")
print(f"Best CV score: {ridge_cv.best_score_:.3f}")
```

### Lasso Regression (L1 Regularization)

```python
from sklearn.linear_model import Lasso

# Lasso for feature selection
alphas_lasso = [0.01, 0.1, 1, 10]
lasso_scores = []

plt.figure(figsize=(12, 8))

for i, alpha in enumerate(alphas_lasso):
    lasso = Lasso(alpha=alpha, max_iter=1000)
    lasso.fit(X_train_ridge, y_train_ridge)
    
    score = lasso.score(X_test_ridge, y_test_ridge)
    lasso_scores.append(score)
    
    # Plot coefficients
    plt.subplot(2, 2, i+1)
    plt.bar(range(len(lasso.coef_)), lasso.coef_)
    plt.title(f'Lasso α={alpha}, R²={score:.3f}')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    
    # Count non-zero coefficients
    non_zero = np.sum(lasso.coef_ != 0)
    plt.text(0.7, 0.9, f'Non-zero: {non_zero}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat'))

plt.tight_layout()
plt.show()

# Feature selection with Lasso
lasso_selector = Lasso(alpha=0.1)
lasso_selector.fit(X_train_ridge, y_train_ridge)

selected_features = np.where(lasso_selector.coef_ != 0)[0]
print(f"Selected features: {selected_features}")
print(f"True relevant features: [0, 1, 2, 3, 4]")
```

### Elastic Net (L1 + L2)

```python
from sklearn.linear_model import ElasticNet

# Elastic Net combines Ridge and Lasso
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
elastic_net.fit(X_train_ridge, y_train_ridge)

en_score = elastic_net.score(X_test_ridge, y_test_ridge)
print(f"Elastic Net R² score: {en_score:.3f}")

# Visualize coefficients comparison
plt.figure(figsize=(12, 5))

methods = ['Ridge', 'Lasso', 'Elastic Net']
models = [
    Ridge(alpha=0.1).fit(X_train_ridge, y_train_ridge),
    Lasso(alpha=0.1).fit(X_train_ridge, y_train_ridge),
    elastic_net
]

for i, (method, model) in enumerate(zip(methods, models)):
    plt.subplot(1, 3, i+1)
    plt.bar(range(len(model.coef_)), model.coef_)
    plt.title(f'{method} Coefficients')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    
    non_zero = np.sum(model.coef_ != 0)
    score = model.score(X_test_ridge, y_test_ridge)
    plt.text(0.05, 0.95, f'Non-zero: {non_zero}\nR²: {score:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor='lightblue'))

plt.tight_layout()
plt.show()
```

## Advanced Regression Techniques

---

### Support Vector Regression (SVR)

```python
from sklearn.svm import SVR

# Generate non-linear data
X_svr = np.sort(5 * np.random.rand(100, 1), axis=0)
y_svr = np.sin(X_svr).ravel() + np.random.randn(100) * 0.1

# Try different kernels
kernels = ['linear', 'poly', 'rbf']
plt.figure(figsize=(15, 5))

for i, kernel in enumerate(kernels):
    svr = SVR(kernel=kernel, gamma='scale')
    svr.fit(X_svr, y_svr)
    
    # Predict on test range
    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    y_plot = svr.predict(X_plot)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X_svr, y_svr, alpha=0.6, label='Training data')
    plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'SVR {kernel}')
    plt.title(f'SVR with {kernel} kernel')
    plt.legend()

plt.tight_layout()
plt.show()
```

### Decision Tree Regression

```python
from sklearn.tree import DecisionTreeRegressor

# Decision trees can capture non-linear patterns
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_svr, y_svr)

y_tree_pred = tree_reg.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_svr, y_svr, alpha=0.6, label='Training data')
plt.plot(X_plot, y_tree_pred, color='red', linewidth=2, label='Decision Tree')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()

# Visualize decision tree (simplified)
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(tree_reg, max_depth=2, filled=True, feature_names=['X'])
plt.title('Decision Tree Structure')
plt.show()
```

## Evaluation Metrics

---

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(y_true, y_pred, model_name="Model"):
    """Comprehensive evaluation of regression model"""
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"=== {model_name} Evaluation ===")
    print(f"MAE (Mean Absolute Error): {mae:.3f}")
    print(f"MSE (Mean Squared Error): {mse:.3f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # Residual analysis
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 4))
    
    # Residuals vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    # QQ plot
    from scipy import stats
    plt.subplot(1, 3, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # Histogram of residuals
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=20, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# Example usage
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_eval = linear_reg.predict(X_test)

metrics = evaluate_regression(y_test, y_pred_eval, "Linear Regression")
```

## Model Comparison

---

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Compare multiple regression models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'SVR': SVR(kernel='rbf'),
    'Decision Tree': DecisionTreeRegressor(max_depth=5),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# Evaluate all models
results = []

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Fit and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'Model': name,
        'CV R² Mean': cv_scores.mean(),
        'CV R² Std': cv_scores.std(),
        'Test R²': r2,
        'Test RMSE': rmse
    })

# Display results
results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
print("Model Comparison:")
print(results_df.round(4))

# Visualize results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(results_df['Model'], results_df['Test R²'])
plt.xlabel('R² Score')
plt.title('Model Performance (R² Score)')

plt.subplot(1, 2, 2)
plt.barh(results_df['Model'], results_df['Test RMSE'])
plt.xlabel('RMSE')
plt.title('Model Performance (RMSE)')

plt.tight_layout()
plt.show()
```

## Key Takeaways

---

1. **Linear Regression** adalah foundation - simple tapi powerful untuk banyak kasus
2. **Polynomial Features** bisa capture non-linearity tanpa ganti algoritma
3. **Regularization** (Ridge/Lasso) essential untuk prevent overfitting dan feature selection
4. **Different algorithms** punya strengths masing-masing - experiment dan compare
5. **Evaluation** harus comprehensive - jangan cuma liat R², perhatikan juga residuals
6. **Feature scaling** penting untuk algorithms yang distance-based

Regression adalah stepping stone yang bagus untuk understanding supervised learning. Konsep-konsep seperti overfitting, regularization, dan evaluation metrics akan kamu pakai terus di ML. Next, kita akan explore classification problems!