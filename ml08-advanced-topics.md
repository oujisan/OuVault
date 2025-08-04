# [ai] #08 - ML Advanced Topics

![ml](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ml.png)

## Feature Selection

---

Feature selection adalah proses memilih subset features yang paling relevan untuk model. Ini membantu mengurangi overfitting, meningkatkan interpretability, dan mempercepat training.

### Filter Methods

Menggunakan statistical measures untuk mengevaluasi relevance setiap feature independent dari model.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.feature_selection import (SelectKBest, chi2, f_classif, 
                                     mutual_info_classif, VarianceThreshold)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=feature_names)
print(f"Dataset shape: {df.shape}")

# 1. Variance Threshold - Remove low variance features
variance_selector = VarianceThreshold(threshold=0.1)
X_variance_selected = variance_selector.fit_transform(X)
selected_features_var = feature_names[variance_selector.get_support()]

print(f"Features after variance threshold: {X_variance_selected.shape[1]}")
print(f"Removed features: {len(feature_names) - len(selected_features_var)}")

# 2. Univariate Statistical Tests
# Chi-square test (for categorical targets)
chi2_selector = SelectKBest(score_func=chi2, k=10)
X_chi2 = chi2_selector.fit_transform(X, y)
chi2_scores = chi2_selector.scores_
chi2_features = feature_names[chi2_selector.get_support()]

# F-test (ANOVA F-value)
f_selector = SelectKBest(score_func=f_classif, k=10)
X_f_test = f_selector.fit_transform(X, y)
f_scores = f_selector.scores_
f_features = feature_names[f_selector.get_support()]

# Mutual Information
mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_mi = mi_selector.fit_transform(X, y)
mi_scores = mi_selector.scores_
mi_features = feature_names[mi_selector.get_support()]

# Visualize feature scores
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chi-square scores
axes[0].bar(range(len(chi2_scores)), chi2_scores)
axes[0].set_title('Chi-square Scores')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Score')

# F-test scores
axes[1].bar(range(len(f_scores)), f_scores)
axes[1].set_title('F-test Scores')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Score')

# Mutual Information scores
axes[2].bar(range(len(mi_scores)), mi_scores)
axes[2].set_title('Mutual Information Scores')
axes[2].set_xlabel('Features')
axes[2].set_ylabel('Score')

plt.tight_layout()
plt.show()

print(f"Top 10 features by Chi-square: {chi2_features}")
print(f"Top 10 features by F-test: {f_features}")
print(f"Top 10 features by Mutual Info: {mi_features}")
```

### Wrapper Methods

Menggunakan model untuk mengevaluasi subset features dengan mencoba berbagai kombinasi.

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Recursive Feature Elimination (RFE)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe_selector = RFE(estimator=rf_model, n_features_to_select=10, step=1)
X_rfe = rfe_selector.fit_transform(X, y)
rfe_features = feature_names[rfe_selector.get_support()]

print(f"RFE selected features: {rfe_features}")
print(f"Feature ranking: {rfe_selector.ranking_}")

# RFE with Cross-Validation
rfecv_selector = RFECV(estimator=rf_model, step=1, cv=5, scoring='accuracy')
X_rfecv = rfecv_selector.fit_transform(X, y)
rfecv_features = feature_names[rfecv_selector.get_support()]

print(f"RFECV selected {rfecv_selector.n_features_} features")
print(f"Optimal number of features: {rfecv_selector.n_features_}")

# Plot number of features vs cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv_selector.cv_results_['mean_test_score']) + 1),
         rfecv_selector.cv_results_['mean_test_score'], marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross Validation Accuracy')
plt.title('RFECV: Feature Selection')
plt.axvline(x=rfecv_selector.n_features_, color='red', linestyle='--', 
            label=f'Optimal: {rfecv_selector.n_features_} features')
plt.legend()
plt.grid(True)
plt.show()
```

### Embedded Methods

Model secara otomatis melakukan feature selection sebagai bagian dari training process.

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 1. L1 Regularization (Lasso)
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=2000)
lasso_cv.fit(X, y)

# Select features using Lasso coefficients
lasso_selector = SelectFromModel(lasso_cv, prefit=True)
X_lasso = lasso_selector.transform(X)
lasso_features = feature_names[lasso_selector.get_support()]

print(f"Lasso selected {len(lasso_features)} features")
print(f"Selected features: {lasso_features}")

# Visualize Lasso coefficients
plt.figure(figsize=(12, 8))
coef_dict = dict(zip(feature_names, lasso_cv.coef_))
coef_series = pd.Series(coef_dict).sort_values(key=abs, ascending=False)

# Plot only non-zero coefficients
non_zero_coef = coef_series[coef_series != 0]
plt.barh(range(len(non_zero_coef)), non_zero_coef.values)
plt.yticks(range(len(non_zero_coef)), non_zero_coef.index)
plt.xlabel('Coefficient Value')
plt.title('Lasso Regression Coefficients (Non-zero)')
plt.tight_layout()
plt.show()

# 2. Tree-based Feature Importance
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X, y)

# Select features based on importance
importance_selector = SelectFromModel(rf_selector, prefit=True)
X_importance = importance_selector.transform(X)
importance_features = feature_names[importance_selector.get_support()]

print(f"Tree-based selected {len(importance_features)} features")

# Visualize feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance (Top 15)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Forward/Backward Selection

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

# Forward Selection
forward_selector = SequentialFeatureSelector(
    LogisticRegression(random_state=42, max_iter=1000),
    n_features_to_select=10,
    direction='forward',
    cv=5,
    scoring='accuracy'
)

X_forward = forward_selector.fit_transform(X, y)
forward_features = feature_names[forward_selector.get_support()]

print(f"Forward selection features: {forward_features}")

# Backward Selection
backward_selector = SequentialFeatureSelector(
    LogisticRegression(random_state=42, max_iter=1000),
    n_features_to_select=10,
    direction='backward',
    cv=5,
    scoring='accuracy'
)

X_backward = backward_selector.fit_transform(X, y)
backward_features = feature_names[backward_selector.get_support()]

print(f"Backward selection features: {backward_features}")
```

## Machine Learning Pipelines

---

Pipelines membantu mengorganisir preprocessing steps dan model training dalam satu workflow yang bersih dan reproducible.

### Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Create basic pipeline
basic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit and evaluate
basic_pipeline.fit(X_train, y_train)
pipeline_score = basic_pipeline.score(X_test, y_test)

print(f"Basic Pipeline Accuracy: {pipeline_score:.3f}")

# Cross-validation with pipeline
cv_scores = cross_val_score(basic_pipeline, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### Advanced Pipeline with Feature Selection

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Create comprehensive pipeline
comprehensive_pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif, k=15)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

comprehensive_pipeline.fit(X_train, y_train)
comp_score = comprehensive_pipeline.score(X_test, y_test)

print(f"Comprehensive Pipeline Accuracy: {comp_score:.3f}")

# Access pipeline components
selected_features_mask = comprehensive_pipeline.named_steps['feature_selection'].get_support()
selected_feature_names = feature_names[selected_features_mask]
print(f"Selected features: {len(selected_feature_names)}")
```

### Pipeline with GridSearch

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid for pipeline
pipeline_params = {
    'feature_selection__k': [10, 15, 20],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7, None]
}

# Grid search with pipeline
pipeline_grid = GridSearchCV(
    comprehensive_pipeline,
    pipeline_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

pipeline_grid.fit(X_train, y_train)

print(f"Best pipeline parameters: {pipeline_grid.best_params_}")
print(f"Best CV score: {pipeline_grid.best_score_:.3f}")
print(f"Test score: {pipeline_grid.score(X_test, y_test):.3f}")
```

### Column Transformer for Mixed Data Types

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Create mixed data example
np.random.seed(42)
n_samples = 1000

mixed_data = pd.DataFrame({
    'numeric_1': np.random.randn(n_samples),
    'numeric_2': np.random.randn(n_samples) * 10 + 5,
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y'], n_samples),
    'numeric_with_missing': np.where(np.random.rand(n_samples) < 0.1, 
                                   np.nan, np.random.randn(n_samples))
})

# Create target
mixed_target = (mixed_data['numeric_1'] + 
                mixed_data['numeric_2'] * 0.5 + 
                np.random.randn(n_samples) * 0.1 > 2).astype(int)

# Define column types
numeric_features = ['numeric_1', 'numeric_2', 'numeric_with_missing']
categorical_features = ['categorical_1', 'categorical_2']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ]), categorical_features)
    ]
)

# Complete pipeline
mixed_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split and fit
X_mixed_train, X_mixed_test, y_mixed_train, y_mixed_test = train_test_split(
    mixed_data, mixed_target, test_size=0.3, random_state=42
)

mixed_pipeline.fit(X_mixed_train, y_mixed_train)
mixed_score = mixed_pipeline.score(X_mixed_test, y_mixed_test)

print(f"Mixed Data Pipeline Accuracy: {mixed_score:.3f}")
```

## Handling Real-World Challenges

---

### Imbalanced Datasets

```python
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# Create imbalanced dataset
X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20, n_classes=2, 
    weights=[0.9, 0.1], random_state=42
)

print(f"Original class distribution: {Counter(y_imb)}")

X_imb_train, X_imb_test, y_imb_train, y_imb_test = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
)

# 1. Without handling imbalance
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_imb_train, y_imb_train)
y_pred_baseline = rf_baseline.predict(X_imb_test)

print("Baseline (No balancing):")
print(classification_report(y_imb_test, y_pred_baseline))

# 2. Random Oversampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_imb_train, y_imb_train)
print(f"After Random Oversampling: {Counter(y_ros)}")

rf_ros = RandomForestClassifier(random_state=42)
rf_ros.fit(X_ros, y_ros)
y_pred_ros = rf_ros.predict(X_imb_test)

print("Random Oversampling:")
print(classification_report(y_imb_test, y_pred_ros))

# 3. SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_imb_train, y_imb_train)
print(f"After SMOTE: {Counter(y_smote)}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_smote, y_smote)
y_pred_smote = rf_smote.predict(X_imb_test)

print("SMOTE:")
print(classification_report(y_imb_test, y_pred_smote))

# 4. Class Weight Balancing
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_imb_train, y_imb_train)
y_pred_weighted = rf_weighted.predict(X_imb_test)

print("Class Weight Balancing:")
print(classification_report(y_imb_test, y_pred_weighted))

# 5. Pipeline with SMOTE
smote_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

smote_pipeline.fit(X_imb_train, y_imb_train)
y_pred_pipe = smote_pipeline.predict(X_imb_test)

print("SMOTE Pipeline:")
print(classification_report(y_imb_test, y_pred_pipe))
```

### Missing Data Handling

```python
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer

# Create dataset with missing values
np.random.seed(42)
n_samples, n_features = 1000, 5

# Generate complete data
complete_data = np.random.randn(n_samples, n_features)
feature_names_missing = [f'feature_{i}' for i in range(n_features)]

# Introduce missing values
missing_data = complete_data.copy()
for i in range(n_features):
    missing_mask = np.random.rand(n_samples) < 0.15  # 15% missing
    missing_data[missing_mask, i] = np.nan

df_missing = pd.DataFrame(missing_data, columns=feature_names_missing)
print(f"Missing values per column:")
print(df_missing.isnull().sum())

# 1. Simple Imputation Strategies
strategies = ['mean', 'median', 'most_frequent', 'constant']
imputation_results = {}

for strategy in strategies:
    if strategy == 'constant':
        imputer = SimpleImputer(strategy=strategy, fill_value=0)
    else:
        imputer = SimpleImputer(strategy=strategy)
    
    imputed_data = imputer.fit_transform(missing_data)
    
    # Calculate imputation error (RMSE with original complete data)
    mask = ~np.isnan(missing_data)
    rmse = np.sqrt(np.mean((imputed_data[~mask] - complete_data[~mask])**2))
    imputation_results[strategy] = rmse

print("\nImputation RMSE (lower is better):")
for strategy, rmse in imputation_results.items():
    print(f"{strategy}: {rmse:.4f}")

# 2. KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
knn_imputed = knn_imputer.fit_transform(missing_data)

mask = ~np.isnan(missing_data)
knn_rmse = np.sqrt(np.mean((knn_imputed[~mask] - complete_data[~mask])**2))
print(f"KNN Imputation RMSE: {knn_rmse:.4f}")

# 3. Iterative Imputation (MICE)
iterative_imputer = IterativeImputer(random_state=42, max_iter=10)
iterative_imputed = iterative_imputer.fit_transform(missing_data)

iterative_rmse = np.sqrt(np.mean((iterative_imputed[~mask] - complete_data[~mask])**2))
print(f"Iterative Imputation RMSE: {iterative_rmse:.4f}")

# Visualize imputation comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

imputation_methods = {
    'Original (with missing)': missing_data,
    'Mean Imputation': SimpleImputer(strategy='mean').fit_transform(missing_data),
    'Median Imputation': SimpleImputer(strategy='median').fit_transform(missing_data),
    'KNN Imputation': knn_imputed,
    'Iterative Imputation': iterative_imputed,
    'Complete Data': complete_data
}

for idx, (method, data) in enumerate(imputation_methods.items()):
    if idx < len(axes):
        axes[idx].scatter(data[:, 0], data[:, 1], alpha=0.6)
        axes[idx].set_title(method)
        axes[idx].set_xlabel('Feature 0')
        axes[idx].set_ylabel('Feature 1')

plt.tight_layout()
plt.show()
```

### Outlier Detection and Handling

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import scipy.stats as stats

# Generate data with outliers
np.random.seed(42)
n_samples = 300
n_outliers = 30

# Normal data
X_normal = np.random.randn(n_samples - n_outliers, 2)

# Outliers
X_outliers = np.random.uniform(-6, 6, (n_outliers, 2))

# Combine
X_with_outliers = np.vstack([X_normal, X_outliers])
y_true = np.hstack([np.ones(n_samples - n_outliers), -np.ones(n_outliers)])

# 1. Statistical Method (Z-score)
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return np.any(z_scores > threshold, axis=1)

outliers_zscore = detect_outliers_zscore(X_with_outliers)
print(f"Z-score method detected {np.sum(outliers_zscore)} outliers")

# 2. Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers_iso = iso_forest.fit_predict(X_with_outliers) == -1

print(f"Isolation Forest detected {np.sum(outliers_iso)} outliers")

# 3. Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outliers_lof = lof.fit_predict(X_with_outliers) == -1

print(f"Local Outlier Factor detected {np.sum(outliers_lof)} outliers")

# 4. Elliptic Envelope
elliptic_env = EllipticEnvelope(contamination=0.1, random_state=42)
outliers_elliptic = elliptic_env.fit_predict(X_with_outliers) == -1

print(f"Elliptic Envelope detected {np.sum(outliers_elliptic)} outliers")

# Visualize outlier detection results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

methods = [
    ('Z-score', outliers_zscore),
    ('Isolation Forest', outliers_iso),
    ('Local Outlier Factor', outliers_lof),
    ('Elliptic Envelope', outliers_elliptic)
]

for idx, (method, outliers) in enumerate(methods):
    ax = axes[idx // 2, idx % 2]
    
    # Plot normal points
    normal_mask = ~outliers
    ax.scatter(X_with_outliers[normal_mask, 0], X_with_outliers[normal_mask, 1], 
              c='blue', alpha=0.6, label='Normal')
    
    # Plot detected outliers
    ax.scatter(X_with_outliers[outliers, 0], X_with_outliers[outliers, 1], 
              c='red', alpha=0.8, label='Outliers')
    
    ax.set_title(f'{method}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Outlier handling strategies
def handle_outliers_iqr(data, factor=1.5):
    """Remove outliers using IQR method"""
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    return data[mask], mask

def cap_outliers(data, lower_percentile=5, upper_percentile=95):
    """Cap outliers to percentile values"""
    lower_bound = np.percentile(data, lower_percentile, axis=0)
    upper_bound = np.percentile(data, upper_percentile, axis=0)
    
    capped_data = np.clip(data, lower_bound, upper_bound)
    return capped_data

# Apply handling strategies
data_cleaned_iqr, iqr_mask = handle_outliers_iqr(X_with_outliers)
data_capped = cap_outliers(X_with_outliers)

print(f"Original data shape: {X_with_outliers.shape}")
print(f"After IQR cleaning: {data_cleaned_iqr.shape}")
print(f"Removed {np.sum(~iqr_mask)} points")
```

### Feature Engineering Automation

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
import itertools

class AutoFeatureEngineer:
    def __init__(self, max_polynomial_degree=2, max_interaction_degree=2):
        self.max_poly_degree = max_polynomial_degree
        self.max_interaction_degree = max_interaction_degree
        self.poly_features = None
        self.feature_selector = None
        self.feature_names = None
        
    def fit(self, X, y, k_best=None):
        """Fit the feature engineering pipeline"""
        # Store original feature names
        if hasattr(X, 'columns'):
            self.original_features = X.columns.tolist()
            X_array = X.values
        else:
            self.original_features = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        # Generate polynomial features
        self.poly_features = PolynomialFeatures(
            degree=self.max_poly_degree, 
            include_bias=False,
            interaction_only=False
        )
        X_poly = self.poly_features.fit_transform(X_array)
        
        # Feature selection
        if k_best is None:
            k_best = min(50, X_poly.shape[1])  # Limit features
            
        self.feature_selector = SelectKBest(score_func=f_regression, k=k_best)
        X_selected = self.feature_selector.fit_transform(X_poly, y)
        
        # Store selected feature names
        poly_feature_names = self.poly_features.get_feature_names_out(self.original_features)
        selected_mask = self.feature_selector.get_support()
        self.selected_features = poly_feature_names[selected_mask]
        
        return self
    
    def transform(self, X):
        """Transform new data using fitted pipeline"""
        if hasattr(X, 'columns'):
            X_array = X.values
        else:
            X_array = X
            
        X_poly = self.poly_features.transform(X_array)
        X_selected = self.feature_selector.transform(X_poly)
        
        return X_selected
    
    def fit_transform(self, X, y, k_best=None):
        """Fit and transform in one step"""
        return self.fit(X, y, k_best).transform(X)
    
    def get_feature_names(self):
        """Get names of selected features"""
        return self.selected_features

# Example usage
# Generate sample regression data
X_fe, y_fe = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)
feature_names_fe = [f'feature_{i}' for i in range(X_fe.shape[1])]
df_fe = pd.DataFrame(X_fe, columns=feature_names_fe)

# Split data
X_fe_train, X_fe_test, y_fe_train, y_fe_test = train_test_split(
    df_fe, y_fe, test_size=0.3, random_state=42
)

# Auto feature engineering
auto_fe = AutoFeatureEngineer(max_polynomial_degree=2)
X_fe_train_engineered = auto_fe.fit_transform(X_fe_train, y_fe_train, k_best=15)
X_fe_test_engineered = auto_fe.transform(X_fe_test)

print(f"Original features: {X_fe_train.shape[1]}")
print(f"Engineered features: {X_fe_train_engineered.shape[1]}")
print(f"Selected feature names: {auto_fe.get_feature_names()[:10]}")  # Show first 10

# Compare performance
from sklearn.linear_model import Ridge

# Original features
ridge_original = Ridge(random_state=42)
ridge_original.fit(X_fe_train, y_fe_train)
score_original = ridge_original.score(X_fe_test, y_fe_test)

# Engineered features
ridge_engineered = Ridge(random_state=42)
ridge_engineered.fit(X_fe_train_engineered, y_fe_train)
score_engineered = ridge_engineered.score(X_fe_test_engineered, y_fe_test)

print(f"R² with original features: {score_original:.4f}")
print(f"R² with engineered features: {score_engineered:.4f}")
print(f"Improvement: {score_engineered - score_original:.4f}")
```

## Production Considerations

---

### Model Persistence and Versioning

```python
import joblib
import pickle
from datetime import datetime
import json

class ModelManager:
    def __init__(self, model_dir='models/'):
        self.model_dir = model_dir
    
    def save_model(self, model, model_name, metadata=None):
        """Save model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.model_dir}{model_name}_{timestamp}.joblib"
        
        # Save model
        joblib.dump(model, model_filename)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'timestamp': timestamp,
            'model_file': model_filename,
            'model_type': type(model).__name__
        })
        
        metadata_filename = f"{self.model_dir}{model_name}_{timestamp}_metadata.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved: {model_filename}")
        print(f"Metadata saved: {metadata_filename}")
        return model_filename, metadata_filename
    
    def load_model(self, model_filename):
        """Load saved model"""
        return joblib.load(model_filename)
    
    def get_model_info(self, metadata_filename):
        """Load model metadata"""
        with open(metadata_filename, 'r') as f:
            return json.load(f)

# Example usage
import os
os.makedirs('models', exist_ok=True)

model_manager = ModelManager()

# Train a model
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

# Save with metadata
metadata = {
    'accuracy': final_model.score(X_test, y_test),
    'feature_count': X_train.shape[1],
    'training_samples': X_train.shape[0],
    'hyperparameters': final_model.get_params()
}

model_file, metadata_file = model_manager.save_model(
    final_model, 'breast_cancer_classifier', metadata
)
```

### Model Monitoring and Drift Detection

```python
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._calculate_stats(reference_data)
    
    def _calculate_stats(self, data):
        """Calculate reference statistics"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
    
    def detect_drift(self, new_data, method='ks_test'):
        """Detect data drift"""
        drift_results = {}
        
        if method == 'ks_test':
            for i in range(new_data.shape[1]):
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[:, i], 
                    new_data[:, i]
                )
                drift_results[f'feature_{i}'] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < self.threshold
                }
        
        elif method == 'statistical':
            new_stats = self._calculate_stats(new_data)
            for i in range(new_data.shape[1]):
                # Check if mean shifted significantly
                z_score = abs(new_stats['mean'][i] - self.reference_stats['mean'][i]) / self.reference_stats['std'][i]
                drift_results[f'feature_{i}'] = {
                    'z_score': z_score,
                    'drift_detected': z_score > 2  # 2 standard deviations
                }
        
        return drift_results
    
    def performance_monitoring(self, model, new_X, new_y, baseline_score):
        """Monitor model performance degradation"""
        current_score = model.score(new_X, new_y)
        performance_drop = baseline_score - current_score
        
        return {
            'baseline_score': baseline_score,
            'current_score': current_score,
            'performance_drop': performance_drop,
            'significant_drop': performance_drop > 0.05  # 5% drop threshold
        }

# Example drift detection
monitor = ModelMonitor(X_train, threshold=0.05)

# Simulate drift by adding noise to test data
X_drift = X_test + np.random.normal(0, 0.5, X_test.shape)

# Detect drift
drift_results = monitor.detect_drift(X_drift, method='ks_test')

print("Drift Detection Results:")
print("-" * 30)
features_with_drift = 0
for feature, result in drift_results.items():
    if result['drift_detected']:
        features_with_drift += 1
        print(f"{feature}: DRIFT DETECTED (p-value: {result['p_value']:.4f})")

print(f"\nTotal features with drift: {features_with_drift}/{len(drift_results)}")

# Performance monitoring
baseline_score = final_model.score(X_test, y_test)
perf_results = monitor.performance_monitoring(
    final_model, X_drift, y_test, baseline_score
)

print(f"\nPerformance Monitoring:")
print(f"Baseline Score: {perf_results['baseline_score']:.4f}")
print(f"Current Score: {perf_results['current_score']:.4f}")
print(f"Performance Drop: {perf_results['performance_drop']:.4f}")
print(f"Significant Drop: {perf_results['significant_drop']}")
```

### A/B Testing Framework

```python
import scipy.stats as stats
from scipy.stats import ttest_ind

class ABTestFramework:
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha  # Significance level
        self.power = power  # Statistical power
        
    def calculate_sample_size(self, baseline_rate, minimum_effect, alpha=None, power=None):
        """Calculate required sample size for A/B test"""
        if alpha is None:
            alpha = self.alpha
        if power is None:
            power = self.power
            
        # Effect size (Cohen's h for proportions)
        p1 = baseline_rate
        p2 = baseline_rate + minimum_effect
        
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Sample size calculation (simplified)
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
    
    def run_test(self, control_group, treatment_group, metric='conversion'):
        """Run A/B test and return results"""
        if metric == 'conversion':
            # For binary outcomes
            control_successes = np.sum(control_group)
            treatment_successes = np.sum(treatment_group)
            
            control_rate = control_successes / len(control_group)
            treatment_rate = treatment_successes / len(treatment_group)
            
            # Chi-square test for independence
            contingency_table = np.array([
                [control_successes, len(control_group) - control_successes],
                [treatment_successes, len(treatment_group) - treatment_successes]
            ])
            
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            
            # Effect size (relative improvement)
            relative_improvement = (treatment_rate - control_rate) / control_rate * 100
            
        else:
            # For continuous outcomes
            control_rate = np.mean(control_group)
            treatment_rate = np.mean(treatment_group)
            
            # T-test for means
            t_stat, p_value = ttest_ind(control_group, treatment_group)
            
            relative_improvement = (treatment_rate - control_rate) / control_rate * 100
        
        results = {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'relative_improvement': relative_improvement,
            'p_value': p_value,
            'statistically_significant': p_value < self.alpha,
            'sample_size_control': len(control_group),
            'sample_size_treatment': len(treatment_group)
        }
        
        return results
    
    def interpret_results(self, results):
        """Provide interpretation of A/B test results"""
        print("A/B Test Results:")
        print("=" * 40)
        print(f"Control Rate: {results['control_rate']:.4f}")
        print(f"Treatment Rate: {results['treatment_rate']:.4f}")
        print(f"Relative Improvement: {results['relative_improvement']:.2f}%")
        print(f"P-value: {results['p_value']:.6f}")
        print(f"Sample Size Control: {results['sample_size_control']}")
        print(f"Sample Size Treatment: {results['sample_size_treatment']}")
        print(f"Statistically Significant: {results['statistically_significant']}")
        
        if results['statistically_significant']:
            if results['relative_improvement'] > 0:
                print("✅ Treatment performs significantly better than control")
            else:
                print("❌ Treatment performs significantly worse than control")
        else:
            print("⚠️ No statistically significant difference detected")

# Example A/B test simulation
ab_tester = ABTestFramework()

# Simulate model A (control) and model B (treatment) predictions
np.random.seed(42)
n_users = 10000

# Control group (baseline model accuracy: 85%)
control_predictions = np.random.choice([0, 1], size=n_users//2, p=[0.15, 0.85])

# Treatment group (new model accuracy: 87%)
treatment_predictions = np.random.choice([0, 1], size=n_users//2, p=[0.13, 0.87])

# Run A/B test
ab_results = ab_tester.run_test(control_predictions, treatment_predictions)
ab_tester.interpret_results(ab_results)

# Calculate required sample size
required_sample_size = ab_tester.calculate_sample_size(
    baseline_rate=0.85, 
    minimum_effect=0.02  # Want to detect 2% improvement
)
print(f"\nRequired sample size per group: {required_sample_size}")
```

## Model Explainability and Interpretability

---

```python
import shap
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
import matplotlib.pyplot as plt

# Train models for interpretation
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 1. Feature Importance (built-in)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 2. Permutation Importance
perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42
)

perm_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

plt.figure(figsize=(10, 8))
top_perm_features = perm_imp_df.head(15)
plt.barh(range(len(top_perm_features)), top_perm_features['importance_mean'])
plt.yticks(range(len(top_perm_features)), top_perm_features['feature'])
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 3. SHAP Values (if shap is available)
try:
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test[:100])  # Use subset for speed
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[1], X_test[:100], feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.show()
    
    # Feature importance based on SHAP
    shap_importance = np.abs(shap_values[1]).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    
    print("Top 10 features by SHAP importance:")
    print(shap_imp_df.head(10))
    
except ImportError:
    print("SHAP not available. Install with: pip install shap")
except Exception as e:
    print(f"SHAP analysis failed: {e}")

# 4. Local Interpretability - LIME (if available)
try:
    import lime
    import lime.lime_tabular
    
    # Create LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, 
        feature_names=feature_names,
        class_names=['Malignant', 'Benign'],
        mode='classification'
    )
    
    # Explain a single prediction
    instance_idx = 0
    lime_explanation = lime_explainer.explain_instance(
        X_test[instance_idx], 
        rf_model.predict_proba, 
        num_features=10
    )
    
    print(f"\nLIME explanation for instance {instance_idx}:")
    print(f"Prediction: {rf_model.predict([X_test[instance_idx]])[0]}")
    print(f"Probability: {rf_model.predict_proba([X_test[instance_idx]])[0]}")
    
    # Show explanation
    lime_explanation.show_in_notebook(show_table=True)
    
except ImportError:
    print("LIME not available. Install with: pip install lime")
except Exception as e:
    print(f"LIME analysis failed: {e}")
```

## Best Practices Summary

---

### Model Development Checklist

```python
class MLBestPractices:
    """
    Comprehensive checklist for ML project best practices
    """
    
    @staticmethod
    def data_preparation_checklist():
        checklist = {
            "Data Quality": [
                "Check for missing values and handle appropriately",
                "Identify and handle outliers",
                "Verify data types and formats",
                "Check for duplicates",
                "Validate data distributions"
            ],
            "Data Splitting": [
                "Use stratified sampling for imbalanced datasets",
                "Maintain temporal order for time series",
                "Ensure no data leakage between splits",
                "Reserve holdout test set",
                "Use cross-validation for model selection"
            ],
            "Feature Engineering": [
                "Handle categorical variables appropriately",
                "Scale/normalize numerical features",
                "Create meaningful feature interactions",
                "Remove highly correlated features",
                "Apply feature selection techniques"
            ]
        }
        return checklist
    
    @staticmethod
    def model_development_checklist():
        checklist = {
            "Model Selection": [
                "Start with simple baseline models",
                "Try multiple algorithms",
                "Use appropriate metrics for problem type",
                "Consider computational constraints",
                "Validate on multiple CV folds"
            ],
            "Hyperparameter Tuning": [
                "Use nested cross-validation",
                "Don't overfit to validation set",
                "Consider computational budget",
                "Use appropriate search strategies",
                "Document best parameters"
            ],
            "Model Evaluation": [
                "Use multiple evaluation metrics",
                "Analyze confusion matrix/residuals",
                "Check for bias in predictions",
                "Validate on unseen test data",
                "Perform error analysis"
            ]
        }
        return checklist
    
    @staticmethod
    def production_checklist():
        checklist = {
            "Model Deployment": [
                "Version control models and code",
                "Document model assumptions",
                "Create model monitoring system",
                "Plan for model updates",
                "Implement A/B testing framework"
            ],
            "Monitoring": [
                "Track model performance metrics",
                "Monitor for data drift",
                "Set up alerting systems",
                "Log predictions and actuals",
                "Regular model retraining schedule"
            ],
            "Interpretability": [
                "Provide feature importance",
                "Create model documentation",
                "Enable local explanations",
                "Validate model fairness",
                "Communicate limitations clearly"
            ]
        }
        return checklist
    
    @staticmethod
    def print_checklist(checklist_type='all'):
        """Print comprehensive checklist"""
        checklists = {
            'data': MLBestPractices.data_preparation_checklist(),
            'model': MLBestPractices.model_development_checklist(),
            'production': MLBestPractices.production_checklist()
        }
        
        if checklist_type == 'all':
            for name, checklist in checklists.items():
                print(f"\n{name.upper()} PREPARATION CHECKLIST")
                print("=" * 40)
                for category, items in checklist.items():
                    print(f"\n{category}:")
                    for item in items:
                        print(f"  ☐ {item}")
        else:
            if checklist_type in checklists:
                checklist = checklists[checklist_type]
                print(f"\n{checklist_type.upper()} CHECKLIST")
                print("=" * 40)
                for category, items in checklist.items():
                    print(f"\n{category}:")
                    for item in items:
                        print(f"  ☐ {item}")

# Print comprehensive checklist
MLBestPractices.print_checklist('all')
```

### Common Pitfalls and How to Avoid Them

```python
def common_ml_pitfalls():
    """
    Document common ML pitfalls and solutions
    """
    pitfalls = {
        "Data Leakage": {
            "description": "Information from future leaks into training data",
            "examples": [
                "Scaling features before train/test split",
                "Including target-derived features",
                "Using future information in time series"
            ],
            "solutions": [
                "Always split data first, then preprocess",
                "Be careful with feature engineering",
                "Use proper time series validation"
            ]
        },
        
        "Overfitting": {
            "description": "Model learns training data too well, poor generalization",
            "examples": [
                "Too complex model for dataset size",
                "No regularization",
                "Excessive hyperparameter tuning"
            ],
            "solutions": [
                "Use cross-validation",
                "Apply regularization techniques",
                "Reduce model complexity",
                "Get more training data"
            ]
        },
        
        "Selection Bias": {
            "description": "Biased sampling leads to unrepresentative training data",
            "examples": [
                "Cherry-picking favorable time periods",
                "Excluding difficult cases",
                "Sampling bias in data collection"
            ],
            "solutions": [
                "Use random sampling",
                "Ensure representative datasets",
                "Document data collection process"
            ]
        },
        
        "Metric Gaming": {
            "description": "Optimizing for wrong metric or gaming the metric",
            "examples": [
                "High accuracy on imbalanced data",
                "Optimizing for precision ignoring recall",
                "Using inappropriate metrics"
            ],
            "solutions": [
                "Choose metrics aligned with business goals",
                "Use multiple complementary metrics",
                "Consider cost-sensitive learning"
            ]
        }
    }
    
    print("COMMON ML PITFALLS AND SOLUTIONS")
    print("=" * 50)
    
    for pitfall, details in pitfalls.items():
        print(f"\n🚨 {pitfall}")
        print(f"Description: {details['description']}")
        
        print("\nExamples:")
        for example in details['examples']:
            print(f"  • {example}")
        
        print("\nSolutions:")
        for solution in details['solutions']:
            print(f"  ✅ {solution}")
        print("-" * 30)

common_ml_pitfalls()
```

## Conclusion dan Next Steps

---

Congratulations! Kamu telah menyelesaikan comprehensive journey melalui advanced machine learning topics. Dari feature selection yang sophisticated hingga handling real-world challenges, production considerations, dan best practices yang essential.

### Key Takeaways:

1. **Feature Selection**: Gunakan kombinasi filter, wrapper, dan embedded methods untuk optimal feature selection
2. **Pipelines**: Selalu gunakan pipelines untuk reproducible dan clean workflows
3. **Real-world Challenges**: Siap handle imbalanced data, missing values, outliers, dan drift
4. **Production**: Model deployment bukan akhir dari journey - monitoring dan maintenance sama pentingnya
5. **Interpretability**: Selalu prioritaskan explainable AI untuk business trust dan regulatory compliance

### Recommended Next Steps:

1. **Deep Learning**: Sekarang kamu siap untuk explore neural networks, CNNs, RNNs, dan Transformers
2. **Specialized Domains**: Time series forecasting, NLP, computer vision, reinforcement learning
3. **MLOps**: CI/CD for ML, model versioning, automated retraining, infrastructure
4. **Advanced Statistics**: Bayesian methods, causal inference, experimental design

### Practice Projects:

- Build end-to-end ML pipeline dengan monitoring
- Implement A/B testing framework untuk model comparison
- Create automated feature engineering system
- Develop model explainability dashboard

Machine learning adalah iterative process - keep experimenting, learning, dan improving. The journey from traditional ML ke deep learning akan smooth karena kamu sudah punya solid foundation ini. Good luck dengan advanced AI journey kamu! 🚀