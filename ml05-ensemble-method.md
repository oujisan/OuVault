# [ai] #05 - ML Ensemble Methods 
![ml](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ml.png)

## Understanding Ensemble Methods

---

Ensemble methods mengkombinasi multiple models untuk membuat prediksi yang lebih akurat dan robust. Konsepnya seperti "wisdom of crowds" - beberapa pendapat yang dikombinasi biasanya lebih baik daripada satu pendapat individual.

Ada tiga pendekatan utama:

- **Bagging**: Train multiple models secara paralel dengan subset data yang berbeda
- **Boosting**: Train models secara sequential, setiap model memperbaiki error dari model sebelumnya
- **Stacking**: Gunakan meta-model untuk mengkombinasi prediksi dari base models

## Random Forest (Bagging)

---

Random Forest adalah ensemble dari decision trees yang menggunakan bagging dan random feature selection untuk mengurangi overfitting.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, load_boston
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier

# Generate classification data
X_class, y_class = make_classification(n_samples=1000, n_features=20, n_informative=10,
                                     n_redundant=5, n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Compare single tree vs Random Forest
single_tree = DecisionTreeClassifier(random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
single_tree.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred_tree = single_tree.predict(X_test)
y_pred_rf = rf_classifier.predict(X_test)

print("Single Decision Tree vs Random Forest:")
print(f"Single Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.3f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")

# Feature importance comparison
feature_importance_tree = single_tree.feature_importances_
feature_importance_rf = rf_classifier.feature_importances_

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(len(feature_importance_tree)), feature_importance_tree)
plt.title('Single Decision Tree - Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')

plt.subplot(1, 2, 2)
plt.bar(range(len(feature_importance_rf)), feature_importance_rf)
plt.title('Random Forest - Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')

plt.tight_layout()
plt.show()
```

### Hyperparameter Tuning for Random Forest

```python
from sklearn.model_selection import GridSearchCV

# Effect of number of estimators
n_estimators_range = [10, 50, 100, 200, 500]
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)
    
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'o-', label='Training Score', linewidth=2)
plt.plot(n_estimators_range, test_scores, 'o-', label='Test Score', linewidth=2)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest Performance vs Number of Estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Grid search for optimal parameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, 
                      cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best cross-validation score: {rf_grid.best_score_:.3f}")
print(f"Test score with best params: {rf_grid.score(X_test, y_test):.3f}")
```

### Out-of-Bag (OOB) Scores

```python
# OOB score provides estimate of model performance without separate validation set
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_oob.oob_score_:.3f}")
print(f"Test Score: {rf_oob.score(X_test, y_test):.3f}")

# OOB vs validation score for different n_estimators
oob_scores = []
test_scores_oob = []
n_estimators_oob = range(10, 201, 10)

for n_est in n_estimators_oob:
    rf = RandomForestClassifier(n_estimators=n_est, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    
    oob_scores.append(rf.oob_score_)
    test_scores_oob.append(rf.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_oob, oob_scores, label='OOB Score', linewidth=2)
plt.plot(n_estimators_oob, test_scores_oob, label='Test Score', linewidth=2)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('OOB Score vs Test Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Gradient Boosting

---

Gradient Boosting builds models sequentially, dengan setiap model baru mencoba memperbaiki errors dari ensemble sebelumnya.

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_regression

# Classification with Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                         max_depth=3, random_state=42)
gb_classifier.fit(X_train, y_train)
y_pred_gb = gb_classifier.predict(X_test, y_test)

print(f"Gradient Boosting Classification Accuracy: {accuracy_score(y_test, y_pred_gb):.3f}")

# Visualize learning curve
test_scores = []
train_scores = []

for i, pred in enumerate(gb_classifier.staged_predict(X_test)):
    test_scores.append(accuracy_score(y_test, pred))

for i, pred in enumerate(gb_classifier.staged_predict(X_train)):
    train_scores.append(accuracy_score(y_train, pred))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(test_scores) + 1), test_scores, label='Test Score', linewidth=2)
plt.plot(range(1, len(train_scores) + 1), train_scores, label='Train Score', linewidth=2)
plt.xlabel('Boosting Iterations')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Regression example
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                       max_depth=3, random_state=42)
gb_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = gb_regressor.predict(X_test_reg)

print(f"Gradient Boosting Regression R²: {r2_score(y_test_reg, y_pred_reg):.3f}")
print(f"Gradient Boosting Regression RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.3f}")
```

### Learning Rate Impact

```python
# Compare different learning rates
learning_rates = [0.01, 0.1, 0.2, 0.5]
plt.figure(figsize=(15, 10))

for i, lr in enumerate(learning_rates):
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=lr, 
                                  max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    
    # Calculate staged scores
    test_scores = []
    for pred in gb.staged_predict(X_test):
        test_scores.append(accuracy_score(y_test, pred))
    
    plt.subplot(2, 2, i+1)
    plt.plot(range(1, len(test_scores) + 1), test_scores, linewidth=2)
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Test Accuracy')
    plt.title(f'Learning Rate = {lr}')
    plt.grid(True, alpha=0.3)
    
    final_score = test_scores[-1]
    plt.text(0.7, 0.1, f'Final Score: {final_score:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

plt.tight_layout()
plt.show()
```

## XGBoost - Extreme Gradient Boosting

---

XGBoost adalah implementasi gradient boosting yang sangat optimized dan sering menang dalam kompetisi ML.

```python
# Install xgboost if not already installed
# pip install xgboost

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    
    # XGBoost Classification
    xgb_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, 
                                 max_depth=3, random_state=42)
    xgb_classifier.fit(X_train, y_train)
    y_pred_xgb = xgb_classifier.predict(X_test)
    
    print(f"XGBoost Classification Accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}")
    
    # Feature importance
    plt.figure(figsize=(10, 6))
    feature_importance = xgb_classifier.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1][:10]  # Top 10 features
    
    plt.bar(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Importance')
    plt.title('XGBoost Feature Importance (Top 10)')
    plt.xticks(range(len(sorted_idx)), sorted_idx)
    plt.show()
    
    # XGBoost Regression
    xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, 
                               max_depth=3, random_state=42)
    xgb_regressor.fit(X_train_reg, y_train_reg)
    y_pred_xgb_reg = xgb_regressor.predict(X_test_reg)
    
    print(f"XGBoost Regression R²: {r2_score(y_test_reg, y_pred_xgb_reg):.3f}")
    
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    print("Using sklearn GradientBoosting as alternative...")
```

### XGBoost Advanced Features

```python
try:
    # Early stopping to prevent overfitting
    xgb_early = XGBClassifier(n_estimators=1000, learning_rate=0.1, 
                            max_depth=3, random_state=42)
    
    xgb_early.fit(X_train, y_train, 
                 eval_set=[(X_test, y_test)], 
                 early_stopping_rounds=10, 
                 verbose=False)
    
    print(f"XGBoost with early stopping - Best iteration: {xgb_early.best_iteration}")
    
    # Cross-validation with XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'eval_metric': 'logloss'
    }
    
    cv_results = xgb.cv(params, dtrain, num_boost_round=100, nfold=5, 
                       early_stopping_rounds=10, seed=42, verbose_eval=False)
    
    print(f"XGBoost CV - Best score: {cv_results['test-logloss-mean'].min():.4f}")
    
except:
    print("Advanced XGBoost features require xgboost package")
```

## AdaBoost

---

AdaBoost (Adaptive Boosting) adalah salah satu boosting algorithms pertama yang focuses pada misclassified examples.

```python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

# AdaBoost Classification
ada_classifier = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
ada_classifier.fit(X_train, y_train)
y_pred_ada = ada_classifier.predict(X_test)

print(f"AdaBoost Classification Accuracy: {accuracy_score(y_test, y_pred_ada):.3f}")

# Compare different base estimators
from sklearn.tree import DecisionTreeClassifier

base_estimators = [
    ("Decision Stump", DecisionTreeClassifier(max_depth=1)),
    ("Decision Tree (depth=3)", DecisionTreeClassifier(max_depth=3)),
    ("Decision Tree (depth=5)", DecisionTreeClassifier(max_depth=5))
]

plt.figure(figsize=(15, 5))

for i, (name, base_est) in enumerate(base_estimators):
    ada = AdaBoostClassifier(base_estimator=base_est, n_estimators=100, 
                           learning_rate=1.0, random_state=42)
    ada.fit(X_train, y_train)
    
    # Calculate staged scores
    test_scores = []
    for pred in ada.staged_predict(X_test):
        test_scores.append(accuracy_score(y_test, pred))
    
    plt.subplot(1, 3, i+1)
    plt.plot(range(1, len(test_scores) + 1), test_scores, linewidth=2)
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Test Accuracy')
    plt.title(f'AdaBoost with {name}')
    plt.grid(True, alpha=0.3)
    
    final_score = test_scores[-1]
    plt.text(0.05, 0.95, f'Final: {final_score:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor='lightblue'))

plt.tight_layout()
plt.show()
```

## Voting Classifiers

---

Voting classifiers mengkombinasi different algorithms dan menggunakan majority vote (hard voting) atau average probabilities (soft voting).

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Create individual classifiers
log_reg = LogisticRegression(random_state=42)
svm_clf = SVC(probability=True, random_state=42)  # probability=True for soft voting
nb_clf = GaussianNB()
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Hard Voting Classifier
hard_voting_clf = VotingClassifier(
    estimators=[('lr', log_reg), ('svm', svm_clf), ('nb', nb_clf), ('rf', rf_clf)],
    voting='hard'
)

# Soft Voting Classifier
soft_voting_clf = VotingClassifier(
    estimators=[('lr', log_reg), ('svm', svm_clf), ('nb', nb_clf), ('rf', rf_clf)],
    voting='soft'
)

# Train all classifiers
classifiers = [
    ("Logistic Regression", log_reg),
    ("SVM", svm_clf),
    ("Naive Bayes", nb_clf),
    ("Random Forest", rf_clf),
    ("Hard Voting", hard_voting_clf),
    ("Soft Voting", soft_voting_clf)
]

results = []
for name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((name, accuracy))
    print(f"{name}: {accuracy:.3f}")

# Visualize results
names, accuracies = zip(*results)
plt.figure(figsize=(12, 6))
colors = ['skyblue'] * 4 + ['orange', 'red']  # Highlight ensemble methods
bars = plt.bar(names, accuracies, color=colors)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Individual vs Ensemble Classifier Performance')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## Stacking

---

Stacking menggunakan meta-learner untuk belajar cara optimally combine predictions dari base models.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict

# Base models
base_models = [
    ('lr', LogisticRegression(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Meta model
meta_model = LogisticRegression(random_state=42)

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # Use cross-validation to train meta-model
)

stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)

print(f"Stacking Classifier Accuracy: {accuracy_score(y_test, y_pred_stack):.3f}")

# Manual stacking implementation for understanding
def manual_stacking(base_models, meta_model, X_train, y_train, X_test, cv=5):
    """Manual implementation of stacking"""
    
    # Generate meta-features using cross-validation
    meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_features_test = np.zeros((X_test.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(base_models):
        # Cross-validation predictions for training set
        cv_preds = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')
        meta_features_train[:, i] = cv_preds[:, 1]  # Use probability of positive class
        
        # Train on full training set and predict test set
        model.fit(X_train, y_train)
        test_preds = model.predict_proba(X_test)
        meta_features_test[:, i] = test_preds[:, 1]
    
    # Train meta-model
    meta_model.fit(meta_features_train, y_train)
    
    # Make final predictions
    final_preds = meta_model.predict(meta_features_test)
    
    return final_preds, meta_features_train, meta_features_test

# Apply manual stacking
manual_preds, meta_train, meta_test = manual_stacking(
    base_models, LogisticRegression(random_state=42), X_train, y_train, X_test)

print(f"Manual Stacking Accuracy: {accuracy_score(y_test, manual_preds):.3f}")

# Visualize meta-features
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(meta_train[:, 0], meta_train[:, 1], c=y_train, alpha=0.6, cmap='RdYlBu')
plt.xlabel('Logistic Regression Predictions')
plt.ylabel('Random Forest Predictions')
plt.title('Meta-features (Training Set)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(meta_test[:, 0], meta_test[:, 1], c=y_test, alpha=0.6, cmap='RdYlBu')
plt.xlabel('Logistic Regression Predictions')
plt.ylabel('Random Forest Predictions')
plt.title('Meta-features (Test Set)')
plt.colorbar()

plt.tight_layout()
plt.show()
```

## Model Comparison and Analysis

---

```python
from sklearn.model_selection import cross_val_score
import time

# Comprehensive comparison of all ensemble methods
ensemble_models = {
    'Single Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Hard Voting': VotingClassifier(
        estimators=[('lr', LogisticRegression(random_state=42)), 
                   ('rf', RandomForestClassifier(n_estimators=50, random_state=42))],
        voting='hard'
    ),
    'Soft Voting': VotingClassifier(
        estimators=[('lr', LogisticRegression(random_state=42)), 
                   ('rf', RandomForestClassifier(n_estimators=50, random_state=42))],
        voting='soft'
    ),
    'Stacking': StackingClassifier(
        estimators=[('lr', LogisticRegression(random_state=42)),
                   ('rf', RandomForestClassifier(n_estimators=50, random_state=42))],
        final_estimator=LogisticRegression(random_state=42),
        cv=3
    )
}

# Add XGBoost if available
try:
    ensemble_models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42)
except:
    pass

results = []

for name, model in ensemble_models.items():
    # Cross-validation
    start_time = time.time()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    training_time = time.time() - start_time
    
    # Test performance
    start_time = time.time()
    model.fit(X_train, y_train)
    prediction_time = time.time() - start_time
    
    test_score = model.score(X_test, y_test)
    
    results.append({
        'Model': name,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test Score': test_score,
        'Training Time': training_time,
        'Prediction Time': prediction_time
    })

# Display results
results_df = pd.DataFrame(results).sort_values('Test Score', ascending=False)
print("Ensemble Methods Comparison:")
print(results_df.round(4))

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Test Score
axes[0, 0].barh(results_df['Model'], results_df['Test Score'])
axes[0, 0].set_xlabel('Test Score')
axes[0, 0].set_title('Model Performance')

# Cross-validation scores with error bars
axes[0, 1].errorbar(results_df['CV Mean'], range(len(results_df)), 
                   xerr=results_df['CV Std'], fmt='o', capsize=5)
axes[0, 1].set_yticks(range(len(results_df)))
axes[0, 1].set_yticklabels(results_df['Model'])
axes[0, 1].set_xlabel('CV Score')
axes[0, 1].set_title('Cross-Validation Performance')

# Training time
axes[1, 0].barh(results_df['Model'], results_df['Training Time'])
axes[1, 0].set_xlabel('Training Time (seconds)')
axes[1, 0].set_title('Training Time Comparison')

# Performance vs Time trade-off
axes[1, 1].scatter(results_df['Training Time'], results_df['Test Score'], s=100, alpha=0.7)
for i, model in enumerate(results_df['Model']):
    axes[1, 1].annotate(model, (results_df.iloc[i]['Training Time'], results_df.iloc[i]['Test Score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[1, 1].set_xlabel('Training Time (seconds)')
axes[1, 1].set_ylabel('Test Score')
axes[1, 1].set_title('Performance vs Training Time')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## When to Use Which Ensemble Method

---

```python
# Create a decision guide
decision_guide = {
    'Random Forest': {
        'Best for': 'General purpose, good default choice',
        'Pros': ['Fast training', 'Feature importance', 'Less overfitting', 'Handles missing values'],
        'Cons': ['Can overfit with very noisy data', 'Less interpretable than single tree'],
        'Use when': 'Need robust baseline, mixed data types, feature selection'
    },
    
    'Gradient Boosting': {
        'Best for': 'High accuracy when properly tuned',
        'Pros': ['Often highest accuracy', 'Good feature importance', 'Handles different data types'],
        'Cons': ['Prone to overfitting', 'Sensitive to outliers', 'Slower training'],
        'Use when': 'Need maximum accuracy, have clean data, computational resources available'
    },
    
    'XGBoost': {
        'Best for': 'Competitions, structured data',
        'Pros': ['State-of-art performance', 'Built-in regularization', 'Handles missing values'],
        'Cons': ['Many hyperparameters', 'Can overfit', 'Complex to tune'],
        'Use when': 'Competing, structured data, need maximum performance'
    },
    
    'AdaBoost': {
        'Best for': 'Binary classification with weak learners',
        'Pros': ['Simple to understand', 'Less prone to overfitting than GB', 'Good with weak learners'],
        'Cons': ['Sensitive to noise and outliers', 'Performance depends on base learner'],
        'Use when': 'Simple binary classification, clean data, interpretability needed'
    },
    
    'Voting': {
        'Best for': 'Combining different algorithm types',
        'Pros': ['Simple to implement', 'Reduces overfitting', 'Improves stability'],
        'Cons': ['Performance limited by weakest model', 'Can be slow'],
        'Use when': 'Have diverse good models, want stability over peak performance'
    },
    
    'Stacking': {
        'Best for': 'Maximum performance with diverse models',
        'Pros': ['Often best performance', 'Learns optimal combination', 'Flexible'],
        'Cons': ['Complex to implement', 'Risk of overfitting', 'Slow training'],
        'Use when': 'Need absolute best performance, have computational resources'
    }
}

print("=== ENSEMBLE METHODS DECISION GUIDE ===\n")
for method, info in decision_guide.items():
    print(f"{method.upper()}:")
    print(f"  Best for: {info['Best for']}")
    print(f"  Pros: {', '.join(info['Pros'])}")
    print(f"  Cons: {', '.join(info['Cons'])}")
    print(f"  Use when: {info['Use when']}\n")
```

## Key Takeaways

---

1. **Ensemble methods** almost always outperform single models - "wisdom of crowds"
2. **Random Forest** is excellent default choice - robust, fast, interpretable
3. **Gradient Boosting** (GB/XGBoost) often gives highest accuracy but needs careful tuning
4. **Voting** is simple way to combine different algorithm types
5. **Stacking** can give best performance but is complex and prone to overfitting
6. **Trade-offs** exist between performance, interpretability, and computational cost
7. **Cross-validation** is crucial for ensemble methods to avoid overfitting

Ensemble methods are powerful tools yang akan frequently used dalam real-world ML projects. Understanding when and how to use each method adalah key skill untuk advanced ML practitioners. Next, kita akan explore unsupervised learning!