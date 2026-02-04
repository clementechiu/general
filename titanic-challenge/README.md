
# General
This repository contains Jupyter notebooks that implement machine learning algorithms using data from public datasets. 

## Titanic Survival Prediction Model

### Project Overview

This project implements a machine learning pipeline to predict passenger survival on the Titanic using XGBoost classification. The analysis contains an end-to-end data science project including exploratory data analysis, feature engineering, model development, and interpretability analysis using SHAP values.

### Technical Stack

- **Python 3.x**
- **Core Libraries**: pandas, numpy, scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Model Interpretation**: SHAP (SHapley Additive exPlanations)

### Project Structure

#### 1. Data Understanding & Exploration

- Statistical summaries and missing value analysis
- Distribution analysis of numerical features (Age, Fare)
- Categorical feature frequency analysis
- Correlation analysis with survival outcome
- Visual exploration using histograms, bar plots, and box plots

#### 2. Feature Engineering

**Created Features:**
- `cabin_letter`: Extracted first letter from cabin number to capture deck information
- `FamilySize`: Combined `SibSp` (siblings/spouses) and `Parch` (parents/children)

**Missing Value Treatment:**
- Age: Filled with median value
- Fare: Filled with median value
- Preserves training set statistics for test set imputation

#### 3. Data Preprocessing

**Encoding Strategy:**
- One-hot encoding for categorical variables using scikit-learn's `ColumnTransformer`
- Handles unknown categories in test data
- Maintains feature name transparency for interpretability

**Categorical Features Encoded:**
- Passenger class (`Pclass`)
- Sex
- Number of siblings/spouses (`SibSp`)
- Number of parents/children (`Parch`)
- Embarkation port (`Embarked`)
- Cabin letter (`cabin_letter`)

#### 4. Model Development

**Algorithm:** XGBoost Classifier
- Objective: Binary logistic classification
- Train-validation split: 70-30

**Key Features:**
- Handles mixed data types efficiently
- Built-in regularization to prevent overfitting
- Feature importance extraction capabilities

#### 5. Model Evaluation & Interpretation

**Performance Metrics:**
- Accuracy score
- Confusion matrix
- Precision-Recall curves across thresholds
- ROC AUC score
- F1 score optimization

**Interpretability Analysis:**
- Feature importance ranking by gain
- SHAP summary plots for global feature impact
- Decile analysis linking predictions to feature distributions
- Threshold optimization for precision-recall trade-offs

#### 6. Evaluation Functions

The code includes modular functions for:
- Extracting top features by importance gain
- Creating prediction deciles
- Generating a table with top features and their averages for each propensity score decile for explainability 
- Plotting precision-recall curves to understand threshold trade-off
- Computing F1 scores across thresholds
- SHAP-based model explanation

### Key Results

The model provides:
1. **Predictive Performance**: Quantified through accuracy, AUC, and confusion matrix
2. **Feature Insights**: Identification of key survival predictors using gain-based importance and SHAP values
3. **Threshold Analysis**: Precision-recall trade-off curves to support decision-making
4. **Interpretability**: Visual explanations of model predictions at both global and local levels

### Usage

```python
# Load and preprocess data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Run full pipeline
# (Feature engineering → Encoding → Model training → Evaluation)

# Generate predictions
predictions = classifier.predict(X_test_encoded)
```

### Future Enhancements

- Hyperparameter tuning using cross-validation
- Ensemble methods combining multiple algorithms
- Advanced feature engineering (title extraction, ticket patterns)
- Cost-sensitive learning for imbalanced outcomes

### Author

Clemente - MPA Data Science for Public Policy, LSE
