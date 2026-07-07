# Predictive Modeling Using Machine Learning

A comprehensive Python project for building predictive models using machine learning algorithms.

## Features

- **Multiple Algorithms**: Linear Regression, Decision Trees, Random Forest, Logistic Regression
- **Model Training**: Train and evaluate models on both regression and classification tasks
- **Performance Metrics**: Comprehensive metrics including R², RMSE, Accuracy, Precision, Recall, F1-Score
- **Visualizations**: Confusion matrices, ROC curves, predictions vs actual, residual plots
- **Feature Importance**: Analyze feature importance for tree-based models
- **Model Comparison**: Compare performance across multiple algorithms

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run All Pipelines

```bash
python main.py
```

### Generate Datasets

```bash
python generate_data.py
```

## Project Structure

- `model.py` - Machine learning model training and evaluation
- `visualization.py` - Performance visualization utilities
- `main.py` - Main execution script for both regression and classification
- `generate_data.py` - Dataset generation
- `requirements.txt` - Project dependencies

## Output

- `outputs/` - Generated visualizations:
  - Confusion matrices for classification models
  - ROC curves
  - Actual vs Predicted plots
  - Residual plots for regression
  - Feature importance charts
  - Model comparison charts

## Algorithms

### Regression Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Classification Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Metrics

### Regression
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve
