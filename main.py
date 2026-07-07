import pandas as pd
import numpy as np
import os
from model import PredictiveModel
from visualization import ModelVisualizer
from generate_data import generate_classification_dataset, generate_regression_dataset
from sklearn.model_selection import train_test_split

def run_classification_pipeline():
    print("=" * 60)
    print("CLASSIFICATION MODEL PIPELINE")
    print("=" * 60)
    
    if not os.path.exists('classification_data.csv'):
        generate_classification_dataset()
    
    df = pd.read_csv('classification_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    model_pipeline = PredictiveModel(X_train, X_test, y_train, y_test, model_type='classification')
    results = model_pipeline.train_all_models()
    
    print("\nModel Results:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if metric != 'predictions':
                print(f"  {metric}: {value:.4f}")
    
    best_model_name, best_metrics = model_pipeline.get_best_model()
    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"{'=' * 60}")
    
    os.makedirs('outputs', exist_ok=True)
    
    visualizer = ModelVisualizer()
    
    for model_name, metrics in results.items():
        if 'accuracy' in metrics:
            y_pred = metrics['predictions']
            visualizer.plot_confusion_matrix(y_test, y_pred, model_name, f'outputs/{model_name.replace(" ", "_")}_confusion_matrix.html')
    
    visualizer.plot_model_comparison(results, 'classification', 'outputs/model_comparison_classification.html')
    
    print("\nClassification outputs saved to outputs/")

def run_regression_pipeline():
    print("\n" + "=" * 60)
    print("REGRESSION MODEL PIPELINE")
    print("=" * 60)
    
    if not os.path.exists('regression_data.csv'):
        generate_regression_dataset()
    
    df = pd.read_csv('regression_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    model_pipeline = PredictiveModel(X_train, X_test, y_train, y_test, model_type='regression')
    results = model_pipeline.train_all_models()
    
    print("\nModel Results:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if metric != 'predictions':
                print(f"  {metric}: {value:.4f}")
    
    best_model_name, best_metrics = model_pipeline.get_best_model()
    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_model_name}")
    print(f"R² Score: {best_metrics['r2']:.4f}")
    print(f"RMSE: {best_metrics['rmse']:.4f}")
    print(f"{'=' * 60}")
    
    os.makedirs('outputs', exist_ok=True)
    
    visualizer = ModelVisualizer()
    
    for model_name, metrics in results.items():
        y_pred = metrics['predictions']
        visualizer.plot_predictions_vs_actual(y_test, y_pred, model_name, f'outputs/{model_name.replace(" ", "_")}_predictions.html')
        visualizer.plot_residuals(y_test, y_pred, model_name, f'outputs/{model_name.replace(" ", "_")}_residuals.html')
    
    visualizer.plot_model_comparison(results, 'regression', 'outputs/model_comparison_regression.html')
    
    print("\nRegression outputs saved to outputs/")

if __name__ == "__main__":
    run_classification_pipeline()
    run_regression_pipeline()
    
    print("\n" + "=" * 60)
    print("All models trained and visualizations generated!")
    print("=" * 60)
