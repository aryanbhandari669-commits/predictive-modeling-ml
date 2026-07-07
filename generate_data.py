import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split

def generate_classification_dataset(filename='classification_data.csv'):
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv(filename, index=False)
    print(f"Classification dataset generated: {filename}")
    return df

def generate_regression_dataset(filename='regression_data.csv'):
    X, y = make_regression(n_samples=500, n_features=10, noise=20, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    df.to_csv(filename, index=False)
    print(f"Regression dataset generated: {filename}")
    return df

if __name__ == "__main__":
    generate_classification_dataset()
    generate_regression_dataset()
