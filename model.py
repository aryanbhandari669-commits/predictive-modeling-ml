import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

class PredictiveModel:
    def __init__(self, X_train, X_test, y_train, y_test, model_type='regression'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type
        self.models = {}
        self.results = {}
    
    def normalize_data(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        return self.X_train_scaled, self.X_test_scaled
    
    def train_linear_regression(self):
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = {
            'predictions': y_pred,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        return model, {'mse': mse, 'rmse': rmse, 'r2': r2}
    
    def train_decision_tree(self):
        if self.model_type == 'regression':
            model = DecisionTreeRegressor(random_state=42, max_depth=10)
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            self.results['Decision Tree'] = {
                'predictions': y_pred,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
        else:
            model = DecisionTreeClassifier(random_state=42, max_depth=10)
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            self.results['Decision Tree'] = {
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        self.models['Decision Tree'] = model
        return model, self.results['Decision Tree']
    
    def train_random_forest(self):
        if self.model_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            self.results['Random Forest'] = {
                'predictions': y_pred,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            self.results['Random Forest'] = {
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        self.models['Random Forest'] = model
        return model, self.results['Random Forest']
    
    def train_logistic_regression(self):
        if self.model_type == 'classification':
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            self.models['Logistic Regression'] = model
            self.results['Logistic Regression'] = {
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            return model, self.results['Logistic Regression']
        return None, None
    
    def train_all_models(self):
        self.normalize_data()
        if self.model_type == 'regression':
            self.train_linear_regression()
        self.train_decision_tree()
        self.train_random_forest()
        if self.model_type == 'classification':
            self.train_logistic_regression()
        return self.results
    
    def get_best_model(self):
        if self.model_type == 'regression':
            best_model = max(self.results.items(), key=lambda x: x[1]['r2'])
        else:
            best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        return best_model
