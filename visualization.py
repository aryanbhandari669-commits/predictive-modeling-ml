import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

class ModelVisualizer:
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, output_path='confusion_matrix.html'):
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            text=cm,
            textposition='auto',
            colorscale='Blues'
        ))
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500,
            width=600
        )
        fig.write_html(output_path)
        return output_path
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, output_path='roc_curve.html'):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            width=700
        )
        fig.write_html(output_path)
        return output_path
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, output_path='predictions.html'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(y_true))),
            y=y_true,
            mode='markers',
            name='Actual',
            marker=dict(size=8, color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode='markers',
            name='Predicted',
            marker=dict(size=8, color='red')
        ))
        fig.update_layout(
            title=f'Actual vs Predicted - {model_name}',
            xaxis_title='Sample Index',
            yaxis_title='Value',
            height=500,
            width=1000
        )
        fig.write_html(output_path)
        return output_path
    
    def plot_residuals(self, y_true, y_pred, model_name, output_path='residuals.html'):
        residuals = y_true - y_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(size=8, color='green')
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title=f'Residual Plot - {model_name}',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            height=500,
            width=800
        )
        fig.write_html(output_path)
        return output_path
    
    def plot_feature_importance(self, model, feature_names, model_name, output_path='feature_importance.html'):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation='h',
                marker=dict(color='purple')
            ))
            fig.update_layout(
                title=f'Top 10 Feature Importance - {model_name}',
                xaxis_title='Importance',
                yaxis_title='Features',
                height=500,
                width=800
            )
            fig.write_html(output_path)
            return output_path
        return None
    
    def plot_model_comparison(self, results, model_type, output_path='model_comparison.html'):
        models = list(results.keys())
        
        if model_type == 'regression':
            r2_scores = [results[m]['r2'] for m in models]
            rmse_scores = [results[m]['rmse'] for m in models]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('R² Score', 'RMSE')
            )
            fig.add_trace(
                go.Bar(x=models, y=r2_scores, name='R²', marker_color='blue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='orange'),
                row=1, col=2
            )
        else:
            accuracy = [results[m]['accuracy'] for m in models]
            precision = [results[m]['precision'] for m in models]
            recall = [results[m]['recall'] for m in models]
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Accuracy', 'Precision', 'Recall')
            )
            fig.add_trace(
                go.Bar(x=models, y=accuracy, name='Accuracy', marker_color='blue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=models, y=precision, name='Precision', marker_color='green'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=models, y=recall, name='Recall', marker_color='red'),
                row=1, col=3
            )
        
        fig.update_layout(height=500, width=1200, showlegend=False, title_text="Model Performance Comparison")
        fig.write_html(output_path)
        return output_path
