"""
Fraud Detection System - Flask Web Application
Beautiful UI for fraud detection with real-time predictions
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.graph_objs as go
import plotly.utils
from model_trainer import FraudModelTrainer
from data_loader import CreditCardFraudDataLoader

app = Flask(__name__)

# Initialize model trainer
trainer = FraudModelTrainer()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/initialize')
def initialize_system():
    """Initialize the fraud detection system with data and models"""
    try:
        # Load credit card fraud data if not exists
        if not os.path.exists('creditcard_processed.csv'):
            loader = CreditCardFraudDataLoader()
            df = loader.load_and_process_data()
            df.to_csv('creditcard_processed.csv', index=False)

        # Train models if not exists
        if not os.path.exists('models/fraud_models.pkl'):
            os.makedirs('models', exist_ok=True)
            df = pd.read_csv('creditcard_processed.csv')
            trainer.train_all_models(df)

        # Load models
        trainer.load_models()

        # Load data for dashboard
        df = pd.read_csv('creditcard_processed.csv')

        # Calculate statistics
        stats = {
            'total_transactions': len(df),
            'fraud_rate': df['is_fraud'].mean(),
            'total_fraud': df['is_fraud'].sum(),
            'avg_amount': df['Amount'].mean(),
            'max_amount': df['Amount'].max(),
            'unique_users': df['user_id'].nunique(),
            'date_range': {
                'start': df['transaction_time'].min(),
                'end': df['transaction_time'].max()
            }
        }

        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully',
            'stats': stats
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/model-performance')
def get_model_performance():
    """Get model performance metrics"""
    try:
        trainer.load_models()

        performance_data = []
        for model_name, scores in trainer.model_scores.items():
            performance_data.append({
                'model': model_name.replace('_', ' ').title(),
                'auc': round(scores['auc'], 4),
                'accuracy': round(scores['accuracy'], 4)
            })

        return jsonify({
            'status': 'success',
            'performance': performance_data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/feature-importance')
def get_feature_importance():
    """Get feature importance for tree-based models"""
    try:
        trainer.load_models()

        feature_data = {}
        for model_name, importance_df in trainer.feature_importance.items():
            feature_data[model_name] = {
                'features': importance_df['feature'].tolist(),
                'importance': importance_df['importance'].tolist()
            }

        return jsonify({
            'status': 'success',
            'feature_importance': feature_data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    """Predict fraud for a transaction"""
    try:
        data = request.json

        # Prepare transaction data (including V1-V28 features)
        transaction_data = {}

        # Add V1-V28 features (PCA components from original dataset)
        for i in range(1, 29):
            transaction_data[f'V{i}'] = float(data.get(f'V{i}', 0))

        # Add behavioral features
        transaction_data.update({
            'Amount': float(data.get('amount', 0)),
            'merchant_category': data.get('merchant_category', 'retail'),
            'hour': int(data.get('hour', 12)),
            'day_of_week': int(data.get('day_of_week', 1)),
            'is_weekend': int(data.get('is_weekend', 0)),
            'is_night': int(data.get('is_night', 0)),
            'transactions_last_24h': int(data.get('transactions_last_24h', 0)),
            'transactions_last_7d': int(data.get('transactions_last_7d', 0)),
            'avg_amount_last_30d': float(data.get('avg_amount_last_30d', 0)),
            'max_amount_last_30d': float(data.get('max_amount_last_30d', 0)),
            'amount_zscore': float(data.get('amount_zscore', 0)),
            'amount_percentile': float(data.get('amount_percentile', 0.5)),
            'merchant_risk_score': int(data.get('merchant_risk_score', 2)),
            'distance_from_home': float(data.get('distance_from_home', 0))
        })

        # Get prediction
        prediction = trainer.predict_fraud(transaction_data)

        return jsonify({
            'status': 'success',
            'prediction': prediction
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/transaction-history')
def get_transaction_history():
    """Get transaction history for visualization"""
    try:
        df = pd.read_csv('creditcard_processed.csv')

        # Convert to datetime
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])

        # Sample recent transactions for performance
        recent_df = df.tail(100).copy()

        # Prepare data for visualization
        transactions = []
        for _, row in recent_df.iterrows():
            transactions.append({
                'id': int(row['transaction_id']),
                'user_id': int(row['user_id']),
                'amount': float(row['amount']),
                'merchant_category': row['merchant_category'],
                'timestamp': row['transaction_time'].isoformat(),
                'is_fraud': bool(row['is_fraud']),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude'])
            })

        return jsonify({
            'status': 'success',
            'transactions': transactions
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/fraud-analysis')
def get_fraud_analysis():
    """Get fraud analysis data"""
    try:
        df = pd.read_csv('creditcard_processed.csv')

        # Fraud by merchant category
        fraud_by_merchant = df.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        fraud_by_merchant.columns = ['merchant_category', 'total_transactions', 'fraud_count', 'fraud_rate']
        fraud_by_merchant = fraud_by_merchant.sort_values('fraud_rate', ascending=False)

        # Fraud by hour
        df['hour'] = pd.to_datetime(df['transaction_time']).dt.hour
        fraud_by_hour = df.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        fraud_by_hour.columns = ['hour', 'total_transactions', 'fraud_count', 'fraud_rate']

        # Amount distribution
        fraud_amounts = df[df['is_fraud'] == 1]['amount']
        legitimate_amounts = df[df['is_fraud'] == 0]['amount']

        return jsonify({
            'status': 'success',
            'fraud_by_merchant': fraud_by_merchant.to_dict('records'),
            'fraud_by_hour': fraud_by_hour.to_dict('records'),
            'amount_stats': {
                'fraud_mean': float(fraud_amounts.mean()),
                'fraud_std': float(fraud_amounts.std()),
                'legitimate_mean': float(legitimate_amounts.mean()),
                'legitimate_std': float(legitimate_amounts.std())
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
