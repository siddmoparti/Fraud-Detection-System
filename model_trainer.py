"""
Fraud Detection Model Trainer
Trains and tunes multiple ML models for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import CreditCardFraudDataLoader

class FraudModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_scores = {}

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for model training"""

        # Encode categorical variables
        le_merchant = LabelEncoder()
        df['merchant_category_encoded'] = le_merchant.fit_transform(df['merchant_category'])
        self.encoders['merchant'] = le_merchant

        # Select features for training (including original V1-V28 features)
        original_features = [f'V{i}' for i in range(1, 29)]  # V1-V28 from original dataset
        behavioral_features = [
            'Amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'transactions_last_24h', 'transactions_last_7d', 'avg_amount_last_30d',
            'max_amount_last_30d', 'amount_zscore', 'amount_percentile',
            'merchant_risk_score', 'distance_from_home', 'merchant_category_encoded'
        ]
        feature_columns = original_features + behavioral_features

        X = df[feature_columns]
        y = df['is_fraud']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")

        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        # Grid search for hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        lr = LogisticRegression(random_state=42, class_weight=class_weight_dict)
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.models['logistic_regression'] = grid_search.best_estimator_

        # Evaluate model
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(y_test, y_pred_proba)
        self.model_scores['logistic_regression'] = {
            'auc': auc_score,
            'accuracy': grid_search.score(X_test, y_test),
            'best_params': grid_search.best_params_
        }

        print(f"Logistic Regression - AUC: {auc_score:.4f}")
        return y_pred, y_pred_proba

    def train_random_forest(self, X_train, y_train, X_test, y_test, feature_columns):
        """Train Random Forest model"""
        print("Training Random Forest...")

        # Grid search for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.models['random_forest'] = grid_search.best_estimator_

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': grid_search.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        self.feature_importance['random_forest'] = feature_importance

        # Evaluate model
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(y_test, y_pred_proba)
        self.model_scores['random_forest'] = {
            'auc': auc_score,
            'accuracy': grid_search.score(X_test, y_test),
            'best_params': grid_search.best_params_
        }

        print(f"Random Forest - AUC: {auc_score:.4f}")
        return y_pred, y_pred_proba

    def train_xgboost(self, X_train, y_train, X_test, y_test, feature_columns):
        """Train XGBoost model"""
        print("Training XGBoost...")

        # Grid search for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.models['xgboost'] = grid_search.best_estimator_

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': grid_search.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        self.feature_importance['xgboost'] = feature_importance

        # Evaluate model
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(y_test, y_pred_proba)
        self.model_scores['xgboost'] = {
            'auc': auc_score,
            'accuracy': grid_search.score(X_test, y_test),
            'best_params': grid_search.best_params_
        }

        print(f"XGBoost - AUC: {auc_score:.4f}")
        return y_pred, y_pred_proba

    def train_all_models(self, df: pd.DataFrame):
        """Train all models and return results"""

        # Prepare data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data(df)

        # Train models
        lr_pred, lr_proba = self.train_logistic_regression(X_train, y_train, X_test, y_test)
        rf_pred, rf_proba = self.train_random_forest(X_train, y_train, X_test, y_test, feature_columns)
        xgb_pred, xgb_proba = self.train_xgboost(X_train, y_train, X_test, y_test, feature_columns)

        # Save models
        self.save_models()

        return {
            'X_test': X_test,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'predictions': {
                'logistic_regression': (lr_pred, lr_proba),
                'random_forest': (rf_pred, rf_proba),
                'xgboost': (xgb_pred, xgb_proba)
            }
        }

    def save_models(self):
        """Save trained models and preprocessors"""
        joblib.dump(self.models, 'models/fraud_models.pkl')
        joblib.dump(self.scalers, 'models/scalers.pkl')
        joblib.dump(self.encoders, 'models/encoders.pkl')
        joblib.dump(self.feature_importance, 'models/feature_importance.pkl')
        joblib.dump(self.model_scores, 'models/model_scores.pkl')
        print("Models saved successfully!")

    def load_models(self):
        """Load trained models and preprocessors"""
        self.models = joblib.load('models/fraud_models.pkl')
        self.scalers = joblib.load('models/scalers.pkl')
        self.encoders = joblib.load('models/encoders.pkl')
        self.feature_importance = joblib.load('models/feature_importance.pkl')
        self.model_scores = joblib.load('models/model_scores.pkl')
        print("Models loaded successfully!")

    def predict_fraud(self, transaction_data: dict) -> dict:
        """Predict fraud for a single transaction"""
        if not self.models:
            self.load_models()

        # Prepare transaction data
        df = pd.DataFrame([transaction_data])

        # Encode merchant category
        df['merchant_category_encoded'] = self.encoders['merchant'].transform(
            df['merchant_category']
        )

        # Select features (same as training)
        original_features = [f'V{i}' for i in range(1, 29)]  # V1-V28 from original dataset
        behavioral_features = [
            'Amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'transactions_last_24h', 'transactions_last_7d', 'avg_amount_last_30d',
            'max_amount_last_30d', 'amount_zscore', 'amount_percentile',
            'merchant_risk_score', 'distance_from_home', 'merchant_category_encoded'
        ]
        feature_columns = original_features + behavioral_features

        X = df[feature_columns]
        X_scaled = self.scalers['main'].transform(X)

        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            proba = model.predict_proba(X_scaled)[0][1]
            predictions[model_name] = {
                'fraud_probability': proba,
                'prediction': 1 if proba > 0.5 else 0
            }

        # Ensemble prediction (average of probabilities)
        ensemble_proba = np.mean([pred['fraud_probability'] for pred in predictions.values()])

        return {
            'individual_predictions': predictions,
            'ensemble_prediction': {
                'fraud_probability': ensemble_proba,
                'prediction': 1 if ensemble_proba > 0.5 else 0
            }
        }

if __name__ == "__main__":
    # Load credit card fraud data
    loader = CreditCardFraudDataLoader()
    df = loader.load_and_process_data()

    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)

    # Train models
    trainer = FraudModelTrainer()
    results = trainer.train_all_models(df)

    # Print results
    print("\n=== Model Performance ===")
    for model_name, scores in trainer.model_scores.items():
        print(f"{model_name}:")
        print(f"  AUC: {scores['auc']:.4f}")
        print(f"  Accuracy: {scores['accuracy']:.4f}")
        print(f"  Best Parameters: {scores['best_params']}")
        print()

    # Print feature importance for tree-based models
    print("=== Feature Importance (Random Forest) ===")
    print(trainer.feature_importance['random_forest'].head(10))

    print("\n=== Feature Importance (XGBoost) ===")
    print(trainer.feature_importance['xgboost'].head(10))
