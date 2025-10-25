"""
Credit Card Fraud Detection Data Loader
Downloads and processes the Kaggle credit card fraud dataset
"""

import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import random

class CreditCardFraudDataLoader:
    def __init__(self):
        self.dataset_path = None
        self.df = None

    def download_dataset(self):
        """Download the credit card fraud dataset from Kaggle"""
        print("Downloading credit card fraud dataset from Kaggle...")

        # Download latest version
        self.dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print(f"Dataset downloaded to: {self.dataset_path}")

        return self.dataset_path

    def load_and_process_data(self):
        """Load and process the credit card fraud dataset"""

        # Download dataset if not already downloaded
        if not self.dataset_path:
            self.download_dataset()

        # Load the dataset
        csv_path = os.path.join(self.dataset_path, "creditcard.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found at {csv_path}")

        print("Loading credit card fraud dataset...")
        self.df = pd.read_csv(csv_path)

        print(f"Dataset loaded: {len(self.df)} transactions")
        print(f"Fraud rate: {self.df['Class'].mean():.2%}")

        # Add behavioral features to make it more realistic
        self.df = self._add_behavioral_features()

        return self.df

    def _add_behavioral_features(self):
        """Add behavioral features to the dataset"""

        # Add time-based features (simulate transaction times)
        np.random.seed(42)
        start_date = datetime.now() - timedelta(days=365)

        # Generate realistic transaction times
        transaction_times = []
        for i in range(len(self.df)):
            # Add some randomness to time
            days_offset = random.randint(0, 365)
            hours_offset = random.randint(0, 23)
            minutes_offset = random.randint(0, 59)

            transaction_time = start_date + timedelta(
                days=days_offset,
                hours=hours_offset,
                minutes=minutes_offset
            )
            transaction_times.append(transaction_time)

        self.df['transaction_time'] = transaction_times

        # Add time-based features
        self.df['hour'] = pd.to_datetime(self.df['transaction_time']).dt.hour
        self.df['day_of_week'] = pd.to_datetime(self.df['transaction_time']).dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['is_night'] = self.df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

        # Add user IDs (simulate different users)
        n_users = 1000  # Simulate 1000 users
        self.df['user_id'] = np.random.randint(1, n_users + 1, len(self.df))

        # Add merchant categories based on amount patterns
        self.df['merchant_category'] = self._assign_merchant_categories()

        # Add location features (simulate geographic data)
        self.df['latitude'] = np.random.uniform(25.0, 49.0, len(self.df))
        self.df['longitude'] = np.random.uniform(-125.0, -66.0, len(self.df))

        # Calculate user-level behavioral features
        self.df = self._calculate_user_features()

        # Rename Class column to is_fraud for consistency
        self.df['is_fraud'] = self.df['Class']
        self.df = self.df.drop('Class', axis=1)

        return self.df

    def _assign_merchant_categories(self):
        """Assign merchant categories based on transaction patterns"""
        categories = []

        for _, row in self.df.iterrows():
            amount = row['Amount']
            is_fraud = row['Class']

            # Assign categories based on amount and fraud patterns
            if amount < 10:
                if is_fraud:
                    categories.append('atm')
                else:
                    categories.append('gas_station')
            elif amount < 50:
                if is_fraud:
                    categories.append('online')
                else:
                    categories.append('grocery')
            elif amount < 200:
                if is_fraud:
                    categories.append('crypto')
                else:
                    categories.append('retail')
            elif amount < 1000:
                if is_fraud:
                    categories.append('gambling')
                else:
                    categories.append('restaurant')
            else:
                if is_fraud:
                    categories.append('adult')
                else:
                    categories.append('pharmacy')

        return categories

    def _calculate_user_features(self):
        """Calculate user-level behavioral features"""

        # Sort by user_id and transaction_time
        self.df = self.df.sort_values(['user_id', 'transaction_time'])

        # Initialize new columns
        self.df['transactions_last_24h'] = 0
        self.df['transactions_last_7d'] = 0
        self.df['avg_amount_last_30d'] = 0
        self.df['max_amount_last_30d'] = 0
        self.df['amount_zscore'] = 0
        self.df['amount_percentile'] = 0

        # Calculate features for each user
        for user_id in self.df['user_id'].unique():
            user_mask = self.df['user_id'] == user_id
            user_df = self.df[user_mask].copy()

            if len(user_df) == 0:
                continue

            # Calculate rolling features
            for i, (idx, row) in enumerate(user_df.iterrows()):
                current_time = row['transaction_time']

                # Transactions in last 24 hours
                time_24h_ago = current_time - timedelta(hours=24)
                recent_24h = user_df[
                    (user_df['transaction_time'] >= time_24h_ago) &
                    (user_df['transaction_time'] < current_time)
                ]
                self.df.loc[idx, 'transactions_last_24h'] = len(recent_24h)

                # Transactions in last 7 days
                time_7d_ago = current_time - timedelta(days=7)
                recent_7d = user_df[
                    (user_df['transaction_time'] >= time_7d_ago) &
                    (user_df['transaction_time'] < current_time)
                ]
                self.df.loc[idx, 'transactions_last_7d'] = len(recent_7d)

                # Amount statistics for last 30 days
                time_30d_ago = current_time - timedelta(days=30)
                recent_30d = user_df[
                    (user_df['transaction_time'] >= time_30d_ago) &
                    (user_df['transaction_time'] < current_time)
                ]

                if len(recent_30d) > 0:
                    self.df.loc[idx, 'avg_amount_last_30d'] = recent_30d['Amount'].mean()
                    self.df.loc[idx, 'max_amount_last_30d'] = recent_30d['Amount'].max()
                else:
                    self.df.loc[idx, 'avg_amount_last_30d'] = 0
                    self.df.loc[idx, 'max_amount_last_30d'] = 0

        # Calculate z-scores and percentiles by user
        for user_id in self.df['user_id'].unique():
            user_mask = self.df['user_id'] == user_id
            user_amounts = self.df.loc[user_mask, 'Amount']

            if len(user_amounts) > 1:
                mean_amount = user_amounts.mean()
                std_amount = user_amounts.std()

                if std_amount > 0:
                    self.df.loc[user_mask, 'amount_zscore'] = (user_amounts - mean_amount) / std_amount

                self.df.loc[user_mask, 'amount_percentile'] = user_amounts.rank(pct=True)

        # Add merchant risk scores
        merchant_risk_map = {
            'grocery': 1, 'gas_station': 1, 'restaurant': 2, 'retail': 2, 'pharmacy': 1,
            'online': 3, 'atm': 4, 'crypto': 5, 'gambling': 5, 'adult': 4
        }
        self.df['merchant_risk_score'] = self.df['merchant_category'].map(merchant_risk_map)

        # Add distance from home (simulate geographic stability)
        self.df['distance_from_home'] = np.sqrt(
            (self.df['latitude'] - self.df.groupby('user_id')['latitude'].transform('mean'))**2 +
            (self.df['longitude'] - self.df.groupby('user_id')['longitude'].transform('mean'))**2
        )

        return self.df

    def get_dataset_info(self):
        """Get information about the dataset"""
        if self.df is None:
            return None

        return {
            'total_transactions': len(self.df),
            'fraud_rate': self.df['is_fraud'].mean(),
            'total_fraud': self.df['is_fraud'].sum(),
            'avg_amount': self.df['Amount'].mean(),
            'max_amount': self.df['Amount'].max(),
            'unique_users': self.df['user_id'].nunique(),
            'date_range': {
                'start': self.df['transaction_time'].min(),
                'end': self.df['transaction_time'].max()
            },
            'features': list(self.df.columns)
        }

if __name__ == "__main__":
    # Load and process the dataset
    loader = CreditCardFraudDataLoader()
    df = loader.load_and_process_data()

    # Print dataset information
    info = loader.get_dataset_info()
    print("\n=== Dataset Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")

    # Save processed dataset
    df.to_csv('creditcard_processed.csv', index=False)
    print(f"\nProcessed dataset saved as 'creditcard_processed.csv'")
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
