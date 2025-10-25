"""
Fraud Detection Data Generator
Generates realistic financial transaction data with behavioral patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple

class FraudDataGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    def generate_transactions(self, n_transactions: int = 2000) -> pd.DataFrame:
        """Generate realistic financial transaction data"""

        # Generate base transaction data
        transactions = []

        # Create user profiles with different risk levels
        user_profiles = self._create_user_profiles()

        for i in range(n_transactions):
            # Select user profile
            user_id = random.randint(1, 500)  # 500 unique users
            user_profile = user_profiles[user_id % len(user_profiles)]

            # Generate transaction based on user profile
            transaction = self._generate_single_transaction(user_id, user_profile, i)
            transactions.append(transaction)

        df = pd.DataFrame(transactions)

        # Add behavioral features
        df = self._add_behavioral_features(df)

        return df

    def _create_user_profiles(self) -> List[Dict]:
        """Create different user risk profiles"""
        profiles = []

        # Low-risk users (80% of users)
        for _ in range(4):
            profiles.append({
                'risk_level': 'low',
                'avg_amount': random.uniform(50, 500),
                'transaction_frequency': random.uniform(0.5, 2.0),  # per day
                'merchant_preference': 'legitimate',
                'time_preference': 'business_hours',
                'geographic_stability': 'high'
            })

        # Medium-risk users (15% of users)
        for _ in range(1):
            profiles.append({
                'risk_level': 'medium',
                'avg_amount': random.uniform(200, 2000),
                'transaction_frequency': random.uniform(1.0, 5.0),
                'merchant_preference': 'mixed',
                'time_preference': 'varied',
                'geographic_stability': 'medium'
            })

        # High-risk users (5% of users)
        profiles.append({
            'risk_level': 'high',
            'avg_amount': random.uniform(1000, 10000),
            'transaction_frequency': random.uniform(3.0, 10.0),
            'merchant_preference': 'suspicious',
            'time_preference': 'unusual_hours',
            'geographic_stability': 'low'
        })

        return profiles

    def _generate_single_transaction(self, user_id: int, profile: Dict, transaction_id: int) -> Dict:
        """Generate a single transaction based on user profile"""

        # Base transaction time
        base_time = datetime.now() - timedelta(days=random.randint(1, 365))

        # Adjust amount based on profile
        if profile['risk_level'] == 'low':
            amount = np.random.lognormal(np.log(profile['avg_amount']), 0.5)
            is_fraud = random.random() < 0.02  # 2% fraud rate for low-risk
        elif profile['risk_level'] == 'medium':
            amount = np.random.lognormal(np.log(profile['avg_amount']), 0.8)
            is_fraud = random.random() < 0.15  # 15% fraud rate for medium-risk
        else:  # high-risk
            amount = np.random.lognormal(np.log(profile['avg_amount']), 1.2)
            is_fraud = random.random() < 0.40  # 40% fraud rate for high-risk

        # Adjust time based on profile
        if profile['time_preference'] == 'business_hours':
            hour = random.randint(9, 17)
        elif profile['time_preference'] == 'varied':
            hour = random.randint(0, 23)
        else:  # unusual_hours
            hour = random.choice([0, 1, 2, 3, 4, 5, 22, 23])

        transaction_time = base_time.replace(hour=hour, minute=random.randint(0, 59))

        # Generate merchant category
        if profile['merchant_preference'] == 'legitimate':
            merchant_categories = ['grocery', 'gas_station', 'restaurant', 'retail', 'pharmacy']
        elif profile['merchant_preference'] == 'mixed':
            merchant_categories = ['grocery', 'gas_station', 'restaurant', 'retail', 'pharmacy', 'online', 'atm']
        else:  # suspicious
            merchant_categories = ['online', 'atm', 'crypto', 'gambling', 'adult']

        merchant_category = random.choice(merchant_categories)

        # Generate location
        if profile['geographic_stability'] == 'high':
            lat, lon = self._get_stable_location()
        elif profile['geographic_stability'] == 'medium':
            lat, lon = self._get_variable_location()
        else:  # low stability
            lat, lon = self._get_unstable_location()

        return {
            'transaction_id': transaction_id,
            'user_id': user_id,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'transaction_time': transaction_time,
            'latitude': lat,
            'longitude': lon,
            'is_fraud': is_fraud
        }

    def _get_stable_location(self) -> Tuple[float, float]:
        """Generate location for stable users (same city)"""
        cities = [
            (40.7128, -74.0060),  # New York
            (34.0522, -118.2437),  # Los Angeles
            (41.8781, -87.6298),   # Chicago
            (29.7604, -95.3698),   # Houston
        ]
        base_lat, base_lon = random.choice(cities)
        return (
            base_lat + random.uniform(-0.1, 0.1),
            base_lon + random.uniform(-0.1, 0.1)
        )

    def _get_variable_location(self) -> Tuple[float, float]:
        """Generate location for variable users (same country)"""
        return (
            random.uniform(25.0, 49.0),  # US latitude range
            random.uniform(-125.0, -66.0)  # US longitude range
        )

    def _get_unstable_location(self) -> Tuple[float, float]:
        """Generate location for unstable users (global)"""
        return (
            random.uniform(-90.0, 90.0),
            random.uniform(-180.0, 180.0)
        )

    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral features to the dataset"""

        # Sort by user_id and transaction_time
        df = df.sort_values(['user_id', 'transaction_time'])

        # Calculate time-based features
        df['hour'] = df['transaction_time'].dt.hour
        df['day_of_week'] = df['transaction_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

        # Calculate user-level features
        user_features = []
        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id].copy()

            # Transaction frequency features
            user_df['transactions_last_24h'] = 0
            user_df['transactions_last_7d'] = 0
            user_df['avg_amount_last_30d'] = 0
            user_df['max_amount_last_30d'] = 0

            for i, row in user_df.iterrows():
                # Transactions in last 24 hours
                time_24h_ago = row['transaction_time'] - timedelta(hours=24)
                recent_transactions = user_df[
                    (user_df['transaction_time'] >= time_24h_ago) &
                    (user_df['transaction_time'] < row['transaction_time'])
                ]
                user_df.loc[i, 'transactions_last_24h'] = len(recent_transactions)

                # Transactions in last 7 days
                time_7d_ago = row['transaction_time'] - timedelta(days=7)
                recent_transactions_7d = user_df[
                    (user_df['transaction_time'] >= time_7d_ago) &
                    (user_df['transaction_time'] < row['transaction_time'])
                ]
                user_df.loc[i, 'transactions_last_7d'] = len(recent_transactions_7d)

                # Amount statistics for last 30 days
                time_30d_ago = row['transaction_time'] - timedelta(days=30)
                recent_transactions_30d = user_df[
                    (user_df['transaction_time'] >= time_30d_ago) &
                    (user_df['transaction_time'] < row['transaction_time'])
                ]
                if len(recent_transactions_30d) > 0:
                    user_df.loc[i, 'avg_amount_last_30d'] = recent_transactions_30d['amount'].mean()
                    user_df.loc[i, 'max_amount_last_30d'] = recent_transactions_30d['amount'].max()

            user_features.append(user_df)

        df = pd.concat(user_features, ignore_index=True)

        # Calculate amount-based features
        df['amount_zscore'] = df.groupby('user_id')['amount'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        df['amount_percentile'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rank(pct=True)
        )

        # Calculate merchant category features
        df['merchant_risk_score'] = df['merchant_category'].map({
            'grocery': 1, 'gas_station': 1, 'restaurant': 2, 'retail': 2, 'pharmacy': 1,
            'online': 3, 'atm': 4, 'crypto': 5, 'gambling': 5, 'adult': 4
        })

        # Calculate geographic features
        df['distance_from_home'] = np.sqrt(
            (df['latitude'] - df.groupby('user_id')['latitude'].transform('mean'))**2 +
            (df['longitude'] - df.groupby('user_id')['longitude'].transform('mean'))**2
        )

        return df

if __name__ == "__main__":
    generator = FraudDataGenerator()
    df = generator.generate_transactions(2000)

    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Features: {list(df.columns)}")

    # Save the dataset
    df.to_csv('fraud_data.csv', index=False)
    print("Dataset saved as 'fraud_data.csv'")
