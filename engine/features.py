### engine/features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from engine.config import CATEGORY_DEFAULTS

def create_purchase_intelligence(df):
    category_stats = df.groupby('Category').agg({
        'Unit Ticket Sales': ['mean', 'std'],
        'Unit Cost': ['mean', 'std'],
        'Profit_Margin': ['mean', 'std'],
        'Customer_Payment_Ratio': 'mean'
    }).round(2)

    category_stats.columns = [
        'Market_Category_Resale_Mean', 'Market_Category_Resale_Std',
        'Market_Category_Cost_Mean', 'Market_Category_Cost_Std', 
        'Market_Category_Margin_Mean', 'Market_Category_Margin_Std',
        'Market_Category_Payment_Ratio'
    ]

    df = df.merge(category_stats, left_on='Category', right_index=True, how='left')

    venue_stats = df.groupby('Venue').agg({
        'Unit Ticket Sales': 'mean',
        'Profit_Margin': 'mean'
    }).round(2)

    venue_stats.columns = ['Market_Venue_Resale_Mean', 'Market_Venue_Margin_Mean']
    df = df.merge(venue_stats, left_on='Venue', right_index=True, how='left')

    market_cols = [col for col in df.columns if col.startswith('Market_')]
    for col in market_cols:
        df[col] = df[col].fillna(df[col].median())

    return df

def prepare_features(self, df):
    print("Preparing features for customer purchase pricing...")
    categorical_features = ['Event Type', 'Category', 'State', 'Venue', 'Performer']
    numerical_features = [
        'QTY', 'Days_Until_Event', 'Event_Month',
        'Is_Weekend_Event', 'Is_Peak_Season', 'Is_Last_Minute', 'Is_Very_Early'
    ]

    df = create_purchase_intelligence(df)

    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')
    for col in numerical_features:
        df[col] = df[col].fillna(df[col].median())

    df_encoded = df.copy()
    self.label_encoders = getattr(self, 'label_encoders', {})

    for col in categorical_features:
        if col not in self.label_encoders:
            self.label_encoders[col] = LabelEncoder()
        df_encoded[col + '_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))

    encoded_cols = [col + '_encoded' for col in categorical_features]
    market_cols = [col for col in df.columns if col.startswith('Market_')]
    self.feature_columns = numerical_features + encoded_cols + market_cols

    print(f"Feature set: {len(self.feature_columns)} features")
    return df_encoded

def add_market_intelligence_for_prediction(input_data, category, venue):
    defaults = CATEGORY_DEFAULTS.get(category, {'resale_mean': 120, 'margin_mean': 0.25, 'payment_ratio': 0.80})

    input_data['Market_Category_Resale_Mean'] = defaults['resale_mean']
    input_data['Market_Category_Resale_Std'] = 50
    input_data['Market_Category_Cost_Mean'] = defaults['resale_mean'] * defaults['payment_ratio']
    input_data['Market_Category_Cost_Std'] = 30
    input_data['Market_Category_Margin_Mean'] = defaults['margin_mean']
    input_data['Market_Category_Margin_Std'] = 0.1
    input_data['Market_Category_Payment_Ratio'] = defaults['payment_ratio']
    input_data['Market_Venue_Resale_Mean'] = defaults['resale_mean']
    input_data['Market_Venue_Margin_Mean'] = defaults['margin_mean']

    return input_data
