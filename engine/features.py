import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from engine.config import CATEGORY_DEFAULTS

def create_purchase_intelligence(df):
    """Enhanced market intelligence feature creation"""
    print("Creating purchase intelligence features...")
    
    # Category-level statistics
    category_stats = df.groupby('Category').agg({
        'Unit Ticket Sales': ['mean', 'std', 'median', 'count'],
        'Unit Cost': ['mean', 'std', 'median'],
        'Profit_Margin': ['mean', 'std', 'median'],
        'Customer_Payment_Ratio': ['mean', 'std'],
        'Days_Until_Event': 'mean'
    }).round(3)

    # Flatten column names
    category_stats.columns = [
        'Market_Category_Resale_Mean', 'Market_Category_Resale_Std', 'Market_Category_Resale_Median', 'Market_Category_Volume',
        'Market_Category_Cost_Mean', 'Market_Category_Cost_Std', 'Market_Category_Cost_Median',
        'Market_Category_Margin_Mean', 'Market_Category_Margin_Std', 'Market_Category_Margin_Median',
        'Market_Category_Payment_Ratio_Mean', 'Market_Category_Payment_Ratio_Std',
        'Market_Category_Avg_Lead_Time'
    ]

    # Merge category stats
    df = df.merge(category_stats, left_on='Category', right_index=True, how='left')

    # Venue-level statistics
    venue_stats = df.groupby('Venue').agg({
        'Unit Ticket Sales': ['mean', 'std', 'count'],
        'Unit Cost': 'mean',
        'Profit_Margin': ['mean', 'std'],
        'Days_Until_Event': 'mean'
    }).round(3)

    venue_stats.columns = [
        'Market_Venue_Resale_Mean', 'Market_Venue_Resale_Std', 'Market_Venue_Volume',
        'Market_Venue_Cost_Mean', 
        'Market_Venue_Margin_Mean', 'Market_Venue_Margin_Std',
        'Market_Venue_Avg_Lead_Time'
    ]

    # Merge venue stats
    df = df.merge(venue_stats, left_on='Venue', right_index=True, how='left')

    # State-level statistics
    state_stats = df.groupby('State').agg({
        'Unit Ticket Sales': 'mean',
        'Unit Cost': 'mean',
        'Profit_Margin': 'mean'
    }).round(3)

    state_stats.columns = ['Market_State_Resale_Mean', 'Market_State_Cost_Mean', 'Market_State_Margin_Mean']
    df = df.merge(state_stats, left_on='State', right_index=True, how='left')

    # Performer-level statistics (only for performers with sufficient data)
    performer_stats = df.groupby('Performer').agg({
        'Unit Ticket Sales': ['mean', 'count'],
        'Profit_Margin': 'mean'
    }).round(3)

    performer_stats.columns = ['Market_Performer_Resale_Mean', 'Market_Performer_Volume', 'Market_Performer_Margin_Mean']
    
    # Only include performers with at least 5 transactions
    performer_stats = performer_stats[performer_stats['Market_Performer_Volume'] >= 5]
    df = df.merge(performer_stats[['Market_Performer_Resale_Mean', 'Market_Performer_Margin_Mean']], 
                  left_on='Performer', right_index=True, how='left')

    # Fill missing market intelligence with median values
    market_cols = [col for col in df.columns if col.startswith('Market_')]
    for col in market_cols:
        df[col] = df[col].fillna(df[col].median())

    print(f"Created {len(market_cols)} market intelligence features")
    return df

def create_advanced_features(df):
    """Create advanced engineered features"""
    print("Creating advanced engineered features...")
    
    # Price-based features
    df['Price_Premium_Ratio'] = df['Unit Ticket Sales'] / (df['Unit Cost'] + 1e-8)
    df['Price_Difference'] = df['Unit Ticket Sales'] - df['Unit Cost']
    df['Log_Unit_Cost'] = np.log1p(df['Unit Cost'])
    df['Log_Unit_Sales'] = np.log1p(df['Unit Ticket Sales'])
    
    # Time-based features
    df['Days_Until_Event_Squared'] = df['Days_Until_Event'] ** 2
    df['Days_Until_Event_Log'] = np.log1p(np.maximum(df['Days_Until_Event'], 1))
    df['Is_Peak_Time'] = ((df['Days_Until_Event'] >= 7) & (df['Days_Until_Event'] <= 30)).astype(int)
    df['Is_Super_Last_Minute'] = (df['Days_Until_Event'] < 3).astype(int)
    df['Is_Month_End'] = (df['Invoice_Date_DT'].dt.day > 25).astype(int)
    
    # Seasonal features
    df['Event_Quarter'] = df['Event_Date_DT'].dt.quarter
    df['Invoice_Quarter'] = df['Invoice_Date_DT'].dt.quarter
    df['Is_Holiday_Season'] = df['Event_Month'].isin([11, 12]).astype(int)
    df['Is_Summer_Season'] = df['Event_Month'].isin([6, 7, 8]).astype(int)
    df['Is_Spring_Season'] = df['Event_Month'].isin([3, 4, 5]).astype(int)
    
    # Market competition features - FIXED column names
    df['Relative_Price_To_Category'] = df['Unit Ticket Sales'] / (df['Market_Category_Resale_Mean'] + 1e-8)
    df['Relative_Price_To_Venue'] = df['Unit Ticket Sales'] / (df['Market_Venue_Resale_Mean'] + 1e-8)
    df['Category_Market_Position'] = pd.cut(df['Relative_Price_To_Category'], 
                                           bins=[0, 0.8, 1.2, float('inf')], 
                                           labels=[0, 1, 2]).astype(float)
    
    # Volume and popularity features
    df['Is_High_Volume_Category'] = (df['Market_Category_Volume'] > df['Market_Category_Volume'].median()).astype(int)
    df['Is_High_Volume_Venue'] = (df['Market_Venue_Volume'] > df['Market_Venue_Volume'].median()).astype(int)
    df['Category_Popularity_Score'] = np.log1p(df['Market_Category_Volume'])
    df['Venue_Popularity_Score'] = np.log1p(df['Market_Venue_Volume'])
    
    # Risk indicators
    df['Price_Volatility_Risk'] = (df['Market_Category_Resale_Std'] > df['Market_Category_Resale_Std'].median()).astype(int)
    df['Margin_Volatility_Risk'] = (df['Market_Category_Margin_Std'] > df['Market_Category_Margin_Std'].median()).astype(int)
    
    # Interaction features
    df['Category_x_Weekend'] = df['Is_Weekend_Event'] * df['Market_Category_Resale_Mean']
    df['Category_x_Peak_Season'] = df['Is_Peak_Season'] * df['Market_Category_Resale_Mean']
    df['Lead_Time_x_Category_Volume'] = df['Days_Until_Event'] * df['Market_Category_Volume']
    
    # Quantity-based features
    df['QTY_Log'] = np.log1p(df['QTY'])
    df['Is_Bulk_Order'] = (df['QTY'] > 2).astype(int)
    df['QTY_x_Unit_Cost'] = df['QTY'] * df['Unit Cost']
    
    print("Advanced feature engineering completed")
    return df

def prepare_features(self, df):
    """Enhanced feature preparation with advanced engineering"""
    print("Preparing features for customer purchase pricing...")
    
    # Define feature categories
    categorical_features = ['Event Type', 'Category', 'State', 'Venue', 'Performer']
    
    basic_numerical_features = [
        'QTY', 'Days_Until_Event', 'Event_Month', 'Invoice_Month',
        'Is_Weekend_Event', 'Is_Peak_Season', 'Is_Last_Minute', 'Is_Very_Early'
    ]

    # Create market intelligence features
    df = create_purchase_intelligence(df)
    
    # Create advanced engineered features
    df = create_advanced_features(df)
    
    # Handle missing values in categorical features
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')
        # Limit number of unique categories to prevent overfitting
        value_counts = df[col].value_counts()
        if len(value_counts) > 50:  # If too many categories
            top_categories = value_counts.head(49).index.tolist()
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
    
    # Handle missing values in numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical features
    df_encoded = df.copy()
    self.label_encoders = getattr(self, 'label_encoders', {})

    for col in categorical_features:
        if col not in self.label_encoders:
            self.label_encoders[col] = LabelEncoder()
            df_encoded[col + '_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
        else:
            # Handle new categories in prediction phase
            try:
                df_encoded[col + '_encoded'] = self.label_encoders[col].transform(df_encoded[col].astype(str))
            except ValueError:
                # If new categories appear, assign them to 'Unknown'
                known_categories = set(self.label_encoders[col].classes_)
                df_encoded[col] = df_encoded[col].apply(lambda x: x if x in known_categories else 'Unknown')
                df_encoded[col + '_encoded'] = self.label_encoders[col].transform(df_encoded[col].astype(str))

    # Collect all features
    encoded_cols = [col + '_encoded' for col in categorical_features]
    market_cols = [col for col in df.columns if col.startswith('Market_')]
    
    # Advanced engineered features
    advanced_features = [
        'Price_Premium_Ratio', 'Price_Difference', 'Log_Unit_Cost', 'Log_Unit_Sales',
        'Days_Until_Event_Squared', 'Days_Until_Event_Log', 'Is_Peak_Time', 'Is_Super_Last_Minute',
        'Is_Month_End', 'Event_Quarter', 'Invoice_Quarter', 'Is_Holiday_Season', 
        'Is_Summer_Season', 'Is_Spring_Season', 'Relative_Price_To_Category', 
        'Relative_Price_To_Venue', 'Category_Market_Position', 'Is_High_Volume_Category',
        'Is_High_Volume_Venue', 'Category_Popularity_Score', 'Venue_Popularity_Score',
        'Price_Volatility_Risk', 'Margin_Volatility_Risk', 'Category_x_Weekend',
        'Category_x_Peak_Season', 'Lead_Time_x_Category_Volume', 'QTY_Log',
        'Is_Bulk_Order', 'QTY_x_Unit_Cost'
    ]
    
    # Filter advanced features that actually exist in the dataframe
    existing_advanced_features = [col for col in advanced_features if col in df_encoded.columns]
    
    # Combine all feature sets
    self.feature_columns = basic_numerical_features + encoded_cols + market_cols + existing_advanced_features
    
    # Remove any features that don't exist in the dataframe
    self.feature_columns = [col for col in self.feature_columns if col in df_encoded.columns]
    
    print(f"Feature set: {len(self.feature_columns)} features")
    print(f"  - Basic numerical: {len(basic_numerical_features)}")
    print(f"  - Categorical encoded: {len(encoded_cols)}")
    print(f"  - Market intelligence: {len(market_cols)}")
    print(f"  - Advanced engineered: {len(existing_advanced_features)}")
    
    return df_encoded

def add_market_intelligence_for_prediction(input_data, category, venue):
    """Enhanced market intelligence for prediction phase"""
    # Get defaults from config
    defaults = CATEGORY_DEFAULTS.get(category, {
        'resale_mean': 120, 
        'margin_mean': 0.25, 
        'payment_ratio': 0.80
    })

    # Basic market intelligence
    input_data['Market_Category_Resale_Mean'] = defaults['resale_mean']
    input_data['Market_Category_Resale_Std'] = defaults['resale_mean'] * 0.3  # 30% std
    input_data['Market_Category_Resale_Median'] = defaults['resale_mean'] * 0.95
    input_data['Market_Category_Volume'] = 100  # Default volume
    
    input_data['Market_Category_Cost_Mean'] = defaults['resale_mean'] * defaults['payment_ratio']
    input_data['Market_Category_Cost_Std'] = input_data['Market_Category_Cost_Mean'] * 0.25
    input_data['Market_Category_Cost_Median'] = input_data['Market_Category_Cost_Mean'] * 0.98
    
    input_data['Market_Category_Margin_Mean'] = defaults['margin_mean']
    input_data['Market_Category_Margin_Std'] = defaults['margin_mean'] * 0.4
    input_data['Market_Category_Margin_Median'] = defaults['margin_mean'] * 0.9
    
    input_data['Market_Category_Payment_Ratio_Mean'] = defaults['payment_ratio']
    input_data['Market_Category_Payment_Ratio_Std'] = 0.1
    input_data['Market_Category_Avg_Lead_Time'] = 30
    
    # Venue intelligence
    input_data['Market_Venue_Resale_Mean'] = defaults['resale_mean']
    input_data['Market_Venue_Resale_Std'] = defaults['resale_mean'] * 0.25
    input_data['Market_Venue_Volume'] = 50
    input_data['Market_Venue_Cost_Mean'] = input_data['Market_Category_Cost_Mean']
    input_data['Market_Venue_Margin_Mean'] = defaults['margin_mean']
    input_data['Market_Venue_Margin_Std'] = defaults['margin_mean'] * 0.3
    input_data['Market_Venue_Avg_Lead_Time'] = 30
    
    # State and performer intelligence (use category defaults)
    input_data['Market_State_Resale_Mean'] = defaults['resale_mean']
    input_data['Market_State_Cost_Mean'] = input_data['Market_Category_Cost_Mean']
    input_data['Market_State_Margin_Mean'] = defaults['margin_mean']
    
    input_data['Market_Performer_Resale_Mean'] = defaults['resale_mean']
    input_data['Market_Performer_Margin_Mean'] = defaults['margin_mean']
    
    return input_data

def create_prediction_features(input_data):
    """Create advanced features for prediction data"""
    
    # Ensure required columns exist
    if 'Unit Cost' not in input_data.columns:
        input_data['Unit Cost'] = input_data['Market_Category_Cost_Mean']
    if 'Unit Ticket Sales' not in input_data.columns:
        input_data['Unit Ticket Sales'] = input_data['Market_Category_Resale_Mean']
    
    # Create advanced features (same as in training)
    input_data['Price_Premium_Ratio'] = input_data['Unit Ticket Sales'] / (input_data['Unit Cost'] + 1e-8)
    input_data['Price_Difference'] = input_data['Unit Ticket Sales'] - input_data['Unit Cost']
    input_data['Log_Unit_Cost'] = np.log1p(input_data['Unit Cost'])
    input_data['Log_Unit_Sales'] = np.log1p(input_data['Unit Ticket Sales'])
    
    # Time features
    input_data['Days_Until_Event_Squared'] = input_data['Days_Until_Event'] ** 2
    input_data['Days_Until_Event_Log'] = np.log1p(np.maximum(input_data['Days_Until_Event'], 1))
    input_data['Is_Peak_Time'] = ((input_data['Days_Until_Event'] >= 7) & (input_data['Days_Until_Event'] <= 30)).astype(int)
    input_data['Is_Super_Last_Minute'] = (input_data['Days_Until_Event'] < 3).astype(int)
    input_data['Is_Month_End'] = 0  # Default for prediction
    
    # Seasonal features
    input_data['Event_Quarter'] = ((input_data['Event_Month'] - 1) // 3) + 1
    input_data['Invoice_Quarter'] = input_data['Event_Quarter']  # Assume same for prediction
    input_data['Is_Holiday_Season'] = input_data['Event_Month'].isin([11, 12]).astype(int)
    input_data['Is_Summer_Season'] = input_data['Event_Month'].isin([6, 7, 8]).astype(int)
    input_data['Is_Spring_Season'] = input_data['Event_Month'].isin([3, 4, 5]).astype(int)
    
    # Market competition features
    input_data['Relative_Price_To_Category'] = input_data['Unit Ticket Sales'] / (input_data['Market_Category_Resale_Mean'] + 1e-8)
    input_data['Relative_Price_To_Venue'] = input_data['Unit Ticket Sales'] / (input_data['Market_Venue_Resale_Mean'] + 1e-8)
    input_data['Category_Market_Position'] = 1.0  # Default middle position
    
    # Volume features
    input_data['Is_High_Volume_Category'] = (input_data['Market_Category_Volume'] > 50).astype(int)
    input_data['Is_High_Volume_Venue'] = (input_data['Market_Venue_Volume'] > 25).astype(int)
    input_data['Category_Popularity_Score'] = np.log1p(input_data['Market_Category_Volume'])
    input_data['Venue_Popularity_Score'] = np.log1p(input_data['Market_Venue_Volume'])
    
    # Risk indicators
    input_data['Price_Volatility_Risk'] = 0  # Default low risk
    input_data['Margin_Volatility_Risk'] = 0  # Default low risk
    
    # Interaction features
    input_data['Category_x_Weekend'] = input_data['Is_Weekend_Event'] * input_data['Market_Category_Resale_Mean']
    input_data['Category_x_Peak_Season'] = input_data['Is_Peak_Season'] * input_data['Market_Category_Resale_Mean']
    input_data['Lead_Time_x_Category_Volume'] = input_data['Days_Until_Event'] * input_data['Market_Category_Volume']
    
    # Quantity features
    input_data['QTY_Log'] = np.log1p(input_data['QTY'])
    input_data['Is_Bulk_Order'] = (input_data['QTY'] > 2).astype(int)
    input_data['QTY_x_Unit_Cost'] = input_data['QTY'] * input_data['Unit Cost']
    
    return input_data