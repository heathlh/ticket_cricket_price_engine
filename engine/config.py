# Enhanced configuration file for pricing engine

# Target margin for pricing strategy
TARGET_MARGIN = 0.25

# Category defaults based on historical analysis
CATEGORY_DEFAULTS = {
    'NFL Football': {
        'resale_mean': 180, 
        'margin_mean': 0.19,  # Updated based on data analysis
        'payment_ratio': 0.80,
        'volatility': 'low',
        'seasonality': 'high'
    },
    'NBA Basketball': {
        'resale_mean': 125, 
        'margin_mean': 0.30,  # Updated based on data analysis
        'payment_ratio': 0.83,
        'volatility': 'medium',
        'seasonality': 'medium'
    },
    'Pop': {
        'resale_mean': 150,  # Updated based on analysis
        'margin_mean': 0.34,  # Updated based on data analysis
        'payment_ratio': 0.75,
        'volatility': 'high',
        'seasonality': 'low'
    },
    'Rock': {
        'resale_mean': 85,   # Updated based on analysis
        'margin_mean': 0.53,  # Updated based on data analysis
        'payment_ratio': 0.65,
        'volatility': 'high',
        'seasonality': 'low'
    },
    'Country and Folk': {
        'resale_mean': 100,  # Updated based on analysis
        'margin_mean': 0.55,  # Updated based on data analysis
        'payment_ratio': 0.65,
        'volatility': 'medium',
        'seasonality': 'medium'
    },
    'Comedy': {
        'resale_mean': 95, 
        'margin_mean': 0.52,  # Updated based on data analysis
        'payment_ratio': 0.65,
        'volatility': 'medium',
        'seasonality': 'low'
    },
    'Classical': {
        'resale_mean': 100, 
        'margin_mean': 0.40, 
        'payment_ratio': 0.71,
        'volatility': 'low',
        'seasonality': 'medium'
    },
    'Alternative': {
        'resale_mean': 75,   # Updated based on analysis
        'margin_mean': 0.52, 
        'payment_ratio': 0.66,
        'volatility': 'high',
        'seasonality': 'low'
    },
    'World Music': {
        'resale_mean': 65,   # Updated based on analysis
        'margin_mean': 0.66, 
        'payment_ratio': 0.60,
        'volatility': 'high',
        'seasonality': 'low'
    },
    'Rap/Hip Hop': {
        'resale_mean': 120,
        'margin_mean': 0.35,
        'payment_ratio': 0.75,
        'volatility': 'high',
        'seasonality': 'low'
    },
    'NHL Hockey': {
        'resale_mean': 140,
        'margin_mean': 0.25,
        'payment_ratio': 0.80,
        'volatility': 'medium',
        'seasonality': 'high'
    },
    'MLB Baseball': {
        'resale_mean': 95,
        'margin_mean': 0.30,
        'payment_ratio': 0.75,
        'volatility': 'medium',
        'seasonality': 'high'
    }
}

# Risk assessment parameters
RISK_PARAMETERS = {
    'high_volatility_threshold': 0.4,
    'low_success_rate_threshold': 0.7,
    'minimum_samples_for_reliability': 30,
    'confidence_threshold': 0.7,
    'last_minute_days': 7,
    'very_early_days': 60,
    'peak_months': [6, 7, 8, 11, 12],
    'holiday_months': [11, 12]
}

# Model hyperparameters
MODEL_PARAMETERS = {
    'random_forest': {
        'n_estimators_options': [150, 200, 250],
        'max_depth_options': [15, 20, 25, None],
        'min_samples_split_options': [5, 10, 15],
        'min_samples_leaf_options': [2, 4, 6],
        'max_features_options': ['sqrt', 'log2', None],
        'random_state': 42,
        'n_jobs': -1
    },
    'cross_validation': {
        'n_splits': 3,
        'test_size': 0.2,
        'validation_method': 'time_series_split'
    }
}

# Feature engineering parameters
FEATURE_PARAMETERS = {
    'outlier_removal': {
        'method': 'iqr',
        'iqr_multiplier': 1.5,
        'profit_margin_bounds': (-0.8, 5.0)
    },
    'categorical_encoding': {
        'max_categories_per_feature': 50,
        'min_category_frequency': 5,
        'handle_unknown': 'ignore'
    },
    'advanced_features': {
        'create_log_features': True,
        'create_interaction_features': True,
        'create_time_features': True,
        'create_market_position_features': True
    }
}

# Pricing strategy parameters
PRICING_PARAMETERS = {
    'conservative_weight_base': 0.3,
    'conservative_weight_last_minute': 0.7,
    'minimum_profit_margin': 0.05,
    'maximum_target_margin': 0.60,
    'risk_adjustment_range': (0.85, 1.05),
    'confidence_weight_threshold': 0.7
}

# Market intelligence parameters
MARKET_INTELLIGENCE = {
    'category_volume_threshold': 10,
    'venue_volume_threshold': 5,
    'performer_volume_threshold': 5,
    'state_volume_threshold': 20,
    'default_std_multiplier': 0.3,
    'default_volume_category': 100,
    'default_volume_venue': 50,
    'default_lead_time': 30
}

# Performance monitoring thresholds
PERFORMANCE_THRESHOLDS = {
    'acceptable_mae': 25.0,  # $25 MAE threshold
    'good_r2_score': 0.70,   # R² above 0.70 is good
    'excellent_r2_score': 0.85,  # R² above 0.85 is excellent
    'mape_threshold': 15.0,   # 15% MAPE threshold
    'min_validation_samples': 100
}

# Business rules
BUSINESS_RULES = {
    'minimum_offer_multiplier': 1.05,  # Always offer at least 5% above cost
    'maximum_offer_multiplier': 0.95,  # Never offer more than 95% of predicted resale
    'bulk_order_threshold': 2,
    'high_value_transaction_threshold': 500,
    'require_approval_threshold': 1000,
    'auto_decline_loss_threshold': -0.1  # Auto decline if expected loss > 10%
}

# Category groupings for analysis
CATEGORY_GROUPS = {
    'sports': [
        'NFL Football', 'NBA Basketball', 'NHL Hockey', 'MLB Baseball', 
        'NCAA Football', 'NCAA Basketball', 'MLS', 'Soccer', 'Tennis',
        'PGA Golf', 'NASCAR Racing', 'Racing', 'Boxing and Fighting',
        'Wrestling', 'WWE', 'Other Sports'
    ],
    'music': [
        'Pop', 'Rock', 'Country and Folk', 'Rap/Hip Hop', 'Classical',
        'Alternative', 'Hard Rock', 'Adult Contemporary', 'Dance/Electronica',
        'Latin Music', 'R&B', 'Blues and Jazz', 'New Age', 'World Music',
        'K-pop', 'Reggae', 'Other Concerts'
    ],
    'entertainment': [
        'Comedy', 'Musical', 'Opera', 'Broadway', 'Arts and Theater',
        'Ballet and Dance', 'Other Theater', 'Magic', 'Cirque',
        'Public Speaking', 'Family'
    ],
    'events': [
        'Music Festivals', 'Parking', 'Extreme Sports', 'Rodeo',
        'Gymnastics', 'Rugby'
    ]
}

# Seasonal adjustments by month
SEASONAL_ADJUSTMENTS = {
    1: 0.95,  # January - slower month
    2: 0.96,  # February - slower month  
    3: 1.02,  # March - spring events start
    4: 1.03,  # April - spring season
    5: 1.04,  # May - spring/early summer
    6: 1.08,  # June - peak summer
    7: 1.10,  # July - peak summer
    8: 1.07,  # August - late summer
    9: 1.05,  # September - fall events
    10: 1.06, # October - fall season
    11: 1.12, # November - holiday season
    12: 1.15  # December - peak holiday season
}

# Venue type classifications (if available)
VENUE_TYPES = {
    'stadium': ['Stadium', 'Dome', 'Field', 'Park'],
    'arena': ['Arena', 'Center', 'Coliseum', 'Garden'],
    'theater': ['Theater', 'Theatre', 'Hall', 'Opera House'],
    'club': ['Club', 'Bar', 'Lounge'],
    'outdoor': ['Amphitheater', 'Pavilion', 'Festival Grounds']
}

# Feature importance weights for manual adjustments
FEATURE_WEIGHTS = {
    'market_intelligence_weight': 1.2,  # Boost market features
    'time_features_weight': 1.1,       # Boost time-based features
    'categorical_features_weight': 1.0,  # Standard weight
    'advanced_features_weight': 0.9     # Slightly reduce complex features
}

# Logging and monitoring configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_predictions': True,
    'log_performance_metrics': True,
    'save_feature_importance': True,
    'performance_review_frequency': 'monthly'
}

# Data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    'minimum_records_training': 1000,
    'maximum_missing_percentage': 0.15,  # 15% missing data threshold
    'minimum_category_samples': 10,
    'outlier_percentage_threshold': 0.05,  # Flag if >5% outliers
    'duplicate_records_threshold': 0.01    # Flag if >1% duplicates
}