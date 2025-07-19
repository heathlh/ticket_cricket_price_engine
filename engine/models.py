import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def train_dual_models(self, df):
    """Enhanced model training with faster hyperparameter optimization"""
    from engine.features import prepare_features
    
    print("Training dual models with enhanced features...")
    df_prepared = prepare_features(self, df)
    X = df_prepared[self.feature_columns]
    
    # Time-based split to avoid data leakage
    df_sorted = df_prepared.sort_values("Invoice_Date_DT")
    split_idx = int(len(df_sorted) * 0.8)

    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_resale_train = df_sorted.loc[train_idx, 'Unit Ticket Sales']
    y_resale_test = df_sorted.loc[test_idx, 'Unit Ticket Sales']
    y_cost_train = df_sorted.loc[train_idx, 'Unit Cost']
    y_cost_test = df_sorted.loc[test_idx, 'Unit Cost']

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train resale price model with good default parameters (faster)
    print("Training resale price model...")
    self.resale_model = train_fast_model(X_train, y_resale_train, "resale")
    
    # Train cost price model with good default parameters (faster)
    print("Training cost price model...")
    self.rf_model = train_fast_model(X_train, y_cost_train, "cost")

    # Make predictions
    resale_pred = self.resale_model.predict(X_test)
    cost_pred = self.rf_model.predict(X_test)

    # Enhanced performance evaluation
    print("\n" + "="*50)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    print("\nResale Price Prediction Performance:")
    evaluate_model_performance(y_resale_test, resale_pred, "Resale")
    
    print("\nCost Price Prediction Performance:")
    evaluate_model_performance(y_cost_test, cost_pred, "Cost")

    # Feature importance analysis
    print_feature_importance_analysis(self.resale_model, self.feature_columns, "Resale Model")

    return X_test, y_resale_test, y_cost_test, resale_pred, cost_pred

def train_fast_model(X_train, y_train, model_type):
    """Train Random Forest with good default parameters (much faster)"""
    
    print(f"Using optimized default parameters for {model_type} model...")
    
    # Use good default parameters instead of grid search for speed
    if model_type == "resale":
        params = {
            'n_estimators': 100,      # Reduced from 200 for speed
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    else:  # cost model
        params = {
            'n_estimators': 100,      # Reduced from 200 for speed
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    
    # Create and train model
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    
    print(f"✅ {model_type} model trained successfully!")
    
    return rf

def train_optimized_model(X_train, y_train, model_type, quick_mode=True):
    """Train Random Forest with optional hyperparameter optimization"""
    
    if quick_mode:
        return train_fast_model(X_train, y_train, model_type)
    
    # Full optimization (slower) - only use if you have time
    print(f"Optimizing {model_type} model hyperparameters (this may take 5-10 minutes)...")
    
    # Smaller parameter grid for faster search
    param_grid = {
        'n_estimators': [100, 150],          # Reduced options
        'max_depth': [15, 20],               # Reduced options
        'min_samples_split': [10],           # Fixed value
        'min_samples_leaf': [4],             # Fixed value
        'max_features': ['sqrt']             # Fixed value
    }
    
    # Use fewer CV splits for speed
    tscv = TimeSeriesSplit(n_splits=2)  # Reduced from 3
    
    # Base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Grid search with time series cross-validation
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=tscv, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1  # Show progress
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_type} model:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"Best CV score: ${-grid_search.best_score_:.2f} MAE")
    
    return grid_search.best_estimator_

def evaluate_model_performance(y_true, y_pred, model_name):
    """Comprehensive model performance evaluation"""
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    residuals = y_true - y_pred
    
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²: {r2:.3f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  Mean Residual: ${np.mean(residuals):.2f}")
    print(f"  Std Residual: ${np.std(residuals):.2f}")
    
    # Prediction accuracy by ranges
    print(f"\n  {model_name} Accuracy by Price Range:")
    price_ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, float('inf'))]
    
    for low, high in price_ranges:
        if high == float('inf'):
            mask = y_true >= low
            range_name = f"  ${low}+"
        else:
            mask = (y_true >= low) & (y_true < high)
            range_name = f"  ${low}-${high}"
        
        if mask.sum() > 5:  # At least 5 samples
            range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            range_r2 = r2_score(y_true[mask], y_pred[mask])
            range_mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            print(f"    {range_name}: MAE=${range_mae:.2f}, R²={range_r2:.3f}, MAPE={range_mape:.1f}%, n={mask.sum()}")

def print_feature_importance_analysis(model, feature_columns, model_name):
    """Print detailed feature importance analysis"""
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n{model_name} - Top 10 Feature Importances:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Feature importance categories
    categorical_features = [f for f in feature_columns if f.endswith('_encoded')]
    market_features = [f for f in feature_columns if f.startswith('Market_')]
    time_features = [f for f in feature_columns if any(time_word in f for time_word in ['Days', 'Month', 'Weekend', 'Peak', 'Minute', 'Early'])]
    
    print(f"\n{model_name} - Feature Category Importance:")
    
    if categorical_features:
        cat_importance = importance_df[importance_df['feature'].isin(categorical_features)]['importance'].sum()
        print(f"  Categorical Features: {cat_importance:.3f}")
    
    if market_features:
        market_importance = importance_df[importance_df['feature'].isin(market_features)]['importance'].sum()
        print(f"  Market Intelligence Features: {market_importance:.3f}")
    
    if time_features:
        time_importance = importance_df[importance_df['feature'].isin(time_features)]['importance'].sum()
        print(f"  Time-based Features: {time_importance:.3f}")

def save_model(self, filepath):
    """Enhanced model saving with additional metadata"""
    
    model_data = {
        'rf_model': self.rf_model,
        'resale_model': self.resale_model,
        'label_encoders': self.label_encoders,
        'feature_columns': self.feature_columns,
        'target_margin': self.target_margin,
        'historical_benchmarks': self.historical_benchmarks,
        'category_performance': getattr(self, 'category_performance', {}),
        'market_trends': getattr(self, 'market_trends', {}),
        'model_metadata': {
            'training_date': pd.Timestamp.now().isoformat(),
            'n_features': len(self.feature_columns),
            'resale_model_params': self.resale_model.get_params() if self.resale_model else None,
            'cost_model_params': self.rf_model.get_params() if self.rf_model else None
        }
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Enhanced model saved to {filepath}")
    print(f"  - Features: {len(self.feature_columns)}")
    print(f"  - Categories analyzed: {len(getattr(self, 'category_performance', {}))}")
    print(f"  - Training date: {model_data['model_metadata']['training_date']}")

def load_model(self, filepath):
    """Enhanced model loading with backward compatibility"""
    
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # Load core model components
    self.rf_model = model_data['rf_model']
    self.resale_model = model_data['resale_model']
    self.label_encoders = model_data['label_encoders']
    self.feature_columns = model_data['feature_columns']
    self.target_margin = model_data['target_margin']
    self.historical_benchmarks = model_data.get('historical_benchmarks', {})
    
    # Load enhanced features (with backward compatibility)
    self.category_performance = model_data.get('category_performance', {})
    self.market_trends = model_data.get('market_trends', {})
    
    # Display loading information
    metadata = model_data.get('model_metadata', {})
    print(f"Enhanced model loaded from {filepath}")
    if metadata:
        print(f"  - Training date: {metadata.get('training_date', 'Unknown')}")
        print(f"  - Features: {metadata.get('n_features', len(self.feature_columns))}")
        print(f"  - Categories: {len(self.category_performance)}")
    else:
        print(f"  - Features: {len(self.feature_columns)}")
        print("  - Legacy model format (some enhanced features may not be available)")

def get_feature_importance(self):
    """Enhanced feature importance analysis"""
    if self.rf_model is None:
        return None
    
    # Get importance from cost model (rf_model)
    importance_df = pd.DataFrame({
        'feature': self.feature_columns,
        'cost_model_importance': self.rf_model.feature_importances_
    })
    
    # Add resale model importance if available
    if self.resale_model is not None:
        importance_df['resale_model_importance'] = self.resale_model.feature_importances_
        importance_df['combined_importance'] = (importance_df['cost_model_importance'] + 
                                              importance_df['resale_model_importance']) / 2
        importance_df = importance_df.sort_values('combined_importance', ascending=False)
    else:
        importance_df = importance_df.sort_values('cost_model_importance', ascending=False)
    
    return importance_df

def validate_model_performance(self, validation_data_path=None):
    """Validate model performance on new data"""
    if validation_data_path is None:
        print("No validation data provided")
        return None
    
    try:
        # Load validation data
        val_df = pd.read_csv(validation_data_path)
        val_df = self._prepare_raw_data(val_df)
        
        # Prepare features
        from engine.features import prepare_features
        val_df_prepared = prepare_features(self, val_df)
        
        X_val = val_df_prepared[self.feature_columns]
        y_resale_val = val_df_prepared['Unit Ticket Sales']
        y_cost_val = val_df_prepared['Unit Cost']
        
        # Make predictions
        resale_pred = self.resale_model.predict(X_val)
        cost_pred = self.rf_model.predict(X_val)
        
        print("\n" + "="*50)
        print("VALIDATION PERFORMANCE")
        print("="*50)
        
        print("\nValidation Resale Price Performance:")
        evaluate_model_performance(y_resale_val, resale_pred, "Validation Resale")
        
        print("\nValidation Cost Price Performance:")
        evaluate_model_performance(y_cost_val, cost_pred, "Validation Cost")
        
        return {
            'resale_mae': mean_absolute_error(y_resale_val, resale_pred),
            'resale_r2': r2_score(y_resale_val, resale_pred),
            'cost_mae': mean_absolute_error(y_cost_val, cost_pred),
            'cost_r2': r2_score(y_cost_val, cost_pred),
            'validation_samples': len(X_val)
        }
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        return None