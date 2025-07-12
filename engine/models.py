### engine/models.py

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_dual_models(self, df):
    from engine.features import prepare_features
    df_prepared = prepare_features(self, df)
    X = df_prepared[self.feature_columns]
    df_sorted = df_prepared.sort_values("Invoice_Date_DT")
    split_idx = int(len(df_sorted) * 0.8)

    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_resale_train = df_sorted.loc[train_idx, 'Unit Ticket Sales']
    y_resale_test = df_sorted.loc[test_idx, 'Unit Ticket Sales']
    y_cost_train = df_sorted.loc[train_idx, 'Unit Cost']
    y_cost_test = df_sorted.loc[test_idx, 'Unit Cost']

    self.resale_model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42, n_jobs=-1)
    self.resale_model.fit(X_train, y_resale_train)

    self.rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42, n_jobs=-1)
    self.rf_model.fit(X_train, y_cost_train)

    resale_pred = self.resale_model.predict(X_test)
    cost_pred = self.rf_model.predict(X_test)

    print("\nResale Price Prediction Performance:")
    print(f"MAE: ${mean_absolute_error(y_resale_test, resale_pred):.2f}, R²: {r2_score(y_resale_test, resale_pred):.3f}")

    print("\nCustomer Payment Prediction Performance:")
    print(f"MAE: ${mean_absolute_error(y_cost_test, cost_pred):.2f}, R²: {r2_score(y_cost_test, cost_pred):.3f}")

    return X_test, y_resale_test, y_cost_test, resale_pred, cost_pred

def save_model(self, filepath):
    model_data = {
        'rf_model': self.rf_model,
        'resale_model': self.resale_model,
        'label_encoders': self.label_encoders,
        'feature_columns': self.feature_columns,
        'target_margin': self.target_margin,
        'historical_benchmarks': self.historical_benchmarks
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")

def load_model(self, filepath):
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    self.rf_model = model_data['rf_model']
    self.resale_model = model_data['resale_model']
    self.label_encoders = model_data['label_encoders']
    self.feature_columns = model_data['feature_columns']
    self.target_margin = model_data['target_margin']
    self.historical_benchmarks = model_data.get('historical_benchmarks', {})
    print(f"Model loaded from {filepath}")

def get_feature_importance(self):
    if self.rf_model is None:
        return None
    importance_df = pd.DataFrame({
        'feature': self.feature_columns,
        'importance': self.rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    return importance_df