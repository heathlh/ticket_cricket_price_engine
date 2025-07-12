### engine/pricing_engine.py

import pandas as pd
from engine.config import TARGET_MARGIN
from engine.features import prepare_features, add_market_intelligence_for_prediction
from engine.models import train_dual_models, save_model, load_model, get_feature_importance

class CustomerPurchasePriceEngine:
    def __init__(self, target_margin=TARGET_MARGIN):
        self.rf_model = None
        self.resale_model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_margin = target_margin
        self.historical_benchmarks = {}

    def fit(self, csv_file):
        df = pd.read_csv(csv_file)
        df = self._prepare_raw_data(df)
        X_test, y1, y2, r1, r2 = train_dual_models(self, df)
        importance = get_feature_importance(self)
        print("\nTop 10 Feature Importances:\n", importance.head(10))
        return self

    def _prepare_raw_data(self, df):
        df = df.dropna(subset=['Unit Ticket Sales', 'Unit Cost'])
        df = df[(df['Unit Ticket Sales'] > 0) & (df['Unit Cost'] > 0)]
        df['Profit'] = df['Unit Ticket Sales'] - df['Unit Cost']
        df['Profit_Margin'] = df['Profit'] / df['Unit Cost']
        df['Customer_Payment_Ratio'] = df['Unit Cost'] / df['Unit Ticket Sales']
        df['Invoice_Date_DT'] = pd.to_datetime(df['Invoice_Date_DT'], errors='coerce')
        df['Event_Date_DT'] = pd.to_datetime(df['Event_Date_DT'], errors='coerce')
        df['Days_Until_Event'] = (df['Event_Date_DT'] - df['Invoice_Date_DT']).dt.days.fillna(30)
        df['Event_Month'] = df['Event_Date_DT'].dt.month
        df['Invoice_Month'] = df['Invoice_Date_DT'].dt.month
        df['Is_Weekend_Event'] = df['Event_Date_DT'].dt.dayofweek.isin([5, 6])
        df['Is_Peak_Season'] = df['Event_Month'].isin([8, 9, 10, 11, 12])
        df['Is_Last_Minute'] = df['Days_Until_Event'] < 7
        df['Is_Very_Early'] = df['Days_Until_Event'] > 60
        return df

    def recommend_customer_offer(self, event_type, category, state, venue, performer, qty=1, days_until_event=30, event_month=6):
        if self.rf_model is None or self.resale_model is None:
            raise ValueError("Model not trained. Please call fit() first.")

        input_data = pd.DataFrame({
            'Event Type': [event_type], 'Category': [category], 'State': [state],
            'Venue': [venue], 'Performer': [performer], 'QTY': [qty],
            'Days_Until_Event': [days_until_event], 'Event_Month': [event_month],
            'Is_Weekend_Event': [event_month in [6, 7]],
            'Is_Peak_Season': [event_month in [8, 9, 10, 11, 12]],
            'Is_Last_Minute': [days_until_event < 7],
            'Is_Very_Early': [days_until_event > 60]
        })

        input_data = add_market_intelligence_for_prediction(input_data, category, venue)

        for col in ['Event Type', 'Category', 'State', 'Venue', 'Performer']:
            if col in self.label_encoders:
                try:
                    input_data[col + '_encoded'] = self.label_encoders[col].transform(input_data[col].astype(str))
                except ValueError:
                    input_data[col + '_encoded'] = 0

        for col in self.feature_columns:
            if col not in input_data:
                input_data[col] = 0

        X_input = input_data[self.feature_columns]
        resale_price = self.resale_model.predict(X_input)[0]
        cost_price = self.rf_model.predict(X_input)[0]

        recommended = (resale_price * (1 - self.target_margin) + cost_price) / 2
        margin = (resale_price - recommended) / recommended * 100
        confidence = self._calculate_confidence(category, days_until_event)

        return {
            'predicted_resale_price': round(resale_price, 2),
            'recommended_offer': round(recommended, 2),
            'expected_profit': round(resale_price - recommended, 2),
            'expected_margin': round(margin, 1),
            'confidence_level': confidence
        }

    def _calculate_confidence(self, category, days_until_event):
        score = 0.75
        if category in ['NFL Football', 'NBA Basketball', 'Comedy']:
            score += 0.15
        if 7 <= days_until_event <= 60:
            score += 0.1
        return min(score, 0.95)