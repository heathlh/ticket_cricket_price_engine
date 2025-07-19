import pandas as pd
import numpy as np
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
        self.category_performance = {}
        self.market_trends = {}

    def fit(self, csv_file):
        """Enhanced training method with comprehensive analysis"""
        df = pd.read_csv(csv_file)
        df = self._prepare_raw_data(df)
        
        # Calculate category performance benchmarks
        self._calculate_category_benchmarks(df)
        
        # Analyze market trends
        self._analyze_market_trends(df)
        
        # Train models
        X_test, y1, y2, r1, r2 = train_dual_models(self, df)
        
        # Get and display feature importance
        importance = get_feature_importance(self)
        print("\nTop 10 Feature Importances:\n", importance.head(10))
        
        # Detailed performance evaluation
        self._detailed_performance_evaluation(X_test, y1, y2, r1, r2)
        
        return self

    def _prepare_raw_data(self, df):
        """Enhanced data preparation with outlier handling"""
        # Basic filtering
        df = df.dropna(subset=['Unit Ticket Sales', 'Unit Cost'])
        df = df[(df['Unit Ticket Sales'] > 0) & (df['Unit Cost'] > 0)]
        
        # Remove extreme outliers using IQR method
        df = self._handle_outliers(df, 'Unit Ticket Sales')
        df = self._handle_outliers(df, 'Unit Cost')
        
        # Calculate derived features
        df['Profit'] = df['Unit Ticket Sales'] - df['Unit Cost']
        df['Profit_Margin'] = df['Profit'] / df['Unit Cost']
        df['Customer_Payment_Ratio'] = df['Unit Cost'] / df['Unit Ticket Sales']
        
        # Filter extreme profit margins (beyond reasonable business range)
        df = df[(df['Profit_Margin'] >= -0.8) & (df['Profit_Margin'] <= 5.0)]
        
        # Date processing
        df['Invoice_Date_DT'] = pd.to_datetime(df['Invoice_Date_DT'], errors='coerce')
        df['Event_Date_DT'] = pd.to_datetime(df['Event_Date_DT'], errors='coerce')
        df['Days_Until_Event'] = (df['Event_Date_DT'] - df['Invoice_Date_DT']).dt.days.fillna(30)
        df['Event_Month'] = df['Event_Date_DT'].dt.month
        df['Invoice_Month'] = df['Invoice_Date_DT'].dt.month
        
        # Boolean features
        df['Is_Weekend_Event'] = df['Event_Date_DT'].dt.dayofweek.isin([5, 6])
        df['Is_Peak_Season'] = df['Event_Month'].isin([8, 9, 10, 11, 12])
        df['Is_Last_Minute'] = df['Days_Until_Event'] < 7
        df['Is_Very_Early'] = df['Days_Until_Event'] > 60
        
        print(f"Data prepared: {len(df)} records after cleaning")
        return df

    def _handle_outliers(self, df, column):
        """Remove outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_count = len(df)
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed_count = initial_count - len(df_filtered)
        
        if removed_count > 0:
            print(f"Removed {removed_count} outliers from {column} ({removed_count/initial_count*100:.1f}%)")
        
        return df_filtered

    def _calculate_category_benchmarks(self, df):
        """Calculate historical performance benchmarks for each category"""
        self.category_performance = {}
        
        for category in df['Category'].unique():
            if pd.isna(category):
                continue
                
            cat_data = df[df['Category'] == category]
            if len(cat_data) < 10:  # Skip categories with insufficient data
                continue
            
            # Calculate key metrics
            margins = cat_data['Profit_Margin']
            
            self.category_performance[category] = {
                'avg_margin': margins.mean(),
                'median_margin': margins.median(),
                'std_margin': margins.std(),
                'avg_cost': cat_data['Unit Cost'].mean(),
                'avg_sales': cat_data['Unit Ticket Sales'].mean(),
                'sample_size': len(cat_data),
                'success_rate': (margins > 0).mean(),  # Percentage of profitable orders
                'high_margin_rate': (margins > 0.3).mean(),  # Percentage of high-margin orders
            }

    def _analyze_market_trends(self, df):
        """Analyze market trends and seasonality"""
        df['month_year'] = df['Invoice_Date_DT'].dt.to_period('M')
        
        # Monthly analysis
        monthly_stats = df.groupby('month_year').agg({
            'Unit Ticket Sales': ['mean', 'count'],
            'Unit Cost': 'mean'
        }).fillna(0)
        
        # Seasonal analysis
        seasonal_stats = df.groupby(df['Event_Date_DT'].dt.month).agg({
            'Unit Ticket Sales': 'mean',
            'Unit Cost': 'mean'
        }).fillna(0)
        
        self.market_trends = {
            'monthly_trends': monthly_stats,
            'seasonal_patterns': seasonal_stats,
            'overall_growth': self._calculate_growth_trend(monthly_stats)
        }

    def _calculate_growth_trend(self, monthly_stats):
        """Calculate overall growth trend using linear regression"""
        if len(monthly_stats) < 6:
            return 0
        
        values = monthly_stats[('Unit Ticket Sales', 'mean')].values
        months = range(len(values))
        
        # Simple linear regression to calculate slope
        n = len(months)
        sum_x = sum(months)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(months, values))
        sum_x2 = sum(x * x for x in months)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

# Emergency fix for pricing_engine.py - Replace the recommend_customer_offer method

    def recommend_customer_offer(self, event_type, category, state, venue, performer, qty=1, days_until_event=30, event_month=6):
        """Enhanced customer offer recommendation with sanity checks and fallback logic"""
        
        try:
            # Get original model prediction
            raw_prediction = self._get_raw_model_prediction(event_type, category, state, venue, performer, qty, days_until_event, event_month)
            
            # Check if prediction is reasonable
            if self._is_prediction_reasonable(raw_prediction):
                return raw_prediction
            else:
                print(f"⚠️ Model prediction unreasonable (${raw_prediction['recommended_offer']:.2f}), using business logic fallback")
                return self._get_business_logic_recommendation(event_type, category, state, venue, performer, qty, days_until_event, event_month)
                
        except Exception as e:
            print(f"⚠️ Model prediction failed: {str(e)}, using business logic fallback")
            return self._get_business_logic_recommendation(event_type, category, state, venue, performer, qty, days_until_event, event_month)

    def _get_raw_model_prediction(self, event_type, category, state, venue, performer, qty, days_until_event, event_month):
        """Get raw prediction from the trained models"""
        
        if self.rf_model is None or self.resale_model is None:
            raise ValueError("Model not trained. Please call fit() first.")

        # Prepare input data (original logic)
        input_data = pd.DataFrame({
            'Event Type': [event_type], 'Category': [category], 'State': [state],
            'Venue': [venue], 'Performer': [performer], 'QTY': [qty],
            'Days_Until_Event': [days_until_event], 'Event_Month': [event_month],
            'Is_Weekend_Event': [event_month in [6, 7]],
            'Is_Peak_Season': [event_month in [8, 9, 10, 11, 12]],
            'Is_Last_Minute': [days_until_event < 7],
            'Is_Very_Early': [days_until_event > 60]
        })

        from engine.features import add_market_intelligence_for_prediction, create_prediction_features
        input_data = add_market_intelligence_for_prediction(input_data, category, venue)
        input_data = create_prediction_features(input_data)

        # Encode categorical features
        for col in ['Event Type', 'Category', 'State', 'Venue', 'Performer']:
            if col in self.label_encoders:
                try:
                    input_data[col + '_encoded'] = self.label_encoders[col].transform(input_data[col].astype(str))
                except ValueError:
                    input_data[col + '_encoded'] = 0

        # Ensure all features are present
        for col in self.feature_columns:
            if col not in input_data:
                input_data[col] = 0

        # Make predictions
        X_input = input_data[self.feature_columns]
        predicted_resale = self.resale_model.predict(X_input)[0]
        predicted_cost = self.rf_model.predict(X_input)[0]
        
        # Get prediction confidence
        resale_confidence = self._get_prediction_confidence(self.resale_model, X_input)
        cost_confidence = self._get_prediction_confidence(self.rf_model, X_input)
        
        # Dynamic target margin
        dynamic_margin = self._get_dynamic_target_margin(category, days_until_event, event_month)
        
        # Risk adjustment
        risk_multiplier = self._calculate_risk_multiplier(category, days_until_event, resale_confidence, cost_confidence)
        
        # Calculate recommended offer (original logic)
        conservative_offer = predicted_cost * (1 + dynamic_margin)
        competitive_offer = predicted_resale * (1 - dynamic_margin - 0.05)
        
        weight_conservative = 0.3 + (0.4 if days_until_event < 7 else 0)
        weight_competitive = 1 - weight_conservative
        
        recommended_offer = (conservative_offer * weight_conservative + 
                        competitive_offer * weight_competitive) * risk_multiplier
        
        final_offer = max(recommended_offer, predicted_cost * 1.05)
        
        # Calculate metrics
        expected_profit = predicted_resale - final_offer
        expected_margin = (expected_profit / final_offer) * 100 if final_offer > 0 else 0
        confidence = min(resale_confidence, cost_confidence)

        return {
            'predicted_resale_price': round(predicted_resale, 2),
            'predicted_cost': round(predicted_cost, 2),
            'recommended_offer': round(final_offer, 2),
            'expected_profit': round(expected_profit, 2),
            'expected_margin': round(expected_margin, 1),
            'target_margin': round(dynamic_margin * 100, 1),
            'confidence_level': round(confidence, 2),
            'risk_adjustment': round(risk_multiplier, 3),
            'pricing_method': 'model_prediction',
            'pricing_components': {
                'conservative_offer': round(conservative_offer * weight_conservative, 2),
                'competitive_offer': round(competitive_offer * weight_competitive, 2),
                'conservative_weight': round(weight_conservative, 2),
                'competitive_weight': round(weight_competitive, 2)
            }
        }

    def _is_prediction_reasonable(self, prediction):
        """Check if model prediction is reasonable"""
        
        # Sanity check thresholds
        MIN_REASONABLE_PRICE = 15   # Minimum ticket price
        MAX_REASONABLE_PRICE = 2000 # Maximum ticket price  
        MIN_COST = 5               # Minimum cost
        MAX_MARGIN = 300           # Maximum margin (300%)
        MIN_MARGIN = -50           # Minimum margin (-50% loss)
        
        offer = prediction['recommended_offer']
        cost = prediction['predicted_cost']
        margin = prediction['expected_margin']
        
        return (
            MIN_REASONABLE_PRICE <= offer <= MAX_REASONABLE_PRICE and
            cost >= MIN_COST and
            MIN_MARGIN <= margin <= MAX_MARGIN
        )

    def _get_business_logic_recommendation(self, event_type, category, state, venue, performer, qty, days_until_event, event_month):
        """Business logic fallback when model predictions are unreasonable"""
        
        from engine.config import CATEGORY_DEFAULTS
        
        print(f"📊 Using business logic for: {category} - {performer} - {venue}")
        
        # Category-based baseline pricing
        category_baselines = {
            'Pop': {'base_cost': 60, 'base_resale': 95, 'margin': 0.35},
            'Rock': {'base_cost': 45, 'base_resale': 75, 'margin': 0.40},
            'Country and Folk': {'base_cost': 50, 'base_resale': 80, 'margin': 0.35},
            'NFL Football': {'base_cost': 120, 'base_resale': 180, 'margin': 0.25},
            'NBA Basketball': {'base_cost': 80, 'base_resale': 125, 'margin': 0.30},
            'Comedy': {'base_cost': 35, 'base_resale': 65, 'margin': 0.45},
            'Classical': {'base_cost': 45, 'base_resale': 75, 'margin': 0.35},
            'Alternative': {'base_cost': 30, 'base_resale': 55, 'margin': 0.45},
            'Rap/Hip Hop': {'base_cost': 55, 'base_resale': 85, 'margin': 0.35},
            'NHL Hockey': {'base_cost': 70, 'base_resale': 110, 'margin': 0.30},
            'MLB Baseball': {'base_cost': 40, 'base_resale': 70, 'margin': 0.35}
        }
        
        # Get baseline or default
        baseline = category_baselines.get(category, {'base_cost': 50, 'base_resale': 80, 'margin': 0.30})
        
        base_cost = baseline['base_cost']
        base_resale = baseline['base_resale']
        base_margin = baseline['margin']
        
        # Venue adjustments
        venue_multiplier = 1.0
        venue_lower = venue.lower()
        
        if any(premium in venue_lower for premium in ['madison square garden', 'staples center', 'msg', 'wembley']):
            venue_multiplier = 1.4
        elif any(major in venue_lower for major in ['center', 'arena', 'stadium', 'garden']):
            venue_multiplier = 1.1
        elif any(outdoor in venue_lower for outdoor in ['amphitheater', 'pavilion', 'ruoff', 'lawn']):
            venue_multiplier = 0.9
        
        # Performer adjustments
        performer_multiplier = 1.0
        performer_lower = performer.lower()
        
        # A-list performers
        if any(star in performer_lower for star in ['taylor swift', 'beyonce', 'drake', 'adele']):
            performer_multiplier = 1.8
        # Popular performers  
        elif any(popular in performer_lower for popular in ['maroon 5', 'coldplay', 'imagine dragons', 'ed sheeran']):
            performer_multiplier = 1.3
        # Sports teams (already adjusted by category)
        elif any(team in performer_lower for team in ['lakers', 'warriors', 'patriots', 'cowboys']):
            performer_multiplier = 1.2
        
        # Time-based adjustments
        time_multiplier = 1.0
        if days_until_event <= 1:
            time_multiplier = 1.3  # Same day premium
        elif days_until_event <= 7:
            time_multiplier = 1.1  # Last week premium
        elif days_until_event > 90:
            time_multiplier = 0.9  # Far future discount
        
        # Seasonal adjustments  
        seasonal_multiplier = 1.0
        if event_month in [11, 12]:  # Holiday season
            seasonal_multiplier = 1.15
        elif event_month in [6, 7, 8]:  # Summer
            seasonal_multiplier = 1.05
        
        # Calculate final prices
        adjusted_cost = base_cost * venue_multiplier * performer_multiplier * time_multiplier * seasonal_multiplier
        adjusted_resale = base_resale * venue_multiplier * performer_multiplier * time_multiplier * seasonal_multiplier
        
        # Calculate recommended offer
        target_margin = base_margin
        if days_until_event < 7:
            target_margin *= 0.9  # Slightly lower margin for urgency
        
        recommended_offer = adjusted_cost * (1 + target_margin)
        
        # Ensure reasonable bounds
        recommended_offer = max(recommended_offer, adjusted_cost * 1.05)  # At least 5% profit
        recommended_offer = min(recommended_offer, adjusted_resale * 0.9)  # Max 90% of resale
        
        # Calculate final metrics
        expected_profit = adjusted_resale - recommended_offer
        expected_margin = (expected_profit / recommended_offer) * 100 if recommended_offer > 0 else 0
        
        return {
            'predicted_resale_price': round(adjusted_resale, 2),
            'predicted_cost': round(adjusted_cost, 2),
            'recommended_offer': round(recommended_offer, 2),
            'expected_profit': round(expected_profit, 2),
            'expected_margin': round(expected_margin, 1),
            'target_margin': round(target_margin * 100, 1),
            'confidence_level': 0.75,  # Business logic confidence
            'risk_adjustment': 1.0,
            'pricing_method': 'business_logic_fallback',
            'adjustments': {
                'venue_multiplier': round(venue_multiplier, 2),
                'performer_multiplier': round(performer_multiplier, 2),
                'time_multiplier': round(time_multiplier, 2),
                'seasonal_multiplier': round(seasonal_multiplier, 2)
            },
            'pricing_components': {
                'conservative_offer': round(recommended_offer * 0.4, 2),
                'competitive_offer': round(recommended_offer * 0.6, 2),
                'conservative_weight': 0.4,
                'competitive_weight': 0.6
            }
        }
    
    def _get_dynamic_target_margin(self, category, days_until_event, event_month):
        """Calculate dynamic target margin based on category and market conditions"""
        # Base margin from historical performance
        if category in self.category_performance:
            base_margin = self.category_performance[category]['avg_margin']
        else:
            base_margin = self.target_margin
        
        # Time-based adjustments
        if days_until_event < 7:
            base_margin *= 0.8  # Lower margin for last-minute sales
        elif days_until_event > 60:
            base_margin *= 1.1  # Higher margin for early sales
        
        # Seasonal adjustments
        if event_month in [11, 12, 6, 7]:  # Peak season
            base_margin *= 1.05
        
        return max(0.05, min(0.60, base_margin))  # Constrain between 5%-60%

    def _calculate_risk_multiplier(self, category, days_until_event, resale_confidence, cost_confidence):
        """Calculate risk adjustment multiplier"""
        base_risk = 1.0
        
        # Confidence-based adjustment
        avg_confidence = (resale_confidence + cost_confidence) / 2
        if avg_confidence < 0.7:
            base_risk *= 0.95  # More conservative when confidence is low
        
        # Time-based risk
        if days_until_event < 3:
            base_risk *= 0.9   # Last-minute risk
        elif days_until_event > 90:
            base_risk *= 0.95  # Long-term uncertainty
        
        # Category-based risk
        if category in self.category_performance:
            perf = self.category_performance[category]
            if perf['std_margin'] > 0.4:  # High volatility category
                base_risk *= 0.98
        
        return max(0.85, min(1.05, base_risk))  # Constrain adjustment range

    def _get_prediction_confidence(self, model, X_input):
        """Calculate prediction confidence based on tree consensus"""
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_input) for tree in model.estimators_])
        
        # Calculate standard deviation as uncertainty measure
        prediction_std = np.std(tree_predictions)
        mean_prediction = np.mean(tree_predictions)
        
        # Convert to confidence score (0-1)
        cv = prediction_std / (mean_prediction + 1e-8)  # Coefficient of variation
        confidence = max(0.1, min(0.95, 1 - cv))  # Inverse mapping to confidence
        
        return confidence

    def get_market_insights(self, category=None):
        """Get market insights for a specific category or overall market"""
        if category and category in self.category_performance:
            perf = self.category_performance[category]
            return {
                'category': category,
                'recommended_target_margin': f"{perf['avg_margin']*100:.1f}%",
                'market_position': 'high-margin' if perf['avg_margin'] > 0.4 else 'standard-margin',
                'risk_level': 'high' if perf['std_margin'] > 0.3 else 'low',
                'success_rate': f"{perf['success_rate']*100:.1f}%",
                'sample_reliability': 'high' if perf['sample_size'] > 100 else 'medium' if perf['sample_size'] > 30 else 'low'
            }
        else:
            # Return overall market overview
            return {
                'total_categories': len(self.category_performance),
                'high_performing_categories': [
                    cat for cat, perf in self.category_performance.items() 
                    if perf['avg_margin'] > 0.4 and perf['success_rate'] > 0.8
                ],
                'risky_categories': [
                    cat for cat, perf in self.category_performance.items()
                    if perf['std_margin'] > 0.4 or perf['success_rate'] < 0.7
                ],
                'market_growth_trend': 'positive' if self.market_trends.get('overall_growth', 0) > 0 else 'negative'
            }

    def _detailed_performance_evaluation(self, X_test, y_resale, y_cost, resale_pred, cost_pred):
        """Comprehensive model performance evaluation"""
        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
        
        print("\n=== Detailed Model Performance Evaluation ===")
        
        # Resale price model evaluation
        resale_mae = mean_absolute_error(y_resale, resale_pred)
        resale_rmse = np.sqrt(mean_squared_error(y_resale, resale_pred))
        resale_r2 = r2_score(y_resale, resale_pred)
        
        print(f"\nResale Price Prediction:")
        print(f"  MAE: ${resale_mae:.2f}")
        print(f"  RMSE: ${resale_rmse:.2f}")
        print(f"  R²: {resale_r2:.3f}")
        
        # Cost price model evaluation
        cost_mae = mean_absolute_error(y_cost, cost_pred)
        cost_rmse = np.sqrt(mean_squared_error(y_cost, cost_pred))
        cost_r2 = r2_score(y_cost, cost_pred)
        
        print(f"\nCost Price Prediction:")
        print(f"  MAE: ${cost_mae:.2f}")
        print(f"  RMSE: ${cost_rmse:.2f}")
        print(f"  R²: {cost_r2:.3f}")
        
        # Performance by price range
        self._performance_by_price_range(y_resale, resale_pred, "Resale Price")
        self._performance_by_price_range(y_cost, cost_pred, "Cost Price")
        
        print("\n=== Evaluation Complete ===")

    def _performance_by_price_range(self, y_true, y_pred, model_name):
        """Analyze performance by price ranges"""
        from sklearn.metrics import mean_absolute_error, r2_score
        
        ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, float('inf'))]
        
        print(f"\n{model_name} Performance by Price Range:")
        for low, high in ranges:
            if high == float('inf'):
                mask = y_true >= low
                range_name = f"${low}+"
            else:
                mask = (y_true >= low) & (y_true < high)
                range_name = f"${low}-${high}"
            
            if mask.sum() > 10:  # At least 10 samples
                range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                range_r2 = r2_score(y_true[mask], y_pred[mask])
                print(f"  {range_name}: MAE=${range_mae:.2f}, R²={range_r2:.3f}, Samples={mask.sum()}")

    def _calculate_confidence(self, category, days_until_event):
        """Legacy confidence calculation method (kept for backward compatibility)"""
        score = 0.75
        if category in ['NFL Football', 'NBA Basketball', 'Comedy']:
            score += 0.15
        if 7 <= days_until_event <= 60:
            score += 0.1
        return min(score, 0.95)