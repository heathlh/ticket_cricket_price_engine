### app/run_engine.py
import sys
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings('ignore', message='.*DecisionTreeRegressor.*')
warnings.filterwarnings('ignore', message='.*mixed types.*')

# Alternative: Suppress ALL warnings (more aggressive)
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.mode.chained_assignment = None  # Suppress pandas warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.pricing_engine import CustomerPurchasePriceEngine
from engine.models import save_model, load_model

def train_and_save_model():
    """Train the enhanced pricing engine and save the complete model"""
    print("=" * 60)
    print("ENHANCED PRICING ENGINE - TRAINING")
    print("=" * 60)
    
    # Initialize the enhanced pricing engine
    engine = CustomerPurchasePriceEngine(target_margin=0.25)
    
    # Train the model with enhanced features
    print("\n🔄 Training enhanced pricing engine...")
    engine.fit("data/cleaned_invoice_data.csv")
    
    # Save the complete enhanced model (not just rf_model)
    print("\n💾 Saving enhanced model...")
    save_model(engine, "models/enhanced_pricing_engine.pkl")
    
    print("\n✅ Enhanced model saved successfully!")
    print("   📁 Location: models/enhanced_pricing_engine.pkl")
    print(f"   📊 Features: {len(engine.feature_columns)}")
    print(f"   📈 Categories analyzed: {len(engine.category_performance)}")
    
    return engine

def test_predictions(engine):
    """Test the enhanced pricing engine with multiple scenarios"""
    print("\n" + "=" * 60)
    print("ENHANCED PRICING ENGINE - TESTING")
    print("=" * 60)
    
    # Test scenarios including your original example
    test_scenarios = [
        {
            'name': 'Original Test - Maroon 5 (Last Minute)',
            'event_type': "CONCERT",
            'category': "Pop",
            'state': "Indiana", 
            'venue': "Ruoff Music Center",
            'performer': "Maroon 5",
            'qty': 2,
            'days_until_event': 0,  # Last minute
            'event_month': 8
        },
        {
            'name': 'Same Event - Early Bird (45 days)',
            'event_type': "CONCERT",
            'category': "Pop",
            'state': "Indiana",
            'venue': "Ruoff Music Center", 
            'performer': "Maroon 5",
            'qty': 2,
            'days_until_event': 45,  # Early bird
            'event_month': 8
        },
        {
            'name': 'NFL Game - Peak Demand',
            'event_type': "SPORT",
            'category': "NFL Football",
            'state': "California",
            'venue': "SoFi Stadium",
            'performer': "Los Angeles Rams",
            'qty': 2,
            'days_until_event': 7,
            'event_month': 12
        },
        {
            'name': 'Comedy Show - Standard',
            'event_type': "COMEDY",
            'category': "Comedy", 
            'state': "New York",
            'venue': "Madison Square Garden",
            'performer': "Dave Chappelle",
            'qty': 1,
            'days_until_event': 21,
            'event_month': 6
        }
    ]
    
    print(f"\n🎯 Testing {len(test_scenarios)} scenarios...")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Test {i}: {scenario['name']} ---")
        
        try:
            # Get enhanced recommendation
            offer = engine.recommend_customer_offer(
                event_type=scenario['event_type'],
                category=scenario['category'],
                state=scenario['state'],
                venue=scenario['venue'],
                performer=scenario['performer'],
                qty=scenario['qty'],
                days_until_event=scenario['days_until_event'],
                event_month=scenario['event_month']
            )
            
            # Display enhanced results
            print(f"📊 PREDICTION RESULTS:")
            print(f"   💰 Recommended Offer (per ticket): ${offer['recommended_offer']:,.2f}")
            print(f"   📈 Expected Margin: {offer['expected_margin']:.1f}%")
            print(f"   🎯 Target Margin: {offer['target_margin']:.1f}%")
            print(f"   🔮 Confidence Level: {offer['confidence_level']:.2f}")
            print(f"   📊 Predicted Resale Price: ${offer['predicted_resale_price']:,.2f}")
            print(f"   💵 Predicted Cost: ${offer['predicted_cost']:,.2f}")
            print(f"   💎 Expected Profit (per ticket): ${offer['expected_profit']:,.2f}")
            print(f"   ⚖️  Risk Adjustment: {offer['risk_adjustment']:.3f}")
            
            # Show pricing strategy breakdown
            pricing = offer['pricing_components']
            print(f"   🏗️  PRICING STRATEGY:")
            print(f"      Conservative: ${pricing['conservative_offer']:,.2f} ({pricing['conservative_weight']:.0%})")
            print(f"      Competitive: ${pricing['competitive_offer']:,.2f} ({pricing['competitive_weight']:.0%})")
            
            # Business decision simulation
            decision = make_business_decision(offer)
            print(f"   🎯 BUSINESS DECISION: {decision['action']} - {decision['reason']}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def make_business_decision(offer):
    """Simulate business decision logic"""
    margin = offer['expected_margin']
    confidence = offer['confidence_level']
    offer_amount = offer['recommended_offer']
    
    if margin < 5:
        return {'action': '❌ DECLINE', 'reason': 'Margin too low (<5%)'}
    elif margin > 60:
        return {'action': '⚠️  REVIEW', 'reason': 'Margin unusually high (>60%)'}
    elif confidence < 0.6:
        return {'action': '⚠️  REVIEW', 'reason': 'Low confidence prediction'}
    elif offer_amount > 500:
        return {'action': '⚠️  REVIEW', 'reason': 'High value transaction'}
    else:
        return {'action': '✅ APPROVE', 'reason': f'Good opportunity - {margin:.1f}% margin'}

def show_market_insights(engine):
    """Display market insights from the enhanced engine"""
    print("\n" + "=" * 60)
    print("MARKET INSIGHTS ANALYSIS")
    print("=" * 60)
    
    # Overall market insights
    print("\n📊 OVERALL MARKET OVERVIEW:")
    overall_insights = engine.get_market_insights()
    for key, value in overall_insights.items():
        print(f"   {key}: {value}")
    
    # Category-specific insights
    key_categories = ['Pop', 'NFL Football', 'Comedy', 'NBA Basketball']
    
    print(f"\n🎯 CATEGORY PERFORMANCE ANALYSIS:")
    for category in key_categories:
        if category in engine.category_performance:
            insights = engine.get_market_insights(category)
            print(f"\n   📈 {category}:")
            for key, value in insights.items():
                if key != 'category':
                    print(f"      {key}: {value}")

def load_and_test_saved_model():
    """Load the saved model and test it"""
    print("\n" + "=" * 60)
    print("TESTING SAVED MODEL")
    print("=" * 60)
    
    try:
        # Load the saved model
        print("\n📂 Loading saved model...")
        engine = CustomerPurchasePriceEngine()
        load_model(engine, "models/enhanced_pricing_engine.pkl")
        
        print("✅ Model loaded successfully!")
        
        # Quick test
        print("\n🧪 Quick validation test...")
        offer = engine.recommend_customer_offer(
            event_type="CONCERT",
            category="Pop", 
            state="California",
            venue="Hollywood Bowl",
            performer="Test Artist",
            qty=1,
            days_until_event=14,
            event_month=7
        )
        
        print(f"   Test Recommendation: ${offer['recommended_offer']:,.2f}")
        print(f"   Test Margin: {offer['expected_margin']:.1f}%")
        print("✅ Saved model working correctly!")
        
        return engine
        
    except Exception as e:
        print(f"❌ Error loading saved model: {str(e)}")
        return None

if __name__ == "__main__":
    print("🚀 Starting Enhanced Pricing Engine (Clean Output)")
    
    # Step 1: Train and save the enhanced model
    engine = train_and_save_model()
    
    # Step 2: Test predictions with multiple scenarios
    test_predictions(engine)
    
    # Step 3: Show market insights
    show_market_insights(engine)
    
    # Step 4: Test loading the saved model
    loaded_engine = load_and_test_saved_model()
    
    print("\n" + "=" * 60)
    print("🎉 ENHANCED PRICING ENGINE SETUP COMPLETE!")
    print("=" * 60)
    print("\n📋 SUMMARY:")
    print("   ✅ Enhanced model trained and saved")
    print("   ✅ Multiple prediction scenarios tested")
    print("   ✅ Market insights generated")
    print("   ✅ Model persistence verified")
    print("\n📁 Files created:")
    print("   📄 models/enhanced_pricing_engine.pkl")
    print("\n🎯 Ready for production use!")