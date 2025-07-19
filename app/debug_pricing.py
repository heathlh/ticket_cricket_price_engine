# Debug script to identify pricing issues
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.pricing_engine import CustomerPurchasePriceEngine
from engine.models import load_model
import pandas as pd

def debug_pricing_issue():
    """Debug the pricing recommendation issue"""
    print("🔍 DEBUGGING PRICING ISSUE")
    print("=" * 50)
    
    # Load model
    engine = CustomerPurchasePriceEngine()
    load_model(engine, "models/enhanced_pricing_engine.pkl")
    
    # Test case that's showing low prices
    test_params = {
        'event_type': "CONCERT",
        'category': "NBA Basketball",  # This is wrong - should be "Pop" for Maroon 5
        'state': "Indiana",
        'venue': "Ruoff Music Center", 
        'performer': "Maroon 5",
        'qty': 2,
        'days_until_event': 14,  # Changed from 0 to 14
        'event_month': 8
    }
    
    print("🎯 Test Parameters:")
    for key, value in test_params.items():
        print(f"   {key}: {value}")
    
    # Check category defaults
    print(f"\n📊 Category Defaults for '{test_params['category']}':")
    from engine.config import CATEGORY_DEFAULTS
    defaults = CATEGORY_DEFAULTS.get(test_params['category'], {})
    for key, value in defaults.items():
        print(f"   {key}: {value}")
    
    # Get recommendation with detailed output
    print(f"\n💰 Getting recommendation...")
    
    try:
        recommendation = engine.recommend_customer_offer(
            event_type=test_params['event_type'],
            category=test_params['category'],
            state=test_params['state'],
            venue=test_params['venue'],
            performer=test_params['performer'],
            qty=test_params['qty'],
            days_until_event=test_params['days_until_event'],
            event_month=test_params['event_month']
        )
        
        print("\n🔍 DETAILED RECOMMENDATION BREAKDOWN:")
        print(f"   Predicted Resale Price: ${recommendation['predicted_resale_price']:,.2f}")
        print(f"   Predicted Cost: ${recommendation['predicted_cost']:,.2f}")
        print(f"   Recommended Offer: ${recommendation['recommended_offer']:,.2f}")
        print(f"   Expected Profit: ${recommendation['expected_profit']:,.2f}")
        print(f"   Expected Margin: {recommendation['expected_margin']:.1f}%")
        print(f"   Target Margin: {recommendation['target_margin']:.1f}%")
        print(f"   Confidence Level: {recommendation['confidence_level']:.3f}")
        print(f"   Risk Adjustment: {recommendation['risk_adjustment']:.3f}")
        
        # Pricing components
        pricing = recommendation['pricing_components']
        print(f"\n🏗️ PRICING STRATEGY BREAKDOWN:")
        print(f"   Conservative Offer: ${pricing['conservative_offer']:,.2f} ({pricing['conservative_weight']:.0%})")
        print(f"   Competitive Offer: ${pricing['competitive_offer']:,.2f} ({pricing['competitive_weight']:.0%})")
        
        # Analysis
        print(f"\n🧐 ANALYSIS:")
        if recommendation['predicted_cost'] < 10:
            print("   ❌ ISSUE: Predicted cost is unusually low!")
        if recommendation['predicted_resale_price'] < 50:
            print("   ❌ ISSUE: Predicted resale price is unusually low!")
        if recommendation['expected_margin'] > 100:
            print("   ❌ ISSUE: Expected margin is unrealistically high!")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None
    
    # Test with corrected category
    print(f"\n" + "=" * 50)
    print("🔄 TESTING WITH CORRECTED CATEGORY")
    print("=" * 50)
    
    corrected_params = test_params.copy()
    corrected_params['category'] = "Pop"  # Correct category for Maroon 5
    
    print("🎯 Corrected Parameters:")
    for key, value in corrected_params.items():
        print(f"   {key}: {value}")
    
    try:
        corrected_rec = engine.recommend_customer_offer(**corrected_params)
        
        print("\n💰 CORRECTED RECOMMENDATION:")
        print(f"   Predicted Resale Price: ${corrected_rec['predicted_resale_price']:,.2f}")
        print(f"   Predicted Cost: ${corrected_rec['predicted_cost']:,.2f}")
        print(f"   Recommended Offer: ${corrected_rec['recommended_offer']:,.2f}")
        print(f"   Expected Profit: ${corrected_rec['expected_profit']:,.2f}")
        print(f"   Expected Margin: {corrected_rec['expected_margin']:.1f}%")
        
        # Compare results
        print(f"\n📈 COMPARISON:")
        print(f"   Original (NBA Basketball): ${recommendation['recommended_offer']:,.2f}")
        print(f"   Corrected (Pop): ${corrected_rec['recommended_offer']:,.2f}")
        print(f"   Difference: ${corrected_rec['recommended_offer'] - recommendation['recommended_offer']:,.2f}")
        
    except Exception as e:
        print(f"   ❌ Error with corrected params: {str(e)}")
    
    # Test realistic scenario
    print(f"\n" + "=" * 50)
    print("🎯 TESTING REALISTIC SCENARIO")
    print("=" * 50)
    
    realistic_params = {
        'event_type': "CONCERT",
        'category': "Pop",
        'state': "Indiana",
        'venue': "Ruoff Music Center", 
        'performer': "Maroon 5",
        'qty': 2,
        'days_until_event': 14,
        'event_month': 8
    }
    
    try:
        realistic_rec = engine.recommend_customer_offer(**realistic_params)
        
        print("💰 REALISTIC RECOMMENDATION:")
        print(f"   Recommended Offer: ${realistic_rec['recommended_offer']:,.2f}")
        print(f"   Expected Margin: {realistic_rec['expected_margin']:.1f}%")
        print(f"   Confidence: {realistic_rec['confidence_level']:.3f}")
        
        # Check if reasonable
        if 50 <= realistic_rec['recommended_offer'] <= 300:
            print("   ✅ This looks more reasonable!")
        else:
            print("   ❌ Still looks unreasonable")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

if __name__ == "__main__":
    debug_pricing_issue()