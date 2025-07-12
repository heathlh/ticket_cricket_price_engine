### app/run_engine.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.pricing_engine import CustomerPurchasePriceEngine

if __name__ == "__main__":
    engine = CustomerPurchasePriceEngine()
    engine.fit("data/cleaned_invoice_data.csv")

    offer = engine.recommend_customer_offer(
        event_type="SPORT",
        category="NFL Football",
        state="California",
        venue="SoFi Stadium",
        performer="Los Angeles Rams",
        qty=2,
        days_until_event=14,
        event_month=9
    )

    print("\nRecommended Offer:", offer['recommended_offer'])
    print("Expected Margin:", offer['expected_margin'])
    print("Confidence:", offer['confidence_level'])