### app/run_engine.py
import sys
import os
import joblib 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.pricing_engine import CustomerPurchasePriceEngine

if __name__ == "__main__":
    engine = CustomerPurchasePriceEngine()
    engine.fit("data/cleaned_invoice_data.csv")

    # ✅ Save model directly
    joblib.dump(engine.rf_model, "models/purchase_price_model.pkl")
    print("✅ Model saved to models/purchase_price_model.pkl")

    offer = engine.recommend_customer_offer(
        event_type="CONCERT",
        category="Pop",
        state="Indiana",
        venue="Ruoff Music Center",
        performer="Maroon 5",
        qty=2,
        days_until_event=0,
        event_month=8
    )

    print("Recommended Offer (per ticket): ${:.2f}".format(offer['recommended_offer']))

    # expected_margin = (predicted_resale_price - recommended_offer_price) / recommended_offer_price
    print("Expected Margin:", offer['expected_margin'])
    print("Confidence:", offer['confidence_level'])
    print("Predicted Resale Price (per ticket): ${:.2f}".format(offer['predicted_resale_price']))
    print("Expected Profit (per ticket): ${:.2f}".format(offer['expected_profit']))