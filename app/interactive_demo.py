### app/interactive_demo.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.pricing_engine import CustomerPurchasePriceEngine

if __name__ == "__main__":
    engine = CustomerPurchasePriceEngine()
    engine.fit("data/cleaned_invoice_data.csv")

    print("\n🎟️  Welcome to the Customer Ticket Offer Demo\n")

    event_type = input("Enter event type (e.g., SPORT, CONCERT, THEATER): ").strip()
    category = input("Enter category (e.g., NFL Football, Classical, Comedy): ").strip()
    state = input("Enter U.S. state: ").strip()
    venue = input("Enter venue: ").strip()
    performer = input("Enter performer: ").strip()
    qty = int(input("Enter quantity of tickets: "))
    days_until_event = int(input("Enter number of days until the event: "))
    event_month = int(input("Enter the event month (1-12): "))

    offer = engine.recommend_customer_offer(
        event_type=event_type,
        category=category,
        state=state,
        venue=venue,
        performer=performer,
        qty=qty,
        days_until_event=days_until_event,
        event_month=event_month
    )

    print("\n📢 Offer Recommendation Result:")
    print(f"  Recommended Offer: ${offer['recommended_offer']}")
    print(f"  Expected Margin: {offer['expected_margin']}%")
    print(f"  Confidence Level: {offer['confidence_level'] * 100:.1f}%")
    print(f"  Predicted Resale Price: ${offer['predicted_resale_price']}")
    print(f"  Expected Profit: ${offer['expected_profit']}")
