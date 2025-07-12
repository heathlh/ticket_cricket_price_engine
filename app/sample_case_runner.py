### app/sample_case_runner.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.pricing_engine import CustomerPurchasePriceEngine


if __name__ == "__main__":
    engine = CustomerPurchasePriceEngine()
    engine.fit("data/cleaned_invoice_data.csv")

    customer_scenarios = [
        {
            'name': '🏈 Customer wants to sell NFL tickets',
            'details': {
                'event_type': 'SPORT',
                'category': 'NFL Football',
                'state': 'California',
                'venue': 'SoFi Stadium',
                'performer': 'Los Angeles Rams',
                'qty': 2,
                'days_until_event': 14
            }
        },
        {
            'name': '🎼 Customer wants to sell Classical tickets',
            'details': {
                'event_type': 'CONCERT',
                'category': 'Classical',
                'state': 'New York',
                'venue': 'Lincoln Center',
                'performer': 'New York Philharmonic',
                'qty': 1,
                'days_until_event': 45
            }
        },
        {
            'name': '🏀 Customer wants to sell NBA tickets (urgent)',
            'details': {
                'event_type': 'SPORT',
                'category': 'NBA Basketball',
                'state': 'California',
                'venue': 'Crypto.com Arena',
                'performer': 'Los Angeles Lakers',
                'qty': 1,
                'days_until_event': 3
            }
        },
        {
            'name': '🎭 Customer wants to sell Comedy show tickets',
            'details': {
                'event_type': 'THEATER',
                'category': 'Comedy',
                'state': 'Nevada',
                'venue': 'Tahoe Blue Event Center',
                'performer': 'Kevin Hart',
                'qty': 2,
                'days_until_event': 21
            }
        }
    ]

    print("\n================ BATCH QUOTE REPORT ================\n")

    for scenario in customer_scenarios:
        offer = engine.recommend_customer_offer(**scenario["details"])
        print(f"{scenario['name']}:")
        print(f"  Recommended Offer: ${offer['recommended_offer']}")
        print(f"  Expected Margin: {offer['expected_margin']}%")
        print(f"  Confidence Level: {offer['confidence_level'] * 100:.1f}%")
        print("----------------------------------------------------")
