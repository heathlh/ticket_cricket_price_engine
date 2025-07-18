import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from engine.pricing_engine import CustomerPurchasePriceEngine

# Load dropdown options from CSV files
def load_options(file_path):
    return pd.read_csv(file_path).iloc[:, 0].dropna().unique().tolist()

# Set paths to option files
option_dir = "data/options"
event_types = load_options(f"{option_dir}/Event_Type_options.csv")
categories = load_options(f"{option_dir}/Category_options.csv")
states = load_options(f"{option_dir}/State_options.csv")
venues = load_options(f"{option_dir}/Venue_options.csv")
performers = load_options(f"{option_dir}/Performer_options.csv")

st.title("🎟️ Customer Offer Recommendation Engine")

st.markdown("""
Use this interface to enter customer ticket details and get an offer recommendation.
""")

with st.form("input_form"):
    st.subheader("📋 Customer Ticket Info")
    event_type = st.selectbox("Event Type", event_types)
    category = st.selectbox("Category", categories)
    state = st.selectbox("State", states)
    venue = st.selectbox("Venue", venues)
    performer = st.selectbox("Performer", performers)
    qty = st.number_input("Quantity", min_value=1, max_value=10, value=1)
    days_until_event = st.slider("Days Until Event", 0, 365, 30)
    event_month = st.slider("Event Month (1-12)", 1, 12, 6)
    submitted = st.form_submit_button("Get Recommendation")

if submitted:
    st.info("Processing your request...")
    engine = CustomerPurchasePriceEngine()
    engine.fit("data/cleaned_invoice_data.csv")
    
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

    st.success("🎯 Offer Recommendation")
    st.write(f"**Recommended Offer (per ticket):** ${offer['recommended_offer']:.2f}")
    st.write(f"**Expected Margin:** {offer['expected_margin']}%")
    st.write(f"**Confidence Level:** {offer['confidence_level'] * 100:.1f}%")

    st.metric("Predicted Resale Price (per ticket)", f"${offer['predicted_resale_price']:.2f}")
    st.metric("Expected Profit (per ticket)", f"${offer['expected_profit']:.2f}")

