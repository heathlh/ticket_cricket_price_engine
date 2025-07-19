# streamlit_app_with_existing_csv.py - Use existing CSV option files

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import warnings

# 🔇 COMPREHENSIVE WARNING SUPPRESSION
warnings.filterwarnings('ignore')
import logging
logging.getLogger('sklearn').setLevel(logging.ERROR)
pd.options.mode.chained_assignment = None

# Add your project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.pricing_engine import CustomerPurchasePriceEngine
from engine.models import load_model

# Page config
st.set_page_config(
    page_title="Enhanced Pricing Engine",
    page_icon="💰",
    layout="wide"
)

@st.cache_data
def load_all_csv_options():
    """Load all options from your existing CSV files"""
    options = {}
    
    csv_files = {
        'categories': 'data/options/Category_options.csv',
        'event_types': 'data/options/Event_Type_options.csv', 
        'performers': 'data/options/Performer_options.csv',
        'states': 'data/options/State_options.csv',
        'venues': 'data/options/Venue_options.csv'
    }
    
    for key, file_path in csv_files.items():
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Use the first column
                column_name = df.columns[0]
                values = df[column_name].dropna().astype(str).tolist()
                # Remove any 'nan' string values and sort
                options[key] = sorted([v for v in values if v.lower() != 'nan'])
                print(f"✅ Loaded {len(options[key])} {key} from {file_path}")
            else:
                st.error(f"❌ File not found: {file_path}")
                options[key] = []
        except Exception as e:
            st.error(f"❌ Error loading {file_path}: {str(e)}")
            options[key] = []
    
    return options

@st.cache_resource
def load_pricing_engine():
    """Load the enhanced pricing engine (cached for performance)"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = CustomerPurchasePriceEngine()
            load_model(engine, "models/enhanced_pricing_engine.pkl")
            return engine, None
    except Exception as e:
        return None, str(e)

def main():
    st.title("🎯 Enhanced Ticket Pricing Engine")
    st.markdown("### Get intelligent pricing recommendations with market insights")
    
    # Load ALL options from your CSV files
    options = load_all_csv_options()
    
    # Load model
    with st.spinner("Loading enhanced pricing engine..."):
        engine, error = load_pricing_engine()
    
    if error:
        st.error(f"❌ Failed to load model: {error}")
        st.info("Make sure you've trained the model by running: `python app/run_engine.py`")
        return
    
    st.success("✅ Enhanced pricing engine loaded successfully!")
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Features:** {len(engine.feature_columns)}")
        st.write(f"**Categories:** {len(engine.category_performance)}")
        st.write(f"**Target Margin:** {engine.target_margin*100:.1f}%")
        
        # Show loaded options info
        st.subheader("📁 Loaded Options")
        for key, values in options.items():
            st.write(f"**{key.title()}:** {len(values)} options")
        
        # Market insights
        if st.button("🔍 Show Market Insights"):
            show_market_insights(engine)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎪 Event Details")
        
        # Event inputs - ALL FROM CSV FILES NOW
        col1a, col1b = st.columns(2)
        
        with col1a:
            # Event Type from CSV
            event_type = st.selectbox(
                "Event Type",
                options=options['event_types'] if options['event_types'] else ["CONCERT"],
                index=0
            )
            
            # Category from CSV
            category = st.selectbox(
                "Category",
                options=options['categories'] if options['categories'] else ["Pop"],
                index=0
            )
            
            # State from CSV
            state = st.selectbox(
                "State",
                options=options['states'] if options['states'] else ["California"],
                index=0
            )
            
        with col1b:
            # Venue from CSV
            venue = st.selectbox(
                "Venue",
                options=options['venues'] if options['venues'] else ["Madison Square Garden"],
                index=0,
                help="Select from available venues"
            )
            
            # Performer from CSV
            performer = st.selectbox(
                "Performer", 
                options=options['performers'] if options['performers'] else ["Taylor Swift"],
                index=0,
                help="Select from available performers"
            )
            
            # Quantity - keep as number input
            qty = st.number_input("Quantity", min_value=1, max_value=20, value=2)
        
        st.header("⏰ Timing Details")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            # Days until event - keep as slider
            days_until_event = st.slider(
                "Days Until Event", 
                min_value=0, 
                max_value=120, 
                value=14,
                help="0 = same day, 120 = 4 months ahead"
            )
            
        with col2b:
            # Event month - keep as dropdown
            event_month = st.selectbox(
                "Event Month",
                list(range(1, 13)),
                index=5,  # June
                format_func=lambda x: [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                ][x-1]
            )
    
    with col2:
        st.header("🎯 Get Recommendation")
        
        if st.button("💰 Calculate Pricing", type="primary", use_container_width=True):
            with st.spinner("Analyzing market data..."):
                recommendation = get_pricing_recommendation(
                    engine, event_type, category, state, venue, 
                    performer, qty, days_until_event, event_month
                )
                
                if recommendation:
                    display_recommendation(recommendation, qty)
                else:
                    st.error("Failed to get recommendation")
        
        # Quick presets with your data
        st.subheader("🚀 Quick Presets")
        
        if st.button("🏈 NFL Game", use_container_width=True):
            show_preset_suggestions("nfl", options)
        
        if st.button("🎵 Pop Concert", use_container_width=True):
            show_preset_suggestions("pop", options)
            
        if st.button("😂 Comedy Show", use_container_width=True):
            show_preset_suggestions("comedy", options)
        
        # Search functionality
        st.subheader("🔍 Quick Search")
        search_term = st.text_input("Search performers/venues", placeholder="Type to search...")
        
        if search_term:
            show_search_results(search_term, options)

def show_preset_suggestions(preset_type, options):
    """Show preset suggestions based on your actual data"""
    
    # Search for matching items in your actual data
    if preset_type == 'nfl':
        # Look for NFL-related performers and venues
        nfl_performers = [p for p in options['performers'] if any(word in p.lower() for word in ['giants', 'cowboys', 'patriots', 'rams', 'chiefs', 'steelers', 'packers'])]
        nfl_venues = [v for v in options['venues'] if any(word in v.lower() for word in ['stadium', 'field'])]
        
        if nfl_performers or nfl_venues:
            st.success("✅ NFL suggestions from your data:")
            if nfl_performers:
                st.write(f"**Performers:** {', '.join(nfl_performers[:3])}")
            if nfl_venues:
                st.write(f"**Venues:** {', '.join(nfl_venues[:3])}")
    
    elif preset_type == 'pop':
        # Look for pop artists and music venues
        pop_performers = [p for p in options['performers'] if any(artist in p.lower() for artist in ['taylor', 'maroon', 'ariana', 'ed sheeran', 'swift'])]
        music_venues = [v for v in options['venues'] if any(word in v.lower() for word in ['garden', 'center', 'hall', 'bowl', 'arena'])]
        
        if pop_performers or music_venues:
            st.success("✅ Pop concert suggestions from your data:")
            if pop_performers:
                st.write(f"**Performers:** {', '.join(pop_performers[:3])}")
            if music_venues:
                st.write(f"**Venues:** {', '.join(music_venues[:3])}")
    
    elif preset_type == 'comedy':
        # Look for comedians and comedy venues
        comedy_performers = [p for p in options['performers'] if any(comedian in p.lower() for comedian in ['chappelle', 'hart', 'seinfeld', 'schumer'])]
        comedy_venues = [v for v in options['venues'] if any(word in v.lower() for word in ['theater', 'theatre', 'club', 'hall'])]
        
        if comedy_performers or comedy_venues:
            st.success("✅ Comedy suggestions from your data:")
            if comedy_performers:
                st.write(f"**Performers:** {', '.join(comedy_performers[:3])}")
            if comedy_venues:
                st.write(f"**Venues:** {', '.join(comedy_venues[:3])}")

def show_search_results(search_term, options):
    """Show search results from your actual data"""
    search_lower = search_term.lower()
    
    # Search performers
    matching_performers = [p for p in options['performers'] if search_lower in p.lower()]
    
    # Search venues  
    matching_venues = [v for v in options['venues'] if search_lower in v.lower()]
    
    # Search categories
    matching_categories = [c for c in options['categories'] if search_lower in c.lower()]
    
    if matching_performers:
        st.write("**Matching Performers:**")
        for performer in matching_performers[:5]:  # Show top 5
            st.write(f"• {performer}")
    
    if matching_venues:
        st.write("**Matching Venues:**")
        for venue in matching_venues[:5]:  # Show top 5
            st.write(f"• {venue}")
    
    if matching_categories:
        st.write("**Matching Categories:**")
        for category in matching_categories[:5]:  # Show top 5
            st.write(f"• {category}")
    
    if not any([matching_performers, matching_venues, matching_categories]):
        st.write("No matches found in your data")

def get_pricing_recommendation(engine, event_type, category, state, venue, 
                             performer, qty, days_until_event, event_month):
    """Get pricing recommendation from the engine"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            recommendation = engine.recommend_customer_offer(
                event_type=event_type,
                category=category,
                state=state,
                venue=venue,
                performer=performer,
                qty=qty,
                days_until_event=days_until_event,
                event_month=event_month
            )
            return recommendation
    except Exception as e:
        st.error(f"Error getting recommendation: {str(e)}")
        return None

def display_recommendation(rec, qty):
    """Display the pricing recommendation"""
    
    # Main metrics
    st.subheader("💰 Pricing Recommendation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Recommended Offer",
            f"${rec['recommended_offer']:,.2f}",
            help="Per ticket price"
        )
    
    with col2:
        st.metric(
            "Expected Margin",
            f"{rec['expected_margin']:.1f}%",
            delta=f"{rec['expected_margin'] - rec['target_margin']:.1f}% vs target"
        )
    
    with col3:
        st.metric(
            "Confidence Level",
            f"{rec['confidence_level']:.2f}",
            help="Model confidence (0-1)"
        )
    
    # Total calculation
    total_offer = rec['recommended_offer'] * qty
    total_profit = rec['expected_profit'] * qty
    
    st.subheader(f"📊 Total for {qty} ticket{'s' if qty > 1 else ''}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Offer", f"${total_offer:,.2f}")
    
    with col2:
        st.metric("Total Profit", f"${total_profit:,.2f}")
    
    with col3:
        st.metric("Predicted Cost", f"${rec['predicted_cost']:,.2f}")
    
    with col4:
        st.metric("Market Price", f"${rec['predicted_resale_price']:,.2f}")
    
    # Pricing strategy breakdown
    st.subheader("🏗️ Pricing Strategy")
    
    pricing = rec['pricing_components']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Conservative Component**")
        st.write(f"${pricing['conservative_offer']:,.2f} ({pricing['conservative_weight']:.0%})")
        st.progress(pricing['conservative_weight'])
    
    with col2:
        st.write("**Competitive Component**")
        st.write(f"${pricing['competitive_offer']:,.2f} ({pricing['competitive_weight']:.0%})")
        st.progress(pricing['competitive_weight'])
    
    # Business decision
    decision = make_business_decision(rec)
    
    if "APPROVE" in decision['action']:
        st.success(f"**{decision['action']}** - {decision['reason']}")
    elif "REVIEW" in decision['action']:
        st.warning(f"**{decision['action']}** - {decision['reason']}")
    else:
        st.error(f"**{decision['action']}** - {decision['reason']}")
    
    # Risk assessment
    st.subheader("⚖️ Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Risk Adjustment", f"{rec['risk_adjustment']:.3f}")
        
    with col2:
        margin_diff = rec['expected_margin'] - rec['target_margin']
        st.metric("Target vs Expected", f"{margin_diff:+.1f}%")

def make_business_decision(rec):
    """Business decision logic"""
    margin = rec['expected_margin']
    confidence = rec['confidence_level']
    offer = rec['recommended_offer']
    
    if margin < 5:
        return {'action': '❌ DECLINE', 'reason': 'Margin too low (<5%)'}
    elif margin > 60:
        return {'action': '⚠️ REVIEW', 'reason': 'Margin unusually high (>60%)'}
    elif confidence < 0.6:
        return {'action': '⚠️ REVIEW', 'reason': 'Low confidence prediction'}
    elif offer > 500:
        return {'action': '⚠️ REVIEW', 'reason': 'High value transaction'}
    else:
        return {'action': '✅ APPROVE', 'reason': f'Good opportunity - {margin:.1f}% margin'}

def show_market_insights(engine):
    """Show market insights in sidebar"""
    st.sidebar.subheader("📈 Market Overview")
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            insights = engine.get_market_insights()
        
        for key, value in insights.items():
            if isinstance(value, list):
                if len(value) > 0:
                    st.sidebar.write(f"**{key}:** {', '.join(value[:3])}...")
                else:
                    st.sidebar.write(f"**{key}:** None")
            else:
                st.sidebar.write(f"**{key}:** {value}")
    except Exception as e:
        st.sidebar.error(f"Error loading insights: {str(e)}")

if __name__ == "__main__":
    main()