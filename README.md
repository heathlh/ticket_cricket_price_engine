#  Customer Purchase Price Engine

A machine learningpowered pricing engine to recommend optimal ticket purchase offers for a secondary marketplace. The system uses historical transaction data, seat quality, and market intelligence to predict both resale price and customer offer price, helping platforms maximize margin while minimizing risk.

---

##  Project Structure


ticket_cricket_price_engine/
 app/
    run_engine.py              # CLI for model training  scenario testing
    streamlit_app.py           # Interactive Streamlit demo for user input
 data/
    cleaned_invoice_data.csv   # Historical ticket transaction data
    options/                   # Dropdown option CSVs (venue, category, etc.)
 engine/
    config.py                  # Default business configs (margins, thresholds)
    enhanced_seat_scorer.py    # in Testing progress
    features.py                # Feature engineering  market intelligence
    models.py                  # Model training, tuning  evaluation
    pricing_engine.py          # Core recommendation engine
 models/                        # Trained model artifacts (.pkl)
 scripts/
    extract_dropdown_options.py # Extracts dropdown choices for UI
 requirements.txt
 README.md


---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/heathlh/ticket_cricket_price_engine.git
cd ticket_cricket_price_engine
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Engine on a Sample Case
```bash
python app/run_engine.py
```

### 5. Start the Streamlit App
```bash
PYTHONPATH=. streamlit run app/streamlit_app.py
```

> If you encounter `ModuleNotFoundError: No module named 'engine'`, make sure to use `PYTHONPATH=.` when launching Streamlit.

---
##  How the Engine Works

The system trains two Random Forest models:
- Resale Model: Predicts future resale value
- Cost Model: Predicts optimal acquisition price (our offer)

It then applies:
-  Margin targeting (default 25)
-  Market intelligence (category, venue, performer)
-  Confidence scoring (based on urgency, price spread, volatility)
-  Risk-adjusted fallback strategy

###  Output:
- Recommended offer price
- Expected resale value
- Profit margin  confidence
- Business decision: approve/reject

---

##  Model Logic  Inputs

The engine is built from the platforms perspective, optimizing for margin rather than customer satisfaction.

### Model Features:
- Event Type, Category, State, Venue, Performer
- Ticket Quantity, Days Until Event, Event Month
- Market stats (mean  std by category/venue)
- Seat quality tier  premium flags

All features are extracted and processed via features.py.

### Model Targets:
- unit_price (expected resale price)
- unit_cost (our predicted offer price)

 No need for customer-reported purchase price  pricing is learned from the market, not individuals.

---

##  Input Fields (UI or API)
- Event Type (e.g., SPORT, CONCERT)
- Category (e.g., NFL Football, Comedy)
- State, Venue, Performer
- Quantity
- Days Until Event
- Event Month

All dropdown options are pre-extracted from cleaned_invoice_data.csv.

---

##  Model Performance (as of July 2025)

 Metric                  Resale Model  Cost Model 
--------------------------------------------------
 MAE                    2.82         3.86      
 RMSE                   4.89         8.62      
 R²                     0.997         0.990      
 MAPE                   6.0          7.6       
 Features Used          67            67         

 Trained on 15,645 records after outlier removal. Split 80/20.

---

##  Quick Demo Scenarios

-  Maroon 5 (Last-Minute): Offer  117.21  Margin  21.0   
-  NFL Game (Peak): Offer  147.36  Margin  12.6   
-  Comedy Show (Standard): Offer  62.82  Margin  48.8 

---

##  Development Utilities

### Extract Dropdown Options
bash
python scripts/extract_dropdown_options.py

 Generates CSVs of unique values (venue, category, etc.) for dropdown inputs.

---

##  Author

Heath Liao  
Data Science Intern
heathlh1018gmail.com  

---

##  License

MIT License. Feel free to fork, adapt, and contribute
