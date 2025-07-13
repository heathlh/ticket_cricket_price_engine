# Customer Purchase Price Engine

This project provides a machine learning–powered pricing engine to recommend optimal ticket purchase offers for customers looking to sell their tickets on a secondary marketplace.

---


## Project Structure

```
ticket_cricket_price_engine/
├── app/
│   ├── run_engine.py              # Command-line script to test a sample case
│   └── streamlit_app.py          # Interactive Streamlit demo app
├── data/
│   ├── cleaned_invoice_data.csv  # Historical ticket transaction data
│   └── options/                  # Dropdown options extracted from data
├── engine/
│   ├── __init__.py
│   ├── config.py                 # Default margin & category assumptions
│   ├── features.py              # Feature generation and market intelligence
│   ├── models.py                # Model training, evaluation, and persistence
│   └── pricing_engine.py        # Core logic for offer recommendation
├── models/                      # Folder to store trained model .pkl files
├── scripts/
│   └── extract_dropdown_options.py  # Script to extract unique field values
├── requirements.txt
└── README.md
```

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

## How It Works

1. Trains two Random Forest models:
   - Predicting ticket resale value
   - Predicting optimal cost (offer) price

2. Adds business logic:
   - Profit margin target (default 25%)
   - Market trends per category/venue
   - Confidence scoring based on urgency & demand

3. Returns:
   - Recommended offer
   - Expected profit & margin
   - Confidence level

---

## Model Logic Explained

The model is designed from the platform's perspective. It takes as input various customer and market characteristics and learns from historical data how much to offer customers and what resale price to expect. 

### Model Inputs:
- Event Type, Category, State, Venue, Performer
- Ticket Quantity, Days Until Event, Event Month
- Historical market-level statistics (mean/std of resale/cost by category/venue)

These features are extracted from historical ticket sales and enhanced with engineered features in `features.py`.

### Prediction Targets:
- `unit_price`: What price the platform can likely resell the ticket for
- `unit_cost`: What price the platform historically paid customers (i.e., our offer)

`unit_cost` is the **main prediction output** used to generate customer-facing price offers. 

The model **does not require the customer to input their original purchase price**, because:
- The goal is to optimize platform margin, not match individual expectations.
- Including customer-provided prices may introduce noise or bias.
- We learn offer behavior purely from market and transactional data.

---

## Input Fields (via UI)
- Event Type (e.g., SPORT, CONCERT)
- Category (e.g., NFL Football, Classical)
- State, Venue, Performer
- Quantity
- Days Until Event, Event Month

All values are sourced from historical data (`cleaned_invoice_data.csv`).

---

## Development Utilities

### Extract Dropdown Options
```bash
python scripts/extract_dropdown_options.py
```
This script reads unique values for `Event Type`, `Category`, `State`, `Venue`, and `Performer` and saves them as CSVs in `data/options/`.

---

## Contact
For questions or contributions, please reach out to Heath Liao at heathlh1018@gmail.com.
