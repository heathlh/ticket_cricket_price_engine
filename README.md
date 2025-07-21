# Customer Purchase Price Engine

A machine learning-powered pricing engine that recommends optimal ticket purchase offers for secondary marketplaces. The system leverages historical transaction data, seat quality metrics, and market intelligence to predict both resale prices and customer offer prices, helping platforms maximize margins while minimizing risk.

---

## Project Structure

```
ticket_cricket_price_engine/
├── app/
│   ├── run_engine.py              # Command-line script to test sample cases
│   └── streamlit_app.py           # Interactive Streamlit demo app
├── data/
│   ├── cleaned_invoice_data.csv   # Historical ticket transaction data
│   └── options/                   # Dropdown options extracted from data
├── engine/
│   ├── __init__.py
│   ├── config.py                  # Default margin & category assumptions
│   ├── features.py                # Feature generation and market intelligence
│   ├── models.py                  # Model training, evaluation, and persistence
│   └── pricing_engine.py          # Core logic for offer recommendation
├── models/                        # Trained model .pkl files storage
├── scripts/
│   └── extract_dropdown_options.py # Script to extract unique field values
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

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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

> **Note:** If you encounter `ModuleNotFoundError: No module named 'engine'`, make sure to use `PYTHONPATH=.` when launching Streamlit.

---

## How the Engine Works

The system trains two Random Forest models:

- **Resale Model**: Predicts future resale value
- **Cost Model**: Predicts optimal acquisition price (our offer)

The engine then applies:
- **Margin targeting** (default 25%)
- **Market intelligence** (category, venue, performer analysis)
- **Confidence scoring** (based on urgency, price spread, volatility)
- **Risk-adjusted fallback strategy**

### Output:
- Recommended offer price
- Expected resale value
- Profit margin & confidence score
- Business decision: approve/reject

---

## Model Logic & Inputs

The engine is built from the platform's perspective, optimizing for margin rather than customer satisfaction.

### Model Features:
- Event Type, Category, State, Venue, Performer
- Ticket Quantity, Days Until Event, Event Month
- Market statistics (mean & std by category/venue)
- Seat quality tier & premium flags

*All features are extracted and processed via `features.py`*

### Model Targets:
- `unit_price` (expected resale price)
- `unit_cost` (our predicted offer price)

> **No need for customer-reported purchase price** – pricing is learned from market data, not individual reports.

---

## Input Fields (UI or API)

- **Event Type** (e.g., SPORT, CONCERT)
- **Category** (e.g., NFL Football, Comedy)
- **State, Venue, Performer**
- **Quantity**
- **Days Until Event**
- **Event Month**

*All dropdown options are pre-extracted from `cleaned_invoice_data.csv`*

---

## Model Performance (July 2025)

| Metric | Resale Model | Cost Model |
|--------|--------------|------------|
| MAE | 2.82 | 3.86 |
| RMSE | 4.89 | 8.62 |
| R² | 0.997 | 0.990 |
| MAPE | 6.0% | 7.6% |
| Features Used | 67 | 67 |

**Trained on 15,645 records** after outlier removal (80/20 train/test split)

---

## Quick Demo Scenarios

| Scenario | Offer Price | Margin |
|----------|-------------|---------|
| **Maroon 5** (Last-Minute) | $117.21 | 21.0% |
| **NFL Game** (Peak Season) | $147.36 | 12.6% |
| **Comedy Show** (Standard) | $62.82 | 48.8% |

---

## Development Utilities

### Extract Dropdown Options
```bash
python scripts/extract_dropdown_options.py
```

This script reads unique values for `Event Type`, `Category`, `State`, `Venue`, and `Performer` and saves them as CSVs in `data/options/`.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

For questions or contributions, please reach out to **Heath Liao** at [heathlh1018@gmail.com](mailto:heathlh1018@gmail.com)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Built for optimizing secondary ticket marketplace pricing</strong>
</div>
