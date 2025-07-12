### engine/config.py

TARGET_MARGIN = 0.25

CATEGORY_DEFAULTS = {
    'NFL Football': {'resale_mean': 180, 'margin_mean': 0.25, 'payment_ratio': 0.80},
    'NBA Basketball': {'resale_mean': 125, 'margin_mean': 0.20, 'payment_ratio': 0.83},
    'Classical': {'resale_mean': 100, 'margin_mean': 0.40, 'payment_ratio': 0.71},
    'Comedy': {'resale_mean': 95, 'margin_mean': 0.30, 'payment_ratio': 0.77},
    'Country and Folk': {'resale_mean': 72, 'margin_mean': 0.29, 'payment_ratio': 0.77},
    'Alternative': {'resale_mean': 51, 'margin_mean': 0.52, 'payment_ratio': 0.66},
    'World Music': {'resale_mean': 46, 'margin_mean': 0.66, 'payment_ratio': 0.66},
}