import pandas as pd
import pprint

# Load the cleaned invoice data
df = pd.read_csv("cleaned_invoice_data.csv")

# Create derived columns if they don't exist
if 'Profit_Margin' not in df.columns:
    df['Profit_Margin'] = (df['Unit Ticket Sales'] - df['Unit Cost']) / df['Unit Ticket Sales']

if 'Customer_Payment_Ratio' not in df.columns:
    df['Customer_Payment_Ratio'] = df['Unit Cost'] / df['Unit Ticket Sales']

# Clean invalid values
df = df.replace([float('inf'), -float('inf')], pd.NA).dropna(subset=['Profit_Margin', 'Customer_Payment_Ratio'])

# Aggregate stats by category
category_defaults = df.groupby('Category').agg({
    'Unit Ticket Sales': 'mean',
    'Profit_Margin': 'mean',
    'Customer_Payment_Ratio': 'mean'
}).round(2)

# Rename columns to match your config style
category_defaults.rename(columns={
    'Unit Ticket Sales': 'resale_mean',
    'Profit_Margin': 'margin_mean',
    'Customer_Payment_Ratio': 'payment_ratio'
}, inplace=True)

# Convert to dictionary format
category_dict = category_defaults.to_dict(orient='index')

# Pretty print for config.py usage
pprint.pprint(category_dict)
