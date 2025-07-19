import pandas as pd
import os

input_path = "data/cleaned_invoice_data.csv"
output_dir = "data/options"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

columns_to_extract = ["Event Type", "Category", "State", "Venue", "Performer"]

for col in columns_to_extract:
    unique_vals = sorted(df[col].dropna().unique())
    output_path = os.path.join(output_dir, f"{col.replace(' ', '_')}_options.csv")
    pd.Series(unique_vals).to_csv(output_path, index=False, header=[col])
    print(f"✅ Saved: {output_path}")
