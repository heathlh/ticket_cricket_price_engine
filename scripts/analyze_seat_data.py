# scripts/analyze_seat_data.py - Fixed version with better path handling

import pandas as pd
import numpy as np
import re
import os
import sys
from collections import Counter

def find_data_file():
    """Find the data file in different possible locations"""
    possible_paths = [
        'data/cleaned_invoice_data.csv',           # From project root
        '../data/cleaned_invoice_data.csv',       # From scripts folder
        '../../data/cleaned_invoice_data.csv',    # From nested scripts folder
        'cleaned_invoice_data.csv'                # Same directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found data file at: {path}")
            return path
    
    print("❌ Could not find cleaned_invoice_data.csv in any of these locations:")
    for path in possible_paths:
        print(f"   - {os.path.abspath(path)}")
    
    return None

def analyze_seat_data():
    """Analyze the actual seat data in your dataset"""
    
    print("🔍 ANALYZING YOUR ACTUAL SEAT DATA")
    print("=" * 60)
    
    # Find the data file
    data_path = find_data_file()
    if not data_path:
        print("\n💡 Please make sure 'cleaned_invoice_data.csv' is in one of the expected locations")
        return
    
    try:
        # Load your data
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} records from {data_path}")
        
        # Show all column names first
        print(f"\n📋 ALL COLUMNS IN YOUR DATA:")
        print("=" * 50)
        for i, col in enumerate(df.columns):
            print(f"   {i+1:2d}. {col}")
        
        # Check what seat-related columns exist
        seat_columns = []
        for col in df.columns:
            if any(word in col.lower() for word in ['section', 'row', 'seat']):
                seat_columns.append(col)
        
        print(f"\n🎪 SEAT-RELATED COLUMNS FOUND:")
        print("=" * 50)
        if seat_columns:
            for col in seat_columns:
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                total_count = len(df)
                fill_rate = ((total_count - null_count) / total_count) * 100
                print(f"   📍 {col}: {unique_count} unique values, {null_count} nulls ({fill_rate:.1f}% filled)")
        else:
            print("   ❌ No columns with 'section', 'row', or 'seat' in the name found!")
            
            # Look for columns that might contain seat data
            print(f"\n🔍 LOOKING FOR POTENTIAL SEAT COLUMNS:")
            potential_columns = []
            for col in df.columns:
                # Check if column contains mixed data that might be seats
                sample_data = df[col].dropna().head(100)
                if len(sample_data) > 0:
                    # Convert to string and check patterns
                    str_data = sample_data.astype(str)
                    # Look for patterns like "A12", "101", "VIP", etc.
                    has_mixed = any(bool(re.search(r'[a-zA-Z].*\d|\d.*[a-zA-Z]', str(val))) for val in str_data)
                    has_short_text = any(len(str(val)) <= 10 and not str(val).replace('.','').isdigit() for val in str_data)
                    
                    if has_mixed or has_short_text:
                        potential_columns.append(col)
            
            for col in potential_columns[:10]:  # Show top 10 potential columns
                sample_values = df[col].dropna().head(5).tolist()
                print(f"      {col}: {sample_values}")
        
        # Analyze each seat column
        for col in seat_columns:
            print(f"\n🔍 ANALYZING {col.upper()}:")
            print("-" * 40)
            
            # Remove nulls for analysis
            data = df[col].dropna()
            
            if len(data) == 0:
                print(f"   No data in {col}")
                continue
            
            # Show data types and basic info
            print(f"   📊 Basic Info:")
            print(f"      Total values: {len(data)}")
            print(f"      Unique values: {data.nunique()}")
            print(f"      Data type: {data.dtype}")
            
            # Show top values
            top_values = data.value_counts().head(20)
            print(f"\n   📈 Top 20 most common values:")
            for value, count in top_values.items():
                percentage = (count / len(data)) * 100
                print(f"      {str(value):<20} {count:>6} ({percentage:4.1f}%)")
            
            # Analyze patterns
            print(f"\n   🔤 Pattern Analysis:")
            analyze_patterns(data, col)
        
        # If we found seat columns, create custom scoring
        if seat_columns:
            create_custom_scoring(df, seat_columns)
        else:
            print(f"\n⚠️  No seat columns found. Please check your data structure.")
            print(f"    You may need to specify which columns contain seat information.")
        
    except Exception as e:
        print(f"❌ Error loading or analyzing data: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_patterns(data, column_name):
    """Analyze patterns in seat data"""
    
    # Convert to string and analyze
    data_str = data.astype(str).str.lower()
    
    # Find numeric patterns
    numeric_values = []
    text_values = []
    mixed_values = []
    
    for value in data_str:
        if str(value).replace('.', '').isdigit():
            try:
                numeric_values.append(float(value))
            except:
                pass
        elif any(char.isdigit() for char in str(value)) and any(char.isalpha() for char in str(value)):
            mixed_values.append(value)
        elif any(char.isalpha() for char in str(value)):
            text_values.append(value)
    
    print(f"      Numeric only: {len(numeric_values)} values")
    if numeric_values:
        print(f"         Range: {min(numeric_values)} - {max(numeric_values)}")
        print(f"         Average: {np.mean(numeric_values):.1f}")
        print(f"         Examples: {sorted(set(numeric_values))[:10]}")
    
    print(f"      Text only: {len(text_values)} values")
    if text_values:
        # Find common words
        words = []
        for value in text_values:
            words.extend(str(value).split())
        if words:
            word_counts = Counter(words).most_common(10)
            print(f"         Common words: {', '.join([f'{word}({count})' for word, count in word_counts])}")
        print(f"         Examples: {list(set(text_values))[:10]}")
    
    print(f"      Mixed text/numbers: {len(mixed_values)} values")
    if mixed_values:
        print(f"         Examples: {list(set(mixed_values))[:10]}")

def create_custom_scoring(df, seat_columns):
    """Create custom scoring based on actual data patterns"""
    
    print(f"\n🎯 CREATING CUSTOM SCORING BASED ON YOUR DATA")
    print("=" * 60)
    
    scoring_suggestions = {}
    
    for col in seat_columns:
        data = df[col].dropna().astype(str).str.lower()
        
        if len(data) == 0:
            continue
        
        print(f"\n📍 CUSTOM SCORING FOR {col.upper()}:")
        
        # Analyze value distribution for scoring
        value_counts = data.value_counts()
        
        # Find premium indicators (usually less common, special names)
        premium_candidates = []
        standard_candidates = []
        poor_candidates = []
        
        for value, count in value_counts.items():
            frequency_pct = (count / len(data)) * 100
            value_str = str(value).lower()
            
            # Look for premium keywords
            if any(word in value_str for word in ['vip', 'premium', 'club', 'suite', 'box', 'luxury', 'floor', 'court', 'field']):
                premium_candidates.append((value, count, frequency_pct))
            # Look for poor indicators
            elif any(word in value_str for word in ['upper', 'nose', 'restricted', 'obstructed', 'partial']):
                poor_candidates.append((value, count, frequency_pct))
            # Look for numeric patterns (lower numbers often better)
            elif value_str.isdigit():
                num_val = int(float(value_str))
                if num_val <= 20:  # Low section numbers often premium
                    premium_candidates.append((value, count, frequency_pct))
                elif num_val >= 300:  # High numbers often poor
                    poor_candidates.append((value, count, frequency_pct))
                else:
                    standard_candidates.append((value, count, frequency_pct))
            # Look for letter patterns (A, B, C often better rows)
            elif len(value_str) == 1 and value_str.isalpha():
                if value_str in ['a', 'b', 'c', 'd', 'e']:
                    premium_candidates.append((value, count, frequency_pct))
                else:
                    standard_candidates.append((value, count, frequency_pct))
            else:
                standard_candidates.append((value, count, frequency_pct))
        
        # Show suggestions
        if premium_candidates:
            print(f"   🏆 SUGGESTED PREMIUM VALUES:")
            for value, count, pct in sorted(premium_candidates, key=lambda x: x[1], reverse=True)[:10]:
                print(f"      '{value}': {count} times ({pct:.1f}%)")
        
        if poor_candidates:
            print(f"   👎 SUGGESTED POOR VALUES:")
            for value, count, pct in sorted(poor_candidates, key=lambda x: x[1], reverse=True)[:10]:
                print(f"      '{value}': {count} times ({pct:.1f}%)")
        
        if standard_candidates:
            print(f"   📊 SUGGESTED STANDARD VALUES (top 10):")
            for value, count, pct in sorted(standard_candidates, key=lambda x: x[1], reverse=True)[:10]:
                print(f"      '{value}': {count} times ({pct:.1f}%)")
        
        scoring_suggestions[col] = {
            'premium': [item[0] for item in premium_candidates[:20]],
            'standard': [item[0] for item in standard_candidates[:50]],
            'poor': [item[0] for item in poor_candidates[:20]]
        }
    
    # Generate Python code for custom scoring
    print(f"\n💻 SUGGESTED PYTHON CODE FOR YOUR DATA:")
    print("=" * 50)
    
    for col, suggestions in scoring_suggestions.items():
        col_name = col.lower().replace(' ', '_')
        print(f"\n# Custom scoring for {col}")
        print(f"{col_name}_premium = {suggestions['premium'][:10]}")
        print(f"{col_name}_standard = {suggestions['standard'][:15]}")
        print(f"{col_name}_poor = {suggestions['poor'][:10]}")

if __name__ == "__main__":
    try:
        analyze_seat_data()
    except Exception as e:
        print(f"❌ Script failed: {str(e)}")
        import traceback
        traceback.print_exc()