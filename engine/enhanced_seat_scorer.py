# enhanced_seat_scorer.py - Upgraded version with pricing integration

import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, Optional

class EnhancedSeatScorer:
    """
    Enhanced seat scoring system with pricing impact integration
    """
    
    def __init__(self):
        # Premium section keywords (highest scores)
        self.section_premium = [
            'floor general admission', 'club', 'floor', 'loge', 'preferred', 
            'vip', 'suite', 'box', 'courtside', 'field level', 'diamond',
            'premium', 'luxury', 'directors', 'champions', 'platinum'
        ]
        
        # Good section keywords
        self.section_good = [
            'general admission', 'lower level', 'mezzanine', 'terrace',
            'pavilion', 'reserved', 'gold', 'silver'
        ]
        
        # Poor section indicators
        self.section_poor = [
            'upper', 'nosebleed', 'lawn', 'standing room', 'obstructed',
            'partial view', 'limited view'
        ]
        
        # Venue-specific premium sections (can be expanded)
        self.venue_premium_sections = {
            'madison square garden': ['1', '2', '3', '101', '102', '103'],
            'staples center': ['100', '101', '102', '111', '112'],
            'wembley stadium': ['101', '102', '103', '104', '105']
        }
        
        # Row scoring patterns
        self.row_premium = ['1', '2', '3', '4', '5', 'a', 'b', 'c', 'd', 'e', 'aa', 'bb']
        self.row_good = ['6', '7', '8', '9', '10', 'f', 'g', 'h', 'i', 'j']
        self.row_standard = ['ga', 'general', 'admission', 'lawn']
        
        # Seat position preferences (center is usually better)
        self.center_seat_ranges = [(8, 22), (10, 20), (12, 18)]  # Common center ranges

    def calculate_comprehensive_seat_score(self, section: str, row: str, seats: str, 
                                         venue: str = None, category: str = None) -> Dict:
        """
        Calculate comprehensive seat score with detailed breakdown
        """
        try:
            # Base score components
            section_score = self._score_section_enhanced(section, venue)
            row_score = self._score_row_enhanced(row)
            seat_position_score = self._score_seat_position_enhanced(seats)
            venue_bonus = self._get_venue_bonus(venue)
            category_adjustment = self._get_category_adjustment(category)
            
            # Weighted final score (0-100)
            final_score = (
                section_score * 0.45 +      # Section is most important
                row_score * 0.30 +          # Row is very important
                seat_position_score * 0.15 + # Seat position matters
                venue_bonus * 0.05 +        # Venue prestige bonus
                category_adjustment * 0.05   # Category-specific adjustment
            )
            
            # Ensure score is between 0-100
            final_score = max(0, min(100, final_score))
            
            # Calculate pricing impact multiplier
            pricing_multiplier = self._calculate_pricing_multiplier(final_score)
            
            return {
                'seat_quality_score': round(final_score, 2),
                'seat_tier': self._get_seat_tier(final_score),
                'pricing_multiplier': round(pricing_multiplier, 3),
                'score_breakdown': {
                    'section_score': round(section_score, 2),
                    'row_score': round(row_score, 2),
                    'seat_position_score': round(seat_position_score, 2),
                    'venue_bonus': round(venue_bonus, 2),
                    'category_adjustment': round(category_adjustment, 2)
                },
                'seat_features': {
                    'is_premium': final_score >= 80,
                    'is_good': 60 <= final_score < 80,
                    'is_average': 40 <= final_score < 60,
                    'is_below_average': 25 <= final_score < 40,
                    'is_poor': final_score < 25
                }
            }
            
        except Exception as e:
            print(f"Error calculating seat score: {e}")
            return self._get_default_seat_score()

    def _score_section_enhanced(self, section: str, venue: str = None) -> float:
        """Enhanced section scoring with venue-specific knowledge"""
        if pd.isna(section):
            return 50  # Neutral score for missing data
            
        section_str = str(section).lower().strip()
        base_score = 50
        
        # Check for premium keywords
        for keyword in self.section_premium:
            if keyword.lower() in section_str:
                base_score += 35
                break
        
        # Check for good keywords
        for keyword in self.section_good:
            if keyword.lower() in section_str:
                base_score += 20
                break
        
        # Check for poor keywords
        for keyword in self.section_poor:
            if keyword.lower() in section_str:
                base_score -= 25
                break
        
        # Venue-specific section scoring
        if venue:
            venue_lower = venue.lower()
            for venue_name, premium_sections in self.venue_premium_sections.items():
                if venue_name in venue_lower:
                    if any(premium in section_str for premium in premium_sections):
                        base_score += 15
                        break
        
        # Numeric section analysis
        numbers = re.findall(r'\d+', section_str)
        if numbers:
            section_num = int(numbers[0])
            
            if section_num <= 20:
                base_score += 25      # Premium low numbers
            elif section_num <= 50:
                base_score += 15      # Good low numbers
            elif section_num <= 100:
                base_score += 5       # Decent mid numbers
            elif section_num <= 200:
                base_score -= 5       # Higher sections
            elif section_num <= 300:
                base_score -= 15      # Upper level
            else:
                base_score -= 30      # Very high/poor sections
        
        return max(0, min(100, base_score))

    def _score_row_enhanced(self, row: str) -> float:
        """Enhanced row scoring with better pattern recognition"""
        if pd.isna(row):
            return 50
            
        row_str = str(row).lower().strip()
        base_score = 50
        
        # GA (General Admission) - neutral
        if row_str in self.row_standard:
            return 50
        
        # Premium front rows
        if row_str in self.row_premium:
            base_score += 30
        elif row_str in self.row_good:
            base_score += 15
        
        # Numeric row analysis
        if row_str.isdigit():
            row_num = int(row_str)
            
            if row_num <= 3:
                base_score += 35      # Front row premium
            elif row_num <= 10:
                base_score += 20      # Very good rows
            elif row_num <= 20:
                base_score += 10      # Good rows
            elif row_num <= 35:
                base_score += 0       # Average rows
            elif row_num <= 50:
                base_score -= 10      # Back rows
            else:
                base_score -= 25      # Very back rows
        
        # Letter rows (A better than Z)
        if len(row_str) == 1 and row_str.isalpha():
            letter_value = ord(row_str.upper()) - ord('A')
            if letter_value <= 5:      # A-F
                base_score += 25
            elif letter_value <= 10:  # G-K
                base_score += 15
            elif letter_value <= 15:  # L-P
                base_score += 5
            else:                      # Q-Z
                base_score -= 10
        
        return max(0, min(100, base_score))

    def _score_seat_position_enhanced(self, seats: str) -> float:
        """Enhanced seat position scoring"""
        if pd.isna(seats):
            return 50
            
        seats_str = str(seats).strip()
        base_score = 50
        
        # Extract numbers from seat ranges
        numbers = re.findall(r'\d+', seats_str)
        
        if not numbers:
            return base_score
        
        # Calculate average seat number for ranges
        if len(numbers) >= 2:
            start_seat = int(numbers[0])
            end_seat = int(numbers[-1])
            avg_seat = (start_seat + end_seat) / 2
            seat_range_width = end_seat - start_seat + 1
        else:
            avg_seat = int(numbers[0])
            seat_range_width = 1
        
        # Center seats are typically better (assuming 20-seat rows)
        center_distance = abs(avg_seat - 10)  # Distance from seat 10 (assumed center)
        
        if center_distance <= 3:      # Seats 7-13 (center)
            base_score += 15
        elif center_distance <= 6:    # Seats 4-16 (good)
            base_score += 8
        elif center_distance <= 10:   # Seats 1-20 (decent)
            base_score += 3
        elif avg_seat > 30:           # Far aisle seats
            base_score -= 8
        
        # Bonus for single seats vs. large groups (sometimes better selection)
        if seat_range_width == 1:
            base_score += 2
        elif seat_range_width <= 2:
            base_score += 5  # Pairs are often premium
        elif seat_range_width >= 6:
            base_score -= 3  # Large groups might be less desirable
        
        return max(0, min(100, base_score))

    def _get_venue_bonus(self, venue: str) -> float:
        """Get venue prestige bonus"""
        if not venue:
            return 0
            
        venue_lower = venue.lower()
        
        # Iconic venues
        if any(iconic in venue_lower for iconic in [
            'madison square garden', 'msg', 'wembley', 'fenway', 
            'yankee stadium', 'staples center', 'oracle arena'
        ]):
            return 10
        
        # Major venues
        if any(major in venue_lower for major in [
            'center', 'arena', 'stadium', 'garden', 'dome', 'coliseum'
        ]):
            return 5
        
        return 0

    def _get_category_adjustment(self, category: str) -> float:
        """Get category-specific seat importance adjustment"""
        if not category:
            return 0
            
        category_lower = category.lower()
        
        # Sports events - seating very important
        if any(sport in category_lower for sport in [
            'nfl', 'nba', 'mlb', 'nhl', 'football', 'basketball', 'baseball', 'hockey'
        ]):
            return 8
        
        # Theater/Opera - seating critical
        if any(theater in category_lower for theater in [
            'opera', 'theater', 'theatre', 'broadway', 'classical'
        ]):
            return 10
        
        # Concerts - seating moderately important
        if any(music in category_lower for music in [
            'pop', 'rock', 'country', 'rap', 'hip hop', 'alternative'
        ]):
            return 5
        
        return 0

    def _calculate_pricing_multiplier(self, seat_score: float) -> float:
        """Calculate pricing impact multiplier based on seat score"""
        # Convert seat score to pricing multiplier
        if seat_score >= 90:
            return 1.25      # Premium seats: +25%
        elif seat_score >= 80:
            return 1.15      # Excellent seats: +15%
        elif seat_score >= 70:
            return 1.08      # Very good seats: +8%
        elif seat_score >= 60:
            return 1.03      # Good seats: +3%
        elif seat_score >= 40:
            return 1.00      # Average seats: baseline
        elif seat_score >= 25:
            return 0.95      # Below average: -5%
        else:
            return 0.88      # Poor seats: -12%

    def _get_seat_tier(self, score: float) -> str:
        """Get descriptive seat tier"""
        if score >= 85:
            return "Premium"
        elif score >= 70:
            return "Excellent"
        elif score >= 55:
            return "Good"
        elif score >= 40:
            return "Average"
        elif score >= 25:
            return "Below Average"
        else:
            return "Poor"

    def _get_default_seat_score(self) -> Dict:
        """Return default seat score when calculation fails"""
        return {
            'seat_quality_score': 50.0,
            'seat_tier': "Average",
            'pricing_multiplier': 1.00,
            'score_breakdown': {
                'section_score': 50.0,
                'row_score': 50.0,
                'seat_position_score': 50.0,
                'venue_bonus': 0.0,
                'category_adjustment': 0.0
            },
            'seat_features': {
                'is_premium': False,
                'is_good': False,
                'is_average': True,
                'is_below_average': False,
                'is_poor': False
            }
        }

    def bulk_score_seats_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive seat scores to entire dataframe"""
        print("🎪 Calculating enhanced seat scores with pricing impact...")
        
        # Ensure required columns exist
        required_cols = ['Section', 'Row', 'Seats']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Optional columns
        venue_col = 'Venue' if 'Venue' in df.columns else None
        category_col = 'Category' if 'Category' in df.columns else None
        
        # Calculate comprehensive seat scores
        seat_results = []
        for idx, row in df.iterrows():
            venue = row.get(venue_col) if venue_col else None
            category = row.get(category_col) if category_col else None
            
            result = self.calculate_comprehensive_seat_score(
                section=row.get('Section'),
                row=row.get('Row'),
                seats=row.get('Seats'),
                venue=venue,
                category=category
            )
            seat_results.append(result)
        
        # Add main seat features
        df['Seat_Quality_Score'] = [r['seat_quality_score'] for r in seat_results]
        df['Seat_Quality_Tier'] = [r['seat_tier'] for r in seat_results]
        df['Seat_Pricing_Multiplier'] = [r['pricing_multiplier'] for r in seat_results]
        
        # Add boolean features
        df['Is_Premium_Seat'] = [r['seat_features']['is_premium'] for r in seat_results]
        df['Is_Good_Seat'] = [r['seat_features']['is_good'] for r in seat_results]
        df['Is_Average_Seat'] = [r['seat_features']['is_average'] for r in seat_results]
        df['Is_Poor_Seat'] = [r['seat_features']['is_poor'] for r in seat_results]
        
        # Add score breakdown for analysis
        df['Section_Score'] = [r['score_breakdown']['section_score'] for r in seat_results]
        df['Row_Score'] = [r['score_breakdown']['row_score'] for r in seat_results]
        df['Seat_Position_Score'] = [r['score_breakdown']['seat_position_score'] for r in seat_results]
        
        # Calculate seat-adjusted pricing features
        if 'Unit Ticket Sales' in df.columns:
            df['Seat_Adjusted_Resale'] = df['Unit Ticket Sales'] * df['Seat_Pricing_Multiplier']
        
        if 'Unit Cost' in df.columns:
            df['Seat_Adjusted_Cost'] = df['Unit Cost'] * df['Seat_Pricing_Multiplier']
        
        # Premium section indicators
        df['Is_Floor_Section'] = df['Section'].astype(str).str.lower().str.contains('floor', na=False).astype(int)
        df['Is_Club_Section'] = df['Section'].astype(str).str.lower().str.contains('club', na=False).astype(int)
        df['Is_Suite_Section'] = df['Section'].astype(str).str.lower().str.contains('suite|box|vip', na=False).astype(int)
        df['Is_GA_Section'] = df['Section'].astype(str).str.lower().str.contains('general admission', na=False).astype(int)
        
        # Summary statistics
        print(f"✅ Enhanced seat scoring completed!")
        print(f"   Average seat score: {df['Seat_Quality_Score'].mean():.1f}")
        print(f"   Premium seats: {df['Is_Premium_Seat'].sum()} ({df['Is_Premium_Seat'].mean()*100:.1f}%)")
        print(f"   Good seats: {df['Is_Good_Seat'].sum()} ({df['Is_Good_Seat'].mean()*100:.1f}%)")
        print(f"   Average seats: {df['Is_Average_Seat'].sum()} ({df['Is_Average_Seat'].mean()*100:.1f}%)")
        print(f"   Poor seats: {df['Is_Poor_Seat'].sum()} ({df['Is_Poor_Seat'].mean()*100:.1f}%)")
        print(f"   Average pricing multiplier: {df['Seat_Pricing_Multiplier'].mean():.3f}")
        
        return df

    def get_seat_insights(self, df: pd.DataFrame) -> Dict:
        """Get insights about seat quality distribution and pricing impact"""
        if 'Seat_Quality_Score' not in df.columns:
            return {"error": "Seat scores not calculated. Run bulk_score_seats_enhanced() first."}
        
        insights = {
            'distribution': {
                'premium_count': int(df['Is_Premium_Seat'].sum()),
                'good_count': int(df['Is_Good_Seat'].sum()),
                'average_count': int(df['Is_Average_Seat'].sum()),
                'poor_count': int(df['Is_Poor_Seat'].sum()),
                'total_count': len(df)
            },
            'score_statistics': {
                'mean_score': round(df['Seat_Quality_Score'].mean(), 2),
                'median_score': round(df['Seat_Quality_Score'].median(), 2),
                'std_score': round(df['Seat_Quality_Score'].std(), 2),
                'min_score': round(df['Seat_Quality_Score'].min(), 2),
                'max_score': round(df['Seat_Quality_Score'].max(), 2)
            },
            'pricing_impact': {
                'avg_multiplier': round(df['Seat_Pricing_Multiplier'].mean(), 3),
                'max_premium': round(df['Seat_Pricing_Multiplier'].max(), 3),
                'min_discount': round(df['Seat_Pricing_Multiplier'].min(), 3)
            }
        }
        
        # Category-specific insights
        if 'Category' in df.columns:
            category_insights = {}
            for category in df['Category'].unique():
                if pd.notna(category):
                    cat_data = df[df['Category'] == category]
                    category_insights[category] = {
                        'avg_seat_score': round(cat_data['Seat_Quality_Score'].mean(), 2),
                        'premium_percentage': round(cat_data['Is_Premium_Seat'].mean() * 100, 1),
                        'avg_pricing_multiplier': round(cat_data['Seat_Pricing_Multiplier'].mean(), 3)
                    }
            insights['by_category'] = category_insights
        
        return insights


def test_enhanced_seat_scoring():
    """Test the enhanced scoring system"""
    
    scorer = EnhancedSeatScorer()
    
    test_cases = [
        # (section, row, seats, venue, category, description)
        ('FLOOR GENERAL ADMISSION', 'GA', '1-2', 'Madison Square Garden', 'Pop', 'Premium floor at iconic venue'),
        ('CLUB F', '1', '7-8', 'Staples Center', 'NBA Basketball', 'Club level front row'),
        ('303', '24', '23-23', 'Regular Arena', 'Rock', 'Upper level back row'),
        ('VIP SUITE 101', 'A', '1-4', 'Wembley Stadium', 'NFL Football', 'VIP suite at premium venue'),
        ('LAWN GENERAL ADMISSION', 'GA', '15-16', 'Outdoor Amphitheater', 'Country', 'Lawn seating'),
        ('PREFERRED LOGE BOX 157', 'B', '9-10', 'Opera House', 'Classical', 'Premium theater seating'),
        ('117', '8', '12-13', 'Basketball Arena', 'NBA Basketball', 'Mid-level center seats'),
        ('OBSTRUCTED VIEW 421', '35', '1-2', 'Old Stadium', 'MLB Baseball', 'Poor view seats')
    ]
    
    print("🧪 TESTING ENHANCED SEAT SCORING SYSTEM")
    print("=" * 100)
    print(f"{'Section':<25} {'Row':<5} {'Seats':<8} {'Score':<6} {'Tier':<10} {'Multiplier':<10} {'Description'}")
    print("-" * 100)
    
    for section, row, seats, venue, category, description in test_cases:
        result = scorer.calculate_comprehensive_seat_score(section, row, seats, venue, category)
        
        print(f"{section:<25} {row:<5} {seats:<8} {result['seat_quality_score']:<6.1f} {result['seat_tier']:<10} {result['pricing_multiplier']:<10.3f} {description}")
        
        # Show detailed breakdown for first few examples
        if test_cases.index((section, row, seats, venue, category, description)) < 2:
            breakdown = result['score_breakdown']
            print(f"    Breakdown: Section={breakdown['section_score']:.1f}, Row={breakdown['row_score']:.1f}, "
                  f"Position={breakdown['seat_position_score']:.1f}, Venue={breakdown['venue_bonus']:.1f}, "
                  f"Category={breakdown['category_adjustment']:.1f}")
    
    print(f"\n🎯 Enhanced Scoring Guide:")
    print("   85-100: Premium (Floor, VIP, Front rows) - Pricing +15% to +25%")
    print("   70-84:  Excellent (Club, Low sections, Good rows) - Pricing +8% to +15%")
    print("   55-69:  Good (Mid-level, Decent rows) - Pricing +3% to +8%")
    print("   40-54:  Average (Standard seating) - Pricing baseline")
    print("   25-39:  Below Average (Higher sections, Back rows) - Pricing -5%")
    print("   0-24:   Poor (Obstructed, Very high sections) - Pricing -12%")


if __name__ == "__main__":
    test_enhanced_seat_scoring()