"""
State Extraction Module
Extracts Indian state information from tweet text and location data
"""

import pandas as pd
import re
from pathlib import Path


class StateExtractor:
    """Extract Indian state information from tweets"""
    
    def __init__(self):
        # Indian states and union territories
        self.indian_states = {
            'Andhra Pradesh': ['andhra', 'andhra pradesh', 'ap'],
            'Arunachal Pradesh': ['arunachal', 'arunachal pradesh'],
            'Assam': ['assam'],
            'Bihar': ['bihar'],
            'Chhattisgarh': ['chhattisgarh', 'chattisgarh'],
            'Goa': ['goa'],
            'Gujarat': ['gujarat'],
            'Haryana': ['haryana'],
            'Himachal Pradesh': ['himachal', 'himachal pradesh', 'hp'],
            'Jharkhand': ['jharkhand'],
            'Karnataka': ['karnataka', 'bengaluru', 'bangalore'],
            'Kerala': ['kerala'],
            'Madhya Pradesh': ['madhya pradesh', 'mp', 'madhya'],
            'Maharashtra': ['maharashtra', 'mumbai', 'pune'],
            'Manipur': ['manipur'],
            'Meghalaya': ['meghalaya'],
            'Mizoram': ['mizoram'],
            'Nagaland': ['nagaland'],
            'Odisha': ['odisha', 'orissa'],
            'Punjab': ['punjab'],
            'Rajasthan': ['rajasthan', 'jaipur'],
            'Sikkim': ['sikkim'],
            'Tamil Nadu': ['tamil nadu', 'tn', 'chennai'],
            'Telangana': ['telangana', 'hyderabad'],
            'Tripura': ['tripura'],
            'Uttar Pradesh': ['uttar pradesh', 'up', 'lucknow'],
            'Uttarakhand': ['uttarakhand', 'uttaranchal'],
            'West Bengal': ['west bengal', 'wb', 'kolkata', 'calcutta'],
            'Delhi': ['delhi', 'new delhi'],
            'Jammu and Kashmir': ['jammu', 'kashmir', 'jammu and kashmir', 'j&k'],
            'Ladakh': ['ladakh'],
            'Andaman and Nicobar Islands': ['andaman', 'nicobar', 'andaman and nicobar'],
            'Chandigarh': ['chandigarh'],
            'Dadra and Nagar Haveli and Daman and Diu': ['dadra', 'nagar haveli', 'daman', 'diu'],
            'Lakshadweep': ['lakshadweep'],
            'Puducherry': ['puducherry', 'pondicherry']
        }
        
        # Major cities to state mapping
        self.city_to_state = {
            'mumbai': 'Maharashtra',
            'pune': 'Maharashtra',
            'nagpur': 'Maharashtra',
            'delhi': 'Delhi',
            'new delhi': 'Delhi',
            'bangalore': 'Karnataka',
            'bengaluru': 'Karnataka',
            'mysore': 'Karnataka',
            'chennai': 'Tamil Nadu',
            'hyderabad': 'Telangana',
            'kolkata': 'West Bengal',
            'calcutta': 'West Bengal',
            'ahmedabad': 'Gujarat',
            'surat': 'Gujarat',
            'jaipur': 'Rajasthan',
            'lucknow': 'Uttar Pradesh',
            'kanpur': 'Uttar Pradesh',
            'varanasi': 'Uttar Pradesh',
            'bhopal': 'Madhya Pradesh',
            'indore': 'Madhya Pradesh',
            'patna': 'Bihar',
            'chandigarh': 'Chandigarh',
            'thiruvananthapuram': 'Kerala',
            'kochi': 'Kerala',
            'guwahati': 'Assam',
            'bhubaneswar': 'Odisha',
            'ranchi': 'Jharkhand',
            'raipur': 'Chhattisgarh',
            'shimla': 'Himachal Pradesh',
            'dehradun': 'Uttarakhand'
        }
    
    def extract_state_from_text(self, text):
        """
        Extract state from tweet text
        
        Args:
            text: Tweet text
            
        Returns:
            State name or None
        """
        if pd.isna(text):
            return None
        
        text_lower = str(text).lower()
        
        # Check for state mentions
        for state, keywords in self.indian_states.items():
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    return state
        
        # Check for city mentions
        for city, state in self.city_to_state.items():
            pattern = r'\b' + re.escape(city) + r'\b'
            if re.search(pattern, text_lower):
                return state
        
        return None
    
    def extract_state_from_location(self, location):
        """
        Extract state from location field
        
        Args:
            location: Location string
            
        Returns:
            State name or None
        """
        if pd.isna(location):
            return None
        
        location_lower = str(location).lower()
        
        # Check for state names
        for state, keywords in self.indian_states.items():
            for keyword in keywords:
                if keyword in location_lower:
                    return state
        
        # Check for cities
        for city, state in self.city_to_state.items():
            if city in location_lower:
                return state
        
        return None
    
    def extract_states(self, df, text_column='text', location_column='location'):
        """
        Extract states from dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            location_column: Name of location column
            
        Returns:
            Dataframe with state column
        """
        print(f"Extracting states from {len(df)} tweets...")
        
        df = df.copy()
        
        # Extract from location first
        if location_column in df.columns:
            df['state_from_location'] = df[location_column].apply(self.extract_state_from_location)
        else:
            df['state_from_location'] = None
        
        # Extract from text
        if text_column in df.columns:
            df['state_from_text'] = df[text_column].apply(self.extract_state_from_text)
        else:
            df['state_from_text'] = None
        
        # Combine: prioritize location over text
        df['state'] = df['state_from_location'].fillna(df['state_from_text'])
        
        # Count states found
        states_found = df['state'].notna().sum()
        print(f"✓ Found states in {states_found}/{len(df)} tweets ({states_found/len(df)*100:.1f}%)")
        
        # Remove temporary columns
        df = df.drop(['state_from_location', 'state_from_text'], axis=1)
        
        return df
    
    def filter_indian_tweets(self, df):
        """
        Filter dataframe to only include tweets with identified Indian states
        
        Args:
            df: Input dataframe
            
        Returns:
            Filtered dataframe
        """
        if 'state' not in df.columns:
            print("Error: 'state' column not found. Run extract_states() first.")
            return df
        
        initial_count = len(df)
        df = df[df['state'].notna()].copy()
        final_count = len(df)
        
        print(f"Filtered to {final_count}/{initial_count} Indian tweets ({final_count/initial_count*100:.1f}%)")
        
        return df


class GeographicAnalyzer:
    """Analyze geographic distribution of tweets"""
    
    def __init__(self):
        self.extractor = StateExtractor()
    
    def get_state_distribution(self, df):
        """
        Get distribution of tweets across states
        
        Args:
            df: Input dataframe with state column
            
        Returns:
            Dataframe with state counts
        """
        if 'state' not in df.columns:
            print("Error: 'state' column not found")
            return pd.DataFrame()
        
        state_counts = df['state'].value_counts().reset_index()
        state_counts.columns = ['state', 'tweet_count']
        state_counts['percentage'] = (state_counts['tweet_count'] / len(df) * 100).round(2)
        
        return state_counts
    
    def get_top_states(self, df, n=10):
        """Get top N states by tweet count"""
        dist = self.get_state_distribution(df)
        return dist.head(n)
    
    def analyze_coverage(self, df):
        """
        Analyze geographic coverage
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with coverage statistics
        """
        if 'state' not in df.columns:
            return {}
        
        total_tweets = len(df)
        tweets_with_state = df['state'].notna().sum()
        unique_states = df['state'].nunique()
        
        coverage = {
            'total_tweets': total_tweets,
            'tweets_with_state': tweets_with_state,
            'tweets_without_state': total_tweets - tweets_with_state,
            'coverage_percentage': round(tweets_with_state / total_tweets * 100, 2),
            'unique_states': unique_states,
            'total_indian_states': 36  # 28 states + 8 union territories
        }
        
        return coverage


def extract_states_from_dataframe(df, text_column='text', location_column='location'):
    """
    Convenience function to extract states from dataframe
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        location_column: Name of location column
        
    Returns:
        Dataframe with state information
    """
    extractor = StateExtractor()
    return extractor.extract_states(df, text_column, location_column)
