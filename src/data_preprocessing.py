"""
Data Preprocessing Module
Handles text cleaning, normalization, and preprocessing for tweets
"""

import pandas as pd
import re
import string
from pathlib import Path


class TweetDatasetProcessor:
    """Process and clean tweet datasets"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        
    def clean_text(self, text):
        """Clean tweet text"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def remove_mentions(self, text):
        """Remove @mentions from text"""
        if pd.isna(text):
            return ""
        return self.mention_pattern.sub('', str(text))
    
    def remove_hashtags(self, text):
        """Remove hashtags from text"""
        if pd.isna(text):
            return ""
        return self.hashtag_pattern.sub('', str(text))
    
    def extract_hashtags(self, text):
        """Extract hashtags from text"""
        if pd.isna(text):
            return []
        matches = self.hashtag_pattern.findall(str(text))
        return [tag.lower() for tag in matches]
    
    def normalize_text(self, text):
        """Normalize text to lowercase"""
        if pd.isna(text):
            return ""
        return str(text).lower()
    
    def remove_punctuation(self, text):
        """Remove punctuation from text"""
        if pd.isna(text):
            return ""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def process_dataframe(self, df, text_column='text'):
        """
        Process entire dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column to process
            
        Returns:
            Processed dataframe
        """
        df = df.copy()
        
        # Check if text column exists
        if text_column not in df.columns:
            print(f"Warning: Column '{text_column}' not found in dataframe")
            return df
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Extract hashtags
        df['hashtags'] = df[text_column].apply(self.extract_hashtags)
        
        # Add text length
        df['text_length'] = df['cleaned_text'].apply(lambda x: len(str(x)))
        
        # Remove empty texts
        df = df[df['text_length'] > 0].copy()
        
        return df
    
    def filter_by_keywords(self, df, keywords, text_column='text'):
        """Filter dataframe by keywords"""
        if text_column not in df.columns:
            return df
        
        pattern = '|'.join(keywords)
        mask = df[text_column].str.contains(pattern, case=False, na=False)
        return df[mask].copy()
    
    def remove_duplicates(self, df, text_column='text'):
        """Remove duplicate tweets"""
        if text_column not in df.columns:
            return df
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=[text_column]).copy()
        final_count = len(df)
        
        print(f"Removed {initial_count - final_count} duplicate tweets")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in dataframe"""
        # Fill missing text with empty string
        if 'text' in df.columns:
            df['text'] = df['text'].fillna('')
        
        # Fill missing numeric values with 0
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def preprocess_pipeline(self, df, text_column='text'):
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            
        Returns:
            Preprocessed dataframe
        """
        print("Starting preprocessing pipeline...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        print(f"✓ Handled missing values")
        
        # Remove duplicates
        df = self.remove_duplicates(df, text_column)
        print(f"✓ Removed duplicates")
        
        # Process text
        df = self.process_dataframe(df, text_column)
        print(f"✓ Processed text")
        
        print(f"Preprocessing complete. Final dataset: {len(df)} tweets")
        
        return df


def preprocess_tweets(df, text_column='text'):
    """
    Convenience function for preprocessing tweets
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        
    Returns:
        Preprocessed dataframe
    """
    processor = TweetDatasetProcessor()
    return processor.preprocess_pipeline(df, text_column)
