"""
State Aggregation Module
Handles state-wise aggregation and analysis of sentiment data
"""

import pandas as pd
import numpy as np
from pathlib import Path


class StateWiseAggregator:
    """Aggregate sentiment data by Indian states"""
    
    def __init__(self):
        self.state_column = 'state'
        self.sentiment_column = 'ensemble_sentiment'
        
    def generate_state_summary(self, df, sentiment_column='ensemble_sentiment'):
        """
        Generate comprehensive summary for each state
        
        Args:
            df: DataFrame with complete analysis
            sentiment_column: Sentiment column name
            
        Returns:
            Summary DataFrame
        """
        if 'state' not in df.columns:
            print("Error: 'state' column not found in dataframe")
            return pd.DataFrame()
        
        print("Generating state-wise summary...")
        
        summary_stats = []
        
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            
            # Basic stats
            total_tweets = len(state_df)
            
            # Sentiment distribution
            sentiment_counts = state_df[sentiment_column].value_counts()
            positive_pct = (sentiment_counts.get('positive', 0) / total_tweets * 100) if total_tweets > 0 else 0
            negative_pct = (sentiment_counts.get('negative', 0) / total_tweets * 100) if total_tweets > 0 else 0
            neutral_pct = (sentiment_counts.get('neutral', 0) / total_tweets * 100) if total_tweets > 0 else 0
            
            # Engagement stats
            avg_retweets = state_df['retweet_count'].mean() if 'retweet_count' in state_df.columns else 0
            avg_likes = state_df['like_count'].mean() if 'like_count' in state_df.columns else 0
            
            # Sentiment index
            sentiment_index = (positive_pct - negative_pct) / 100
            
            # Overall sentiment
            if sentiment_index > 0.1:
                overall = 'Positive'
            elif sentiment_index < -0.1:
                overall = 'Negative'
            else:
                overall = 'Neutral'
            
            summary_stats.append({
                'state': state,
                'total_tweets': total_tweets,
                'positive_pct': round(positive_pct, 2),
                'neutral_pct': round(neutral_pct, 2),
                'negative_pct': round(negative_pct, 2),
                'sentiment_index': round(sentiment_index, 3),
                'overall_sentiment': overall,
                'avg_retweets': round(avg_retweets, 2),
                'avg_likes': round(avg_likes, 2)
            })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df.sort_values('total_tweets', ascending=False)
        
        print(f"✓ Generated summary for {len(summary_df)} states")
        
        return summary_df
    
    def get_top_states(self, summary_df, n=10, metric='total_tweets'):
        """
        Get top N states by metric
        
        Args:
            summary_df: State summary dataframe
            n: Number of states to return
            metric: Metric to sort by
            
        Returns:
            Top N states dataframe
        """
        if metric not in summary_df.columns:
            print(f"Error: Metric '{metric}' not found")
            return summary_df
        
        return summary_df.nlargest(n, metric)
    
    def get_top_positive_states(self, summary_df, n=10):
        """Get states with highest positive sentiment"""
        return summary_df.nlargest(n, 'positive_pct')
    
    def get_top_negative_states(self, summary_df, n=10):
        """Get states with highest negative sentiment"""
        return summary_df.nlargest(n, 'negative_pct')
    
    def save_summary(self, summary_df, output_dir):
        """
        Save state summary to CSV files
        
        Args:
            summary_df: State summary dataframe
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main summary
        main_file = output_path / 'state_sentiment_summary.csv'
        summary_df.to_csv(main_file, index=False)
        print(f"✓ Saved state summary to {main_file}")
        
        # Save top positive states
        top_positive = self.get_top_positive_states(summary_df, n=10)
        pos_file = output_path / 'state_sentiment_top_positive.csv'
        top_positive.to_csv(pos_file, index=False)
        print(f"✓ Saved top positive states to {pos_file}")
        
        # Save top negative states
        top_negative = self.get_top_negative_states(summary_df, n=10)
        neg_file = output_path / 'state_sentiment_top_negative.csv'
        top_negative.to_csv(neg_file, index=False)
        print(f"✓ Saved top negative states to {neg_file}")
        
        # Save full state aggregation
        agg_file = output_path / 'state_sentiment_aggregation.csv'
        summary_df.to_csv(agg_file, index=False)
        print(f"✓ Saved state aggregation to {agg_file}")


class RegionalAnalyzer:
    """Analyze sentiment by Indian regions"""
    
    def __init__(self):
        # Define Indian regions
        self.regions = {
            'North': ['Delhi', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 
                     'Punjab', 'Rajasthan', 'Uttarakhand', 'Uttar Pradesh', 'Chandigarh'],
            'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana',
                     'Andaman and Nicobar Islands', 'Lakshadweep', 'Puducherry'],
            'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal'],
            'West': ['Goa', 'Gujarat', 'Maharashtra', 'Dadra and Nagar Haveli and Daman and Diu'],
            'Northeast': ['Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 
                         'Mizoram', 'Nagaland', 'Sikkim', 'Tripura'],
            'Central': ['Chhattisgarh', 'Madhya Pradesh']
        }
    
    def add_region_column(self, df):
        """Add region column to dataframe based on state"""
        if 'state' not in df.columns:
            return df
        
        # Create state to region mapping
        state_to_region = {}
        for region, states in self.regions.items():
            for state in states:
                state_to_region[state] = region
        
        # Add region column
        df['region'] = df['state'].map(state_to_region)
        df['region'] = df['region'].fillna('Other')
        
        return df
    
    def generate_regional_summary(self, df, sentiment_column='ensemble_sentiment'):
        """
        Generate regional sentiment summary
        
        Args:
            df: Input dataframe with region column
            sentiment_column: Sentiment column name
            
        Returns:
            Regional summary dataframe
        """
        # Add region if not present
        if 'region' not in df.columns:
            df = self.add_region_column(df)
        
        print("Generating regional summary...")
        
        regional_stats = []
        
        for region in df['region'].unique():
            region_df = df[df['region'] == region]
            
            # Basic stats
            total_tweets = len(region_df)
            num_states = region_df['state'].nunique() if 'state' in region_df.columns else 0
            
            # Sentiment distribution
            sentiment_counts = region_df[sentiment_column].value_counts()
            positive_pct = (sentiment_counts.get('positive', 0) / total_tweets * 100) if total_tweets > 0 else 0
            negative_pct = (sentiment_counts.get('negative', 0) / total_tweets * 100) if total_tweets > 0 else 0
            neutral_pct = (sentiment_counts.get('neutral', 0) / total_tweets * 100) if total_tweets > 0 else 0
            
            # Sentiment index
            sentiment_index = (positive_pct - negative_pct) / 100
            
            regional_stats.append({
                'region': region,
                'total_tweets': total_tweets,
                'num_states': num_states,
                'positive_pct': round(positive_pct, 2),
                'neutral_pct': round(neutral_pct, 2),
                'negative_pct': round(negative_pct, 2),
                'sentiment_index': round(sentiment_index, 3)
            })
        
        regional_df = pd.DataFrame(regional_stats)
        regional_df = regional_df.sort_values('total_tweets', ascending=False)
        
        print(f"✓ Generated summary for {len(regional_df)} regions")
        
        return regional_df


def aggregate_by_state(df, sentiment_column='ensemble_sentiment', output_dir=None):
    """
    Convenience function for state-wise aggregation
    
    Args:
        df: Input dataframe
        sentiment_column: Sentiment column name
        output_dir: Optional output directory to save results
        
    Returns:
        State summary dataframe
    """
    aggregator = StateWiseAggregator()
    summary = aggregator.generate_state_summary(df, sentiment_column)
    
    if output_dir:
        aggregator.save_summary(summary, output_dir)
    
    return summary
