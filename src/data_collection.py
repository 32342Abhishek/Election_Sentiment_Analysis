"""
Data Collection Module
Handles data collection from various sources including Twitter API, Kaggle, and sample data generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random


class SampleDataGenerator:
    """Generate sample Twitter data for testing purposes"""
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
        # Sample templates
        self.positive_templates = [
            "Great speech by {leader}! #IndianElections",
            "{leader} is doing amazing work for the country",
            "Support {leader} for better India! #Vote2024",
            "Impressed by {leader}'s vision",
            "{leader}'s policies are working great"
        ]
        
        self.negative_templates = [
            "Disappointed with {leader}'s performance",
            "{leader} needs to improve #IndianElections",
            "Not happy with {leader}'s decisions",
            "{leader}'s policies are failing",
            "Can't support {leader} anymore"
        ]
        
        self.neutral_templates = [
            "{leader} spoke at the rally today",
            "News: {leader} visits {state}",
            "{leader} announces new policy",
            "Meeting scheduled with {leader}",
            "{leader} campaign update"
        ]
        
        self.leaders = ['Modi', 'Rahul Gandhi', 'Kejriwal', 'Mamata Banerjee']
        
    def generate(self):
        """Generate sample dataset"""
        data = []
        
        for i in range(self.num_samples):
            # Random sentiment
            sentiment = random.choice(['positive', 'negative', 'neutral'])
            
            # Select template based on sentiment
            if sentiment == 'positive':
                template = random.choice(self.positive_templates)
            elif sentiment == 'negative':
                template = random.choice(self.negative_templates)
            else:
                template = random.choice(self.neutral_templates)
            
            # Fill template
            leader = random.choice(self.leaders)
            text = template.format(leader=leader, state=random.choice(['Delhi', 'Mumbai', 'Bangalore']))
            
            # Create record
            record = {
                'id': i,
                'text': text,
                'created_at': datetime.now() - timedelta(days=random.randint(0, 30)),
                'user_id': f'user_{random.randint(1000, 9999)}',
                'retweet_count': random.randint(0, 100),
                'like_count': random.randint(0, 500),
                'location': random.choice(['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bangalore']),
                'verified': random.choice([True, False])
            }
            
            data.append(record)
        
        return pd.DataFrame(data)


class TwitterDataCollector:
    """Collect data from Twitter API"""
    
    def __init__(self, api_keys=None):
        self.api_keys = api_keys
        
    def collect(self, query, max_results=1000):
        """Collect tweets based on query"""
        print(f"Note: Twitter API collection requires valid API keys")
        print(f"Generating sample data instead...")
        
        generator = SampleDataGenerator(num_samples=max_results)
        return generator.generate()


class KaggleDataLoader:
    """Load data from Kaggle datasets"""
    
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        
    def load(self):
        """Load dataset from Kaggle"""
        if self.dataset_path and Path(self.dataset_path).exists():
            print(f"Loading data from {self.dataset_path}")
            return pd.read_csv(self.dataset_path)
        else:
            print("Dataset path not found. Using sample data.")
            generator = SampleDataGenerator(num_samples=1000)
            return generator.generate()
    
    def load_multiple_datasets(self, dataset_paths):
        """Load and combine multiple datasets"""
        dfs = []
        for path in dataset_paths:
            if Path(path).exists():
                df = pd.read_csv(path)
                dfs.append(df)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            return combined
        else:
            print("No valid datasets found. Using sample data.")
            generator = SampleDataGenerator(num_samples=1000)
            return generator.generate()


def collect_twitter_data(query, max_results=1000, api_keys=None):
    """
    Convenience function to collect Twitter data
    
    Args:
        query: Search query
        max_results: Maximum number of results
        api_keys: Twitter API keys
        
    Returns:
        DataFrame with collected tweets
    """
    collector = TwitterDataCollector(api_keys)
    return collector.collect(query, max_results)


def generate_sample_data(num_samples=1000):
    """
    Generate sample Twitter data for testing
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample tweets
    """
    generator = SampleDataGenerator(num_samples)
    return generator.generate()
