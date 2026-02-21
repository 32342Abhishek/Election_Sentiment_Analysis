"""
Twitter Sentiment Analysis - Source Package
Contains utility modules for data processing and analysis
"""

__version__ = "1.0.0"
__author__ = "Election Sentiment Analysis Team"

# Package exports
from .data_collection import SampleDataGenerator, TwitterDataCollector, KaggleDataLoader
from .data_preprocessing import TweetDatasetProcessor
from .real_data_processor import combine_all_datasets, save_real_data
from .sentiment_analysis import EnsembleSentimentAnalyzer, MLSentimentAnalyzer
from .state_aggregation import StateWiseAggregator, RegionalAnalyzer
from .state_extraction import StateExtractor, GeographicAnalyzer
from .visualization import create_all_visualizations

__all__ = [
    'SampleDataGenerator',
    'TwitterDataCollector',
    'KaggleDataLoader',
    'TweetDatasetProcessor',
    'combine_all_datasets',
    'save_real_data',
    'EnsembleSentimentAnalyzer',
    'MLSentimentAnalyzer',
    'StateWiseAggregator',
    'RegionalAnalyzer',
    'StateExtractor',
    'GeographicAnalyzer',
    'create_all_visualizations'
]
