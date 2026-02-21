"""
Sentiment Analysis Module
Implements multiple sentiment analysis approaches including VADER, TextBlob, and ensemble methods
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class EnsembleSentimentAnalyzer:
    """Ensemble sentiment analyzer combining multiple methods"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_vader(self, text):
        """Analyze sentiment using VADER"""
        if pd.isna(text) or text == "":
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
        
        scores = self.vader.polarity_scores(str(text))
        return scores
    
    def analyze_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        if pd.isna(text) or text == "":
            return 0
        
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0
    
    def get_sentiment_label(self, compound_score):
        """Convert compound score to sentiment label"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_single(self, text):
        """
        Analyze single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and label
        """
        # VADER scores
        vader_scores = self.analyze_vader(text)
        
        # TextBlob score
        textblob_score = self.analyze_textblob(text)
        
        # Ensemble score (weighted average)
        ensemble_score = (vader_scores['compound'] * 0.7 + textblob_score * 0.3)
        
        # Get sentiment label
        sentiment = self.get_sentiment_label(ensemble_score)
        
        result = {
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neu': vader_scores['neu'],
            'vader_neg': vader_scores['neg'],
            'textblob_polarity': textblob_score,
            'ensemble_score': ensemble_score,
            'ensemble_sentiment': sentiment
        }
        
        return result
    
    def analyze_dataframe(self, df, text_column='text'):
        """
        Analyze sentiment for entire dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            
        Returns:
            Dataframe with sentiment scores
        """
        if text_column not in df.columns:
            print(f"Error: Column '{text_column}' not found")
            return df
        
        print(f"Analyzing sentiment for {len(df)} tweets...")
        
        # Analyze each tweet
        results = []
        for idx, text in enumerate(df[text_column]):
            if idx % 1000 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(df)} tweets")
            
            result = self.analyze_single(text)
            results.append(result)
        
        # Add results to dataframe
        result_df = pd.DataFrame(results)
        df = pd.concat([df.reset_index(drop=True), result_df], axis=1)
        
        print(f"✓ Sentiment analysis complete")
        
        # Print summary
        sentiment_counts = df['ensemble_sentiment'].value_counts()
        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {sentiment.capitalize()}: {count} ({pct:.1f}%)")
        
        return df


class MLSentimentAnalyzer:
    """Machine Learning based sentiment analyzer"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        
    def train(self, X_train, y_train):
        """
        Train sentiment model
        
        Args:
            X_train: Training texts
            y_train: Training labels
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        print("Training ML sentiment model...")
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_vec, y_train)
        
        print("✓ Model training complete")
    
    def predict(self, texts):
        """
        Predict sentiment for texts
        
        Args:
            texts: List of texts or single text
            
        Returns:
            Predicted sentiments
        """
        if self.model is None or self.vectorizer is None:
            print("Error: Model not trained yet")
            return None
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Vectorize and predict
        X_vec = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_vec)
        
        return predictions
    
    def predict_dataframe(self, df, text_column='text'):
        """
        Predict sentiment for dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            
        Returns:
            Dataframe with predictions
        """
        if text_column not in df.columns:
            print(f"Error: Column '{text_column}' not found")
            return df
        
        predictions = self.predict(df[text_column].tolist())
        df['ml_sentiment'] = predictions
        
        return df


def analyze_sentiment(df, text_column='text', method='ensemble'):
    """
    Convenience function for sentiment analysis
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        method: Analysis method ('ensemble', 'vader', 'textblob')
        
    Returns:
        Dataframe with sentiment scores
    """
    if method == 'ensemble':
        analyzer = EnsembleSentimentAnalyzer()
        return analyzer.analyze_dataframe(df, text_column)
    elif method == 'vader':
        analyzer = EnsembleSentimentAnalyzer()
        vader_results = df[text_column].apply(analyzer.analyze_vader)
        vader_df = pd.DataFrame(vader_results.tolist())
        vader_df['sentiment'] = vader_df['compound'].apply(analyzer.get_sentiment_label)
        return pd.concat([df, vader_df], axis=1)
    elif method == 'textblob':
        df['textblob_polarity'] = df[text_column].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
        )
        df['sentiment'] = df['textblob_polarity'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        return df
    else:
        print(f"Unknown method: {method}")
        return df
