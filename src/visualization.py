"""
Visualization Module
Creates various visualizations for sentiment analysis results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_sentiment_distribution(df, sentiment_column='ensemble_sentiment', save_path=None):
    """
    Plot sentiment distribution
    
    Args:
        df: Input dataframe
        sentiment_column: Name of sentiment column
        save_path: Optional path to save plot
    """
    if sentiment_column not in df.columns:
        print(f"Error: Column '{sentiment_column}' not found")
        return
    
    # Count sentiments
    sentiment_counts = df[sentiment_column].value_counts()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    sentiment_colors = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]
    
    ax1.bar(sentiment_counts.index, sentiment_counts.values, color=sentiment_colors)
    ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=sentiment_colors, startangle=90)
    ax2.set_title('Sentiment Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sentiment distribution plot to {save_path}")
    
    plt.show()


def plot_state_sentiment(state_summary, top_n=15, save_path=None):
    """
    Plot state-wise sentiment
    
    Args:
        state_summary: State summary dataframe
        top_n: Number of top states to show
        save_path: Optional path to save plot
    """
    # Get top states by tweet count
    top_states = state_summary.nlargest(top_n, 'total_tweets')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sentiment index plot
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_states['sentiment_index']]
    ax1.barh(top_states['state'], top_states['sentiment_index'], color=colors)
    ax1.set_title('Sentiment Index by State (Top States)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment Index')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Sentiment distribution by state
    x = range(len(top_states))
    width = 0.25
    
    ax2.bar([i - width for i in x], top_states['positive_pct'], width, 
            label='Positive', color='#2ecc71')
    ax2.bar(x, top_states['neutral_pct'], width,
            label='Neutral', color='#95a5a6')
    ax2.bar([i + width for i in x], top_states['negative_pct'], width,
            label='Negative', color='#e74c3c')
    
    ax2.set_title('Sentiment Distribution by State', fontsize=14, fontweight='bold')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_states['state'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved state sentiment plot to {save_path}")
    
    plt.show()


def plot_tweet_count_by_state(state_summary, top_n=15, save_path=None):
    """
    Plot tweet count by state
    
    Args:
        state_summary: State summary dataframe
        top_n: Number of top states to show
        save_path: Optional path to save plot
    """
    # Get top states
    top_states = state_summary.nlargest(top_n, 'total_tweets')
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    colors = sns.color_palette('viridis', len(top_states))
    plt.barh(top_states['state'], top_states['total_tweets'], color=colors)
    
    plt.title(f'Tweet Count by State (Top {top_n})', fontsize=14, fontweight='bold')
    plt.xlabel('Total Tweets')
    plt.ylabel('State')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved tweet count plot to {save_path}")
    
    plt.show()


def plot_engagement_analysis(state_summary, top_n=10, save_path=None):
    """
    Plot engagement metrics by state
    
    Args:
        state_summary: State summary dataframe
        top_n: Number of top states to show
        save_path: Optional path to save plot
    """
    # Get top states by tweet count
    top_states = state_summary.nlargest(top_n, 'total_tweets')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average retweets
    ax1.barh(top_states['state'], top_states['avg_retweets'], color='#3498db')
    ax1.set_title('Average Retweets by State', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Average Retweets')
    ax1.grid(axis='x', alpha=0.3)
    
    # Average likes
    ax2.barh(top_states['state'], top_states['avg_likes'], color='#e74c3c')
    ax2.set_title('Average Likes by State', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Average Likes')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved engagement analysis plot to {save_path}")
    
    plt.show()


def plot_sentiment_heatmap(state_summary, save_path=None):
    """
    Plot sentiment heatmap for states
    
    Args:
        state_summary: State summary dataframe
        save_path: Optional path to save plot
    """
    # Prepare data for heatmap
    heatmap_data = state_summary[['state', 'positive_pct', 'neutral_pct', 'negative_pct']].head(20)
    heatmap_data = heatmap_data.set_index('state')
    
    # Create heatmap
    plt.figure(figsize=(10, 12))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=33.3, cbar_kws={'label': 'Percentage (%)'})
    
    plt.title('Sentiment Distribution Heatmap (Top 20 States)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Sentiment Type')
    plt.ylabel('State')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sentiment heatmap to {save_path}")
    
    plt.show()


def create_all_visualizations(df, state_summary, output_dir=None):
    """
    Create all visualizations
    
    Args:
        df: Main dataframe with sentiment analysis
        state_summary: State summary dataframe
        output_dir: Optional directory to save plots
    """
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating visualizations...")
    print("=" * 60)
    
    # Sentiment distribution
    print("\n1. Sentiment Distribution")
    save_path = output_path / 'sentiment_distribution.png' if output_dir else None
    plot_sentiment_distribution(df, save_path=save_path)
    
    # State sentiment
    print("\n2. State Sentiment Analysis")
    save_path = output_path / 'state_sentiment.png' if output_dir else None
    plot_state_sentiment(state_summary, save_path=save_path)
    
    # Tweet count by state
    print("\n3. Tweet Count by State")
    save_path = output_path / 'tweet_count_by_state.png' if output_dir else None
    plot_tweet_count_by_state(state_summary, save_path=save_path)
    
    # Engagement analysis
    print("\n4. Engagement Analysis")
    save_path = output_path / 'engagement_analysis.png' if output_dir else None
    plot_engagement_analysis(state_summary, save_path=save_path)
    
    # Sentiment heatmap
    print("\n5. Sentiment Heatmap")
    save_path = output_path / 'sentiment_heatmap.png' if output_dir else None
    plot_sentiment_heatmap(state_summary, save_path=save_path)
    
    print("\n" + "=" * 60)
    print("✓ All visualizations created successfully!")


def save_all_plots(df, state_summary, output_dir):
    """
    Save all plots to directory
    
    Args:
        df: Main dataframe
        state_summary: State summary dataframe
        output_dir: Output directory path
    """
    create_all_visualizations(df, state_summary, output_dir)
