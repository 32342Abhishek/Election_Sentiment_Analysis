"""
Election Sentiment Analysis - Professional Web Application
Real-time Interactive Dashboard for Twitter Sentiment Analysis on Indian Elections
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    INDIAN_STATES, OUTPUTS_DIR, PROCESSED_DATA_DIR, 
    RAW_DATA_DIR, VISUALIZATIONS_DIR
)

# Page configuration
st.set_page_config(
    page_title="Election Sentiment Analysis",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main background and theme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin: 20px;
    }
    
    /* Header styling */
    h1 {
        color: #1a202c;
        font-weight: 800;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 700;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-top: 2rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 600;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a202c;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #4a5568;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid #e2e8f0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar styling - DARK BACKGROUND WITH WHITE TEXT */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%) !important;
        border-right: 3px solid #667eea;
    }
    
    section[data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* WHITE TEXT ON DARK BACKGROUND */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #ffffff !important;
    }
    
    /* Selectbox and dropdown text */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSelectbox div {
        color: #ffffff !important;
    }
    
    /* Checkbox text */
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Metric labels and values - DARK TEXT LIKE STATUS BOXES */
    section[data-testid="stSidebar"] .stMetric {
        background-color: transparent !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] *,
    section[data-testid="stSidebar"] .stMetric label {
        color: #1a202c !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricValue"],
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] *,
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #1a202c !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricDelta"],
    section[data-testid="stSidebar"] [data-testid="stMetricDelta"] * {
        color: #2d3748 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    
    /* Force all metric text to dark color */
    section[data-testid="stSidebar"] .stMetric * {
        color: #1a202c !important;
    }
    
    section[data-testid="stSidebar"] .stMetric div[data-testid="stMetricValue"] {
        color: #1a202c !important;
    }
    
    /* Markdown text in sidebar */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Status boxes - Make text highly visible */
    .stSuccess, .stError, .stInfo, .stWarning {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    .stSuccess > div, .stSuccess p, .stSuccess * {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    .stError > div, .stError p, .stError * {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    .stInfo > div, .stInfo p, .stInfo * {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    [data-testid="stMarkdownContainer"] > div > div.stSuccess,
    [data-testid="stMarkdownContainer"] > div > div.stError,
    [data-testid="stMarkdownContainer"] > div > div.stInfo {
        color: #1a202c !important;
    }
    
    div[data-baseweb="notification"] {
        color: #1a202c !important;
    }
    
    div[data-baseweb="notification"] * {
        color: #1a202c !important;
    }
    
    /* Preserve inline styled sentiment cards */
    [data-testid="stMarkdownContainer"] > div > div[style*="background: linear-gradient"],
    [data-testid="stMarkdownContainer"] > div > div[style*="background: linear-gradient"] * {
        color: white !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f7fafc;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #667eea;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Plotly chart containers */
    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed sentiment data with optimization"""
    try:
        # Load the latest processed data
        sentiment_file = PROCESSED_DATA_DIR / 'tweets_with_sentiment.csv'
        if sentiment_file.exists():
            # Only load essential columns for dashboard to speed up loading
            essential_cols = [
                'text', 'created_at', 'like_count', 'retweet_count', 'reply_count',
                'state', 'ensemble_sentiment', 'vader_compound',
                'textblob_polarity', 'cleaned_text'
            ]
            
            # Load 100k rows for comprehensive analysis with reasonable speed
            # This provides good state coverage while keeping dashboard responsive
            df = pd.read_csv(
                sentiment_file,
                nrows=100000,
                usecols=essential_cols,
                low_memory=False,
                encoding='utf-8'
            )
            
            return df
        else:
            st.warning("No data found. Please run the analysis pipeline first.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


@st.cache_data
def load_state_summary():
    """Load state aggregation summary"""
    try:
        summary_file = OUTPUTS_DIR / 'state_sentiment_summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file, index_col=0)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading summary: {str(e)}")
        return pd.DataFrame()


def get_sentiment_color(sentiment):
    """Return color based on sentiment"""
    colors = {
        'positive': '#2ecc71',
        'neutral': '#95a5a6',
        'negative': '#e74c3c'
    }
    return colors.get(sentiment.lower(), '#95a5a6')


def create_sentiment_gauge(positive_pct, negative_pct, neutral_pct):
    """Create sentiment gauge chart"""
    sentiment_index = (positive_pct - negative_pct) / 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_index * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Index", 'font': {'size': 24}},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-100, -33], 'color': '#ffcccc'},
                {'range': [-33, 33], 'color': '#ffffcc'},
                {'range': [33, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_index * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_state_map(state_data):
    """Create interactive India map with state sentiments"""
    fig = px.choropleth(
        state_data,
        locations='state',
        locationmode='country names',
        color='sentiment_index',
        hover_name='state',
        hover_data={
            'total_tweets': True,
            'positive_pct': ':.1f',
            'negative_pct': ':.1f',
            'sentiment_index': ':.3f'
        },
        color_continuous_scale=[
            [0, '#e74c3c'],    # Red for negative
            [0.5, '#f39c12'],  # Orange for neutral
            [1, '#2ecc71']     # Green for positive
        ],
        labels={'sentiment_index': 'Sentiment Index'},
        title="State-wise Sentiment Distribution"
    )
    
    fig.update_geos(
        visible=False,
        showcountries=True,
        countrycolor="lightgray"
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_trend_chart(df, state=None):
    """Create sentiment trend over time"""
    if 'created_at' in df.columns:
        df_copy = df.copy()
        # Handle various date formats with error handling
        df_copy['created_at'] = pd.to_datetime(df_copy['created_at'], errors='coerce', utc=True, format='mixed')
        
        # Remove rows with invalid dates
        df_copy = df_copy.dropna(subset=['created_at'])
        
        if df_copy.empty:
            # If no valid dates, return empty chart
            fig = go.Figure()
            fig.add_annotation(text="No date information available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        if state and state != 'All States':
            df_copy = df_copy[df_copy['state'] == state]
        
        # Group by date and sentiment
        trend_data = df_copy.groupby([
            df_copy['created_at'].dt.date,
            'ensemble_sentiment'
        ]).size().reset_index(name='count')
        
        fig = px.line(
            trend_data,
            x='created_at',
            y='count',
            color='ensemble_sentiment',
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#95a5a6',
                'negative': '#e74c3c'
            },
            title=f"Sentiment Trend Over Time{' - ' + state if state and state != 'All States' else ''}",
            labels={'created_at': 'Date', 'count': 'Number of Tweets'}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        return fig
    return None


def create_comparison_chart(state_summary, top_n=10):
    """Create state comparison bar chart"""
    top_states = state_summary.nlargest(top_n, 'total_tweets')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Positive',
        x=top_states.index,
        y=top_states['positive_pct'],
        marker_color='#2ecc71'
    ))
    
    fig.add_trace(go.Bar(
        name='Neutral',
        x=top_states.index,
        y=top_states['neutral_pct'],
        marker_color='#95a5a6'
    ))
    
    fig.add_trace(go.Bar(
        name='Negative',
        x=top_states.index,
        y=top_states['negative_pct'],
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        barmode='stack',
        title=f"Top {top_n} States by Tweet Volume - Sentiment Distribution",
        xaxis_title="State",
        yaxis_title="Percentage (%)",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_engagement_analysis(df, state=None):
    """Create engagement analysis chart"""
    df_copy = df.copy()
    
    if state and state != 'All States':
        df_copy = df_copy[df_copy['state'] == state]
    
    engagement_data = df_copy.groupby('ensemble_sentiment').agg({
        'like_count': 'mean',
        'retweet_count': 'mean',
        'reply_count': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Likes',
        x=engagement_data['ensemble_sentiment'],
        y=engagement_data['like_count'],
        marker_color='#3498db'
    ))
    
    fig.add_trace(go.Bar(
        name='Retweets',
        x=engagement_data['ensemble_sentiment'],
        y=engagement_data['retweet_count'],
        marker_color='#9b59b6'
    ))
    
    fig.add_trace(go.Bar(
        name='Replies',
        x=engagement_data['ensemble_sentiment'],
        y=engagement_data['reply_count'],
        marker_color='#e67e22'
    ))
    
    fig.update_layout(
        barmode='group',
        title=f"Average Engagement by Sentiment{' - ' + state if state and state != 'All States' else ''}",
        xaxis_title="Sentiment",
        yaxis_title="Average Count",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def extract_hashtags(text):
    """Extract hashtags from tweet text"""
    if pd.isna(text):
        return []
    return re.findall(r'#\w+', str(text).lower())


def create_wordcloud_fig(df, sentiment_filter=None, max_words=100):
    """Create word cloud visualization"""
    try:
        # Filter by sentiment if specified
        if sentiment_filter:
            df_filtered = df[df['ensemble_sentiment'] == sentiment_filter]
        else:
            df_filtered = df
        
        if df_filtered.empty:
            return None
        
        # Limit to 10000 tweets max for performance
        if len(df_filtered) > 10000:
            df_filtered = df_filtered.sample(10000, random_state=42)
        
        # Combine all text
        text = ' '.join(df_filtered['text'].dropna().astype(str))
        
        # Remove URLs, mentions, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)  # Remove # but keep the word
        
        # Create word cloud
        if sentiment_filter == 'positive':
            colormap = 'Greens'
        elif sentiment_filter == 'negative':
            colormap = 'Reds'
        else:
            colormap = 'Blues'
        
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap=colormap,
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        plt.close('all')  # Clean up
        return None


def analyze_hashtags(df, sentiment_filter=None, top_n=20):
    """Analyze hashtag frequency"""
    try:
        # Filter by sentiment if specified
        if sentiment_filter:
            df_filtered = df[df['ensemble_sentiment'] == sentiment_filter]
        else:
            df_filtered = df
        
        # Limit to 50000 tweets max for performance
        if len(df_filtered) > 50000:
            df_filtered = df_filtered.sample(50000, random_state=42)
        
        # Extract all hashtags
        all_hashtags = []
        for text in df_filtered['text'].dropna():
            all_hashtags.extend(extract_hashtags(text))
        
        if not all_hashtags:
            return None
        
        # Count frequency
        hashtag_counts = Counter(all_hashtags)
        top_hashtags = hashtag_counts.most_common(top_n)
        
        # Create dataframe
        hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
        
        # Create bar chart
        fig = px.bar(
            hashtag_df,
            x='Count',
            y='Hashtag',
            orientation='h',
            title=f"Top {top_n} Hashtags{' - ' + sentiment_filter.title() if sentiment_filter else ''}",
            color='Count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    except Exception as e:
        st.error(f"Error analyzing hashtags: {str(e)}")
        return None


@st.cache_data
def load_leader_data():
    """Load politician-specific datasets with fast VADER sentiment analysis"""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    leaders = {}
    leader_files = {
        'Narendra Modi': 'Narendra Modi_data.csv',
        'Rahul Gandhi': 'Rahul Gandhi_data.csv',
        'Arvind Kejriwal': 'Arvind Kejriwal_data.csv'
    }
    
    vader = SentimentIntensityAnalyzer()
    
    for leader, filename in leader_files.items():
        file_path = RAW_DATA_DIR / filename
        if file_path.exists():
            try:
                # Load sample data for faster processing
                df = pd.read_csv(file_path, nrows=3000, encoding='utf-8', on_bad_lines='skip')
                
                # Check for text column (try different possible names)
                text_col = None
                for col in ['text', 'Text', 'tweet', 'Tweet', 'content', 'Content']:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col:
                    # Quick sentiment analysis with VADER
                    sentiments = []
                    for text in df[text_col].dropna().head(1000):  # Analyze first 1000 tweets
                        try:
                            scores = vader.polarity_scores(str(text))
                            if scores['compound'] >= 0.05:
                                sentiments.append('positive')
                            elif scores['compound'] <= -0.05:
                                sentiments.append('negative')
                            else:
                                sentiments.append('neutral')
                        except:
                            sentiments.append('neutral')
                    
                    # Pad with neutral if needed
                    df['sentiment'] = sentiments + ['neutral'] * (len(df) - len(sentiments))
                    leaders[leader] = df
            except Exception as e:
                pass  # Silently skip failed loads
    
    return leaders


def create_leader_comparison(leaders_data):
    """Create leader sentiment comparison using pre-analyzed data"""
    try:
        leader_stats = []
        
        for leader, df in leaders_data.items():
            if 'sentiment' in df.columns:
                total = len(df)
                positive = len(df[df['sentiment'] == 'positive'])
                negative = len(df[df['sentiment'] == 'negative'])
                neutral = len(df[df['sentiment'] == 'neutral'])
                
                sentiment_index = ((positive - negative) / total) if total > 0 else 0
                
                leader_stats.append({
                    'Leader': leader,
                    'Total Tweets': total,
                    'Positive': positive,
                    'Negative': negative,
                    'Neutral': neutral,
                    'Positive %': (positive / total * 100) if total > 0 else 0,
                    'Negative %': (negative / total * 100) if total > 0 else 0,
                    'Neutral %': (neutral / total * 100) if total > 0 else 0,
                    'Sentiment Index': sentiment_index
                })
        
        if not leader_stats:
            return None, None
        
        stats_df = pd.DataFrame(leader_stats)
        
        # Create comparison bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Positive',
            x=stats_df['Leader'],
            y=stats_df['Positive %'],
            marker_color='#2ecc71'
        ))
        
        fig.add_trace(go.Bar(
            name='Neutral',
            x=stats_df['Leader'],
            y=stats_df['Neutral %'],
            marker_color='#95a5a6'
        ))
        
        fig.add_trace(go.Bar(
            name='Negative',
            x=stats_df['Leader'],
            y=stats_df['Negative %'],
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            barmode='group',
            title="Leader Sentiment Comparison",
            xaxis_title="Political Leader",
            yaxis_title="Percentage (%)",
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig, stats_df
    except Exception as e:
        st.error(f"Error creating leader comparison: {str(e)}")
        return None, None


def create_india_heatmap(state_summary):
    """Create India map heatmap showing state-wise sentiment"""
    try:
        if state_summary.empty:
            return None
        
        # Prepare data
        map_data = state_summary.reset_index()
        map_data['hover_text'] = (
            map_data['state'] + '<br>' +
            'Sentiment Index: ' + map_data['sentiment_index'].round(3).astype(str) + '<br>' +
            'Total Tweets: ' + map_data['total_tweets'].astype(str) + '<br>' +
            'Positive: ' + map_data['positive_pct'].round(1).astype(str) + '%<br>' +
            'Negative: ' + map_data['negative_pct'].round(1).astype(str) + '%'
        )
        
        # Get actual data range for better color mapping
        min_sentiment = map_data['sentiment_index'].min()
        max_sentiment = map_data['sentiment_index'].max()
        
        # Expand range slightly for better visualization
        range_min = max(min_sentiment - 0.05, -1)
        range_max = min(max_sentiment + 0.05, 1)
        
        # Create a gradient color scale for positive sentiment
        # Light green (less positive) -> Dark green (more positive)
        custom_colorscale = [
            [0.0, '#d4efdf'],    # Very light green
            [0.2, '#a9dfbf'],    # Light green
            [0.4, '#7dcea0'],    # Medium-light green
            [0.6, '#52be80'],    # Medium green
            [0.8, '#27ae60'],    # Dark green
            [1.0, '#1e8449']     # Very dark green
        ]
        
        # Create choropleth map for India
        fig = px.choropleth(
            map_data,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations='state',
            color='sentiment_index',
            color_continuous_scale=custom_colorscale,
            range_color=[range_min, range_max],  # Use actual data range for better contrast
            hover_data={'state': True, 'sentiment_index': ':.3f', 'total_tweets': ':,'},
            labels={'sentiment_index': 'Sentiment Index'},
            title='State-wise Sentiment Heatmap of India'
        )
        
        fig.update_geos(
            visible=False,
            fitbounds="locations",
            projection_type="mercator"
        )
        
        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(
                font=dict(color='#1a202c', size=18, family='Arial, sans-serif'),
                x=0.5,
                xanchor='center'
            ),
            geo=dict(
                bgcolor='rgba(0,0,0,0)',
                lakecolor='#e8f4f8',
                landcolor='#f5f5f5'
            ),
            coloraxis_colorbar=dict(
                title="Sentiment Index",
                tickformat='.2f'
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating India heatmap: {str(e)}")
        return None


def main():
    """Main application"""
    
    # Professional Header with Icon and Subtitle
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='margin-bottom: 0;'>🗳️ Election Sentiment Analysis</h1>
            <p style='font-size: 1.3rem; color: #667eea; font-weight: 600; margin-top: 0.5rem;'>
                Real-time Twitter Sentiment Analysis on Indian Elections
            </p>
            <p style='font-size: 0.95rem; color: #718096; margin-top: 0.5rem;'>
                Powered by AI/ML • Analyzing Public Opinion Across 28 States & 8 Union Territories
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data FIRST before sidebar with loading indicator
    with st.spinner('🔄 Loading sentiment data... This may take a moment...'):
        df = load_data()
        state_summary = load_state_summary()
    
    if df.empty:
        st.error("⚠️ No data available. Please run the analysis pipeline first.")
        st.info("Run the following command to generate data:")
        st.code("py main.py --data-source sample --num-samples 75000", language="bash")
        return
    
    # Show data loaded successfully with comprehensive info
    total_loaded = len(df)
    unique_states_loaded = df['state'].nunique() if 'state' in df.columns else 0
    st.success(f"✅ Analyzing {total_loaded:,} tweets across {unique_states_loaded} states • Dashboard ready!")
    
    # Define regions for filtering
    regions = {
        'All India': None,
        'North': ['Delhi', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 
                 'Ladakh', 'Punjab', 'Rajasthan', 'Uttar Pradesh', 'Uttarakhand', 'Chandigarh'],
        'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 
                 'Telangana', 'Puducherry', 'Lakshadweep', 'Andaman and Nicobar Islands'],
        'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal'],
        'West': ['Goa', 'Gujarat', 'Maharashtra', 'Dadra and Nagar Haveli and Daman and Diu'],
        'Northeast': ['Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 
                     'Mizoram', 'Nagaland', 'Sikkim', 'Tripura'],
        'Central': ['Chhattisgarh', 'Madhya Pradesh']
    }
    
    # Sidebar
    with st.sidebar:
        # Indian Flag and Logo
        st.markdown("""
            <div style='text-align: center; padding: 10px 0 20px 0;'>
                <img src='https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg' width='120' style='border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);'/>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; color: white; font-weight: 700;'>⚙️ Controls</h2>", unsafe_allow_html=True)
        
        # Load data button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Region filter
        st.subheader("📍 Geographic Filters")
        
        selected_region = st.selectbox(
            "Select Region",
            options=list(regions.keys()),
            index=0
        )
        
        # State filter
        if selected_region == 'All India':
            available_states = ['All States'] + sorted(INDIAN_STATES)
        else:
            available_states = ['All States'] + sorted(regions[selected_region])
        
        selected_state = st.selectbox(
            "Select State",
            options=available_states,
            index=0
        )
        
        st.markdown("---")
        st.subheader("📊 Analysis Options")
        
        show_trends = st.checkbox("Show Trends", value=True)
        show_engagement = st.checkbox("Show Engagement Analysis", value=True)
        show_comparison = st.checkbox("Show State Comparison", value=True)
    
    # Filter data based on sidebar selection
    filtered_df = df.copy()
    if selected_region != 'All India' and regions[selected_region]:
        filtered_df = filtered_df[filtered_df['state'].isin(regions[selected_region])]
    
    if selected_state != 'All States':
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    
    # Calculate metrics
    total_tweets = len(filtered_df)
    
    if 'ensemble_sentiment' in filtered_df.columns:
        sentiment_counts = filtered_df['ensemble_sentiment'].value_counts()
        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        neutral_count = sentiment_counts.get('neutral', 0)
        
        positive_pct = (positive_count / total_tweets * 100) if total_tweets > 0 else 0
        negative_pct = (negative_count / total_tweets * 100) if total_tweets > 0 else 0
        neutral_pct = (neutral_count / total_tweets * 100) if total_tweets > 0 else 0
        
        sentiment_index = ((positive_pct - negative_pct) / 100)
    else:
        positive_pct = negative_pct = neutral_pct = 0
        sentiment_index = 0
    
    unique_states = filtered_df['state'].nunique() if 'state' in filtered_df.columns else 0
    
    # Add Quick Stats to sidebar now that we have the data
    with st.sidebar:        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Tweets", f"{total_tweets:,}", help="Number of tweets analyzed")
        st.metric("States Covered", unique_states, help="Number of unique states in analysis")
        st.metric("Sentiment Index", f"{sentiment_index:.3f}", 
                 delta="Score: -1 (negative) to +1 (positive)",
                 help="Overall sentiment score")
        
        # Download section
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        if not filtered_df.empty:
            # CSV Download
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Full Data (CSV)",
                data=csv,
                file_name=f"sentiment_data_{selected_region.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Excel Download with openpyxl
            try:
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name='Sentiment Data', index=False)
                    if not state_summary.empty:
                        state_summary.to_excel(writer, sheet_name='State Summary')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_data,
                    file_name=f"sentiment_report_{selected_region.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.caption("Excel export unavailable")
            
            # Summary Statistics
            summary_stats = f"""
ELECTION SENTIMENT ANALYSIS REPORT
{'='*50}

Region: {selected_region}
State: {selected_state}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

OVERALL STATISTICS
------------------
Total Tweets Analyzed: {total_tweets:,}
States Covered: {unique_states}
Sentiment Index: {sentiment_index:.3f}

SENTIMENT BREAKDOWN
-------------------
Positive: {positive_pct:.2f}% ({positive_count:,} tweets)
Neutral: {neutral_pct:.2f}% ({neutral_count:,} tweets)
Negative: {negative_pct:.2f}% ({negative_count:,} tweets)

Overall Sentiment: {'POSITIVE' if sentiment_index > 0.2 else ('NEGATIVE' if sentiment_index < -0.2 else 'NEUTRAL')}

Generated by Election Sentiment Intelligence System
Data Source: Kaggle
Models: VADER, TextBlob, ML Ensemble
"""
            st.download_button(
                label="📝 Download Summary Report (TXT)",
                data=summary_stats,
                file_name=f"summary_report_{selected_region.replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Information section
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
            <div style='font-size: 0.85rem; color: #e2e8f0;'>
                <p><strong>Data Source:</strong> Kaggle</p>
                <p><strong>Analysis Models:</strong></p>
                <ul style='margin: 5px 0;'>
                    <li>VADER (Social Media)</li>
                    <li>TextBlob (NLP)</li>
                    <li>ML Ensemble</li>
                </ul>
                <p><strong>Coverage:</strong> 28 States + 8 UTs (36 Total)</p>
                <p><strong>Updated:</strong> Real-time</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Display professional summary card at the top
    region_text = selected_region if selected_region != 'All India' else 'India-wide Analysis'
    state_text = f" • {selected_state}" if selected_state != 'All States' else ''
    sentiment_emoji = '🟢' if sentiment_index > 0.2 else ('🔴' if sentiment_index < -0.2 else '🟡')
    sentiment_label = 'Positive' if sentiment_index > 0.2 else ('Negative' if sentiment_index < -0.2 else 'Neutral')
    
    overview_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; color: white; 
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); margin-bottom: 30px;'>
            <div style='text-align: center;'>
                <h2 style='color: white; margin: 0; font-size: 2rem;'>📊 Analysis Overview</h2>
                <p style='font-size: 1.1rem; margin-top: 10px; opacity: 0.95;'>
                    {region_text}{state_text}
                </p>
                <div style='display: flex; justify-content: space-around; margin-top: 20px; flex-wrap: wrap;'>
                    <div style='text-align: center; padding: 10px; min-width: 150px;'>
                        <div style='font-size: 2.5rem; font-weight: 700;'>{total_tweets:,}</div>
                        <div style='font-size: 0.9rem; opacity: 0.9;'>Total Tweets</div>
                    </div>
                    <div style='text-align: center; padding: 10px; min-width: 150px;'>
                        <div style='font-size: 2.5rem; font-weight: 700;'>{unique_states}</div>
                        <div style='font-size: 0.9rem; opacity: 0.9;'>States Covered</div>
                    </div>
                    <div style='text-align: center; padding: 10px; min-width: 150px;'>
                        <div style='font-size: 2.5rem; font-weight: 700;'>{sentiment_index:.3f}</div>
                        <div style='font-size: 0.9rem; opacity: 0.9;'>Sentiment Index</div>
                    </div>
                    <div style='text-align: center; padding: 10px; min-width: 150px;'>
                        <div style='font-size: 2.5rem; font-weight: 700;'>{sentiment_emoji}</div>
                        <div style='font-size: 0.9rem; opacity: 0.9;'>{sentiment_label}</div>
                    </div>
                </div>
            </div>
        </div>
    """
    st.markdown(overview_html, unsafe_allow_html=True)
    
    # Main content area - Enhanced metric cards
    st.markdown("<h2 style='text-align: center; margin-top: 20px;'>📈 Detailed Sentiment Breakdown</h2>", unsafe_allow_html=True)
    
    # Pre-calculate values for metric cards
    unique_text_count = len(filtered_df['text'].unique()) if 'text' in filtered_df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        card1_html = f"""
            <div style='background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                        padding: 25px; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);'>
                <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 10px;'>📊 TOTAL ANALYZED</div>
                <div style='font-size: 2.5rem; font-weight: 700; margin: 10px 0;'>{total_tweets:,}</div>
                <div style='font-size: 0.85rem; opacity: 0.85;'>{unique_states} States • {unique_text_count} Unique</div>
            </div>
        """
        st.markdown(card1_html, unsafe_allow_html=True)
    
    with col2:
        card2_html = f"""
            <div style='background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
                        padding: 25px; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);'>
                <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 10px;'>✅ POSITIVE</div>
                <div style='font-size: 2.5rem; font-weight: 700; margin: 10px 0;'>{positive_pct:.1f}%</div>
                <div style='font-size: 0.85rem; opacity: 0.85;'>{positive_count:,} tweets</div>
            </div>
        """
        st.markdown(card2_html, unsafe_allow_html=True)
    
    with col3:
        card3_html = f"""
            <div style='background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                        padding: 25px; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);'>
                <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 10px;'>⚠️ NEGATIVE</div>
                <div style='font-size: 2.5rem; font-weight: 700; margin: 10px 0;'>{negative_pct:.1f}%</div>
                <div style='font-size: 0.85rem; opacity: 0.85;'>{negative_count:,} tweets</div>
            </div>
        """
        st.markdown(card3_html, unsafe_allow_html=True)
    
    with col4:
        card4_html = f"""
            <div style='background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%); 
                        padding: 25px; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 15px rgba(149, 165, 166, 0.3);'>
                <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 10px;'>⚪ NEUTRAL</div>
                <div style='font-size: 2.5rem; font-weight: 700; margin: 10px 0;'>{neutral_pct:.1f}%</div>
                <div style='font-size: 0.85rem; opacity: 0.85;'>{neutral_count:,} tweets</div>
            </div>
        """
        st.markdown(card4_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sentiment distribution
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Sentiment Overview")
        
        # Pie chart
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [positive_count, neutral_count, negative_count],
            'Percentage': [positive_pct, neutral_pct, negative_pct]
        })
        
        fig_pie = px.pie(
            sentiment_data,
            values='Count',
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#95a5a6',
                'Negative': '#e74c3c'
            },
            hole=0.4
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Sentiment Gauge")
        gauge_fig = create_sentiment_gauge(positive_pct, negative_pct, neutral_pct)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Overall sentiment interpretation
        if sentiment_index > 0.2:
            st.success("🟢 **Overall Sentiment: POSITIVE** - Strong positive sentiment detected")
        elif sentiment_index < -0.2:
            st.error("🔴 **Overall Sentiment: NEGATIVE** - Strong negative sentiment detected")
        else:
            st.info("🟡 **Overall Sentiment: NEUTRAL** - Mixed or balanced sentiment")
    
    st.markdown("---")
    
    # State comparison
    if show_comparison and not state_summary.empty:
        st.subheader("🗺️ State-wise Comparison")
        
        top_n = st.slider("Number of states to display", 5, 25, 10)
        comparison_fig = create_comparison_chart(state_summary, top_n)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # India Heatmap - NEW FEATURE
        st.markdown("---")
        st.subheader("🗾 India Sentiment Heatmap")
        
        st.markdown("""
            <div style='background: #e8f4f8; padding: 12px; border-radius: 8px; margin-bottom: 15px;'>
                <p style='margin: 0; color: #2c3e50; font-size: 0.9rem;'>
                    <strong>🗺️ Geographic Visualization:</strong> Interactive map showing sentiment intensity 
                    across all Indian states. Greener states indicate positive sentiment, while redder states 
                    show negative sentiment.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        heatmap_fig = create_india_heatmap(state_summary)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("Heatmap visualization unavailable for current selection")
    
    # Trends
    if show_trends:
        st.markdown("---")
        st.subheader("📈 Sentiment Trends Over Time")
        trend_fig = create_trend_chart(filtered_df, selected_state)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
    
    # Engagement analysis
    if show_engagement:
        st.markdown("---")
        st.subheader("💬 Engagement Analysis")
        engagement_fig = create_engagement_analysis(filtered_df, selected_state)
        st.plotly_chart(engagement_fig, use_container_width=True)
    
    # Word Cloud Analysis - NEW FEATURE
    st.markdown("---")
    st.subheader("☁️ Word Cloud Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Select sentiment for word cloud:**")
        wordcloud_sentiment = st.selectbox(
            "Sentiment Filter",
            options=['All', 'positive', 'negative', 'neutral'],
            key='wordcloud_sentiment'
        )
    
    with col2:
        max_words = st.slider("Maximum words to display", 50, 200, 100, step=25, key='max_words')
    
    # Generate word cloud with loading spinner
    sentiment_filter = None if wordcloud_sentiment == 'All' else wordcloud_sentiment
    
    with st.spinner('Generating word cloud...'):
        wordcloud_fig = create_wordcloud_fig(filtered_df, sentiment_filter, max_words)
    
    if wordcloud_fig:
        st.pyplot(wordcloud_fig)
        plt.close(wordcloud_fig)  # Clean up matplotlib figure
    else:
        st.info("No data available for word cloud generation")
    
    # Hashtag Analysis - NEW FEATURE
    st.markdown("---")
    st.subheader("🏷️ Trending Hashtags")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hashtag_sentiment = st.selectbox(
            "Filter by sentiment",
            options=['All', 'positive', 'negative', 'neutral'],
            key='hashtag_sentiment'
        )
    
    with col2:
        top_n_hashtags = st.slider("Number of top hashtags", 10, 30, 20, step=5, key='top_hashtags')
    
    # Generate hashtag analysis
    hashtag_filter = None if hashtag_sentiment == 'All' else hashtag_sentiment
    
    with st.spinner('Analyzing hashtags...'):
        hashtag_fig = analyze_hashtags(filtered_df, hashtag_filter, top_n_hashtags)
    
    if hashtag_fig:
        st.plotly_chart(hashtag_fig, use_container_width=True)
    else:
        st.info("No hashtags found in the selected data")
    
    # Leader Comparison - NEW FEATURE
    st.markdown("---")
    st.subheader("👥 Political Leader Analysis")
    
    # Custom CSS to make expander text more visible
    st.markdown("""
        <style>
        .streamlit-expanderHeader {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #1f77b4 !important;
        }
        div[data-testid="stExpander"] details summary p {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #1f77b4 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 View Leader Sentiment Comparison", expanded=False):
        with st.spinner('🔄 Loading and analyzing leader data...'):
            leaders_data = load_leader_data()
        
        if leaders_data:
            total_leader_tweets = sum(len(df) for df in leaders_data.values())
            
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 12px; margin-bottom: 20px;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    <p style='margin: 0; color: white; font-size: 1.05rem; line-height: 1.6;'>
                        <strong>📊 Leader Sentiment Analysis:</strong> Analyzing sentiment across tweets about 
                        <strong>Narendra Modi</strong>, <strong>Rahul Gandhi</strong>, and <strong>Arvind Kejriwal</strong>
                        <br><br>
                        <span style='font-size: 0.95rem; opacity: 0.95;'>
                            Based on <strong>{total_leader_tweets:,}</strong> tweets sampled from dedicated leader datasets
                        </span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            leader_fig, leader_stats = create_leader_comparison(leaders_data)
            
            if leader_fig and leader_stats is not None:
                st.plotly_chart(leader_fig, use_container_width=True)
                
                st.markdown("### 📋 Detailed Leader Statistics")
                
                # Format the stats dataframe for display
                display_stats = leader_stats.copy()
                display_stats['Positive %'] = display_stats['Positive %'].apply(lambda x: f"{x:.2f}%")
                display_stats['Negative %'] = display_stats['Negative %'].apply(lambda x: f"{x:.2f}%")
                display_stats['Neutral %'] = display_stats['Neutral %'].apply(lambda x: f"{x:.2f}%")
                display_stats['Sentiment Index'] = display_stats['Sentiment Index'].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(
                    display_stats,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Leader": st.column_config.TextColumn("Leader", width="medium"),
                        "Total Tweets": st.column_config.NumberColumn("Total Tweets", format="%d"),
                        "Sentiment Index": st.column_config.TextColumn("Sentiment Index", width="small")
                    }
                )
                
                # Add insights
                best_leader = leader_stats.loc[leader_stats['Sentiment Index'].idxmax(), 'Leader']
                best_index = leader_stats['Sentiment Index'].max()
                
                st.success(f"🌟 **Most Positive Sentiment**: {best_leader} (Index: {best_index:.3f})")
            else:
                st.warning("Leader data is available but could not be processed with current filters")
        else:
            st.info("Leader-specific data files not found in data/raw/ directory")
    
    # Top performing tweets
    st.markdown("---")
    st.subheader("🔝 Top Performing Tweets")
    
    # Add slider to control number of tweets displayed
    num_tweets = st.slider("Number of tweets to display", min_value=10, max_value=100, value=25, step=5)
    
    tab1, tab2, tab3 = st.tabs(["Most Liked", "Most Retweeted", "Most Replied"])
    
    with tab1:
        top_liked = filtered_df.nlargest(num_tweets, 'like_count')[
            ['text', 'state', 'ensemble_sentiment', 'like_count', 'retweet_count']
        ]
        st.dataframe(top_liked, use_container_width=True, hide_index=True, height=600)
    
    with tab2:
        top_retweeted = filtered_df.nlargest(num_tweets, 'retweet_count')[
            ['text', 'state', 'ensemble_sentiment', 'retweet_count', 'like_count']
        ]
        st.dataframe(top_retweeted, use_container_width=True, hide_index=True, height=600)
    
    with tab3:
        top_replied = filtered_df.nlargest(num_tweets, 'reply_count')[
            ['text', 'state', 'ensemble_sentiment', 'reply_count', 'like_count']
        ]
        st.dataframe(top_replied, use_container_width=True, hide_index=True, height=600)
    
    # State details table
    if selected_state == 'All States':
        st.markdown("---")
        st.subheader("📋 Detailed State Statistics")
        
        if not state_summary.empty:
            display_summary = state_summary.copy()
            display_summary = display_summary.reset_index()
            
            # Select only the columns we want to display
            cols_to_display = ['state', 'total_tweets', 'positive_pct', 'negative_pct', 
                              'neutral_pct', 'sentiment_index', 'overall_sentiment']
            display_summary = display_summary[cols_to_display]
            display_summary.columns = ['State', 'Total Tweets', 'Positive %', 'Negative %', 
                                       'Neutral %', 'Sentiment Index', 'Overall Sentiment']
            
            # Format percentages
            for col in ['Positive %', 'Negative %', 'Neutral %']:
                display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2f}%")
            
            display_summary['Sentiment Index'] = display_summary['Sentiment Index'].apply(
                lambda x: f"{x:.3f}"
            )
            
            st.dataframe(
                display_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "State": st.column_config.TextColumn("State", width="medium"),
                    "Total Tweets": st.column_config.NumberColumn("Total Tweets", format="%d"),
                    "Overall Sentiment": st.column_config.TextColumn("Overall Sentiment", width="small")
                }
            )
    
    # Professional Footer
    st.markdown("---")
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); 
                    padding: 40px; border-radius: 15px; color: white; margin-top: 40px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
            <div style='text-align: center;'>
                <h3 style='color: white; margin-bottom: 20px; font-size: 1.5rem;'>
                    🗳️ Election Sentiment Analysis Platform
                </h3>
                <p style='font-size: 1rem; margin: 10px 0; opacity: 0.9;'>
                    <strong>Professional AI-Powered Sentiment Analysis for Indian Elections</strong>
                </p>
                <div style='display: flex; justify-content: center; gap: 30px; margin: 25px 0; flex-wrap: wrap;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.8rem; margin-bottom: 5px;'>🤖</div>
                        <div style='font-size: 0.85rem; opacity: 0.8;'>AI/ML Models</div>
                        <div style='font-size: 0.75rem; opacity: 0.7;'>VADER • TextBlob • Ensemble</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.8rem; margin-bottom: 5px;'>🇮🇳</div>
                        <div style='font-size: 0.85rem; opacity: 0.8;'>Pan-India Coverage</div>
                        <div style='font-size: 0.75rem; opacity: 0.7;'>28 States + 8 UTs</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.8rem; margin-bottom: 5px;'>📊</div>
                        <div style='font-size: 0.85rem; opacity: 0.8;'>Real-time Analysis</div>
                        <div style='font-size: 0.75rem; opacity: 0.7;'>Live Dashboard Updates</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.8rem; margin-bottom: 5px;'>🔒</div>
                        <div style='font-size: 0.85rem; opacity: 0.8;'>Secure & Reliable</div>
                        <div style='font-size: 0.75rem; opacity: 0.7;'>Python • Streamlit</div>
                    </div>
                </div>
                <div style='margin-top: 25px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);'>
                    <p style='font-size: 0.8rem; margin: 5px 0; opacity: 0.7;'>
                        © 2026 Election Sentiment Analysis Platform • Built with ❤️ for Democracy
                    </p>
                    <p style='font-size: 0.75rem; margin: 5px 0; opacity: 0.6;'>
                        Powered by Python • Streamlit • Pandas • NLTK • Scikit-learn • Plotly
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
