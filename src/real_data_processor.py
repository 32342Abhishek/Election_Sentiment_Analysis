"""
Real Data Processor Module
Handles loading and combining real Twitter datasets from multiple sources
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_csv_safely(file_path, encoding='utf-8'):
    """
    Safely load CSV file with multiple encoding attempts
    
    Args:
        file_path: Path to CSV file
        encoding: Initial encoding to try
        
    Returns:
        DataFrame or None
    """
    encodings = [encoding, 'latin-1', 'iso-8859-1', 'cp1252']
    
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            print(f"✓ Loaded {file_path.name} with {enc} encoding ({len(df)} rows)")
            return df
        except Exception as e:
            continue
    
    print(f"✗ Failed to load {file_path.name}")
    return None


def standardize_columns(df, dataset_name=""):
    """
    Standardize column names across different datasets
    
    Args:
        df: Input dataframe
        dataset_name: Name of dataset for logging
        
    Returns:
        Dataframe with standardized columns
    """
    # Common text column names
    text_columns = ['text', 'tweet', 'Tweet', 'content', 'Content', 'full_text', 'message']
    
    # Find text column
    text_col = None
    for col in text_columns:
        if col in df.columns:
            text_col = col
            break
    
    if text_col and text_col != 'text':
        df['text'] = df[text_col]
        print(f"  └─ Renamed '{text_col}' to 'text' in {dataset_name}")
    
    # Standardize other common columns
    column_mappings = {
        'created_at': ['timestamp', 'date', 'created', 'time'],
        'user_id': ['user', 'username', 'author'],
        'location': ['user_location', 'place', 'geo'],
        'retweet_count': ['retweets', 'rt_count'],
        'like_count': ['likes', 'favorite_count', 'favorites']
    }
    
    for standard_col, alternatives in column_mappings.items():
        if standard_col not in df.columns:
            for alt_col in alternatives:
                if alt_col in df.columns:
                    df[standard_col] = df[alt_col]
                    break
    
    return df


def combine_all_datasets(data_dir, output_path=None):
    """
    Combine all CSV datasets from a directory
    
    Args:
        data_dir: Directory containing CSV files
        output_path: Optional path to save combined dataset
        
    Returns:
        Combined dataframe
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return pd.DataFrame()
    
    # Find all CSV files
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame()
    
    print(f"\nFound {len(csv_files)} CSV files")
    print("=" * 60)
    
    # Load and combine datasets
    all_dataframes = []
    
    for csv_file in csv_files:
        df = load_csv_safely(csv_file)
        
        if df is not None and not df.empty:
            # Standardize columns
            df = standardize_columns(df, csv_file.name)
            
            # Add source column
            df['data_source'] = csv_file.stem
            
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("No valid dataframes to combine")
        return pd.DataFrame()
    
    # Combine all dataframes
    print("\n" + "=" * 60)
    print("Combining datasets...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\n✓ Combined dataset created:")
    print(f"  • Total tweets: {len(combined_df):,}")
    print(f"  • Columns: {len(combined_df.columns)}")
    print(f"  • Data sources: {combined_df['data_source'].nunique()}")
    
    # Save if output path provided
    if output_path:
        save_real_data(combined_df, output_path)
    
    return combined_df


def save_real_data(df, output_path):
    """
    Save dataframe to CSV file
    
    Args:
        df: Dataframe to save
        output_path: Output file path
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ Saved combined dataset to: {output_file}")
        print(f"  • File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"Error saving file: {str(e)}")


def filter_by_date_range(df, start_date=None, end_date=None, date_column='created_at'):
    """
    Filter dataframe by date range
    
    Args:
        df: Input dataframe
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        date_column: Name of date column
        
    Returns:
        Filtered dataframe
    """
    if date_column not in df.columns:
        print(f"Warning: Column '{date_column}' not found")
        return df
    
    df = df.copy()
    
    try:
        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Filter by date range
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df[date_column] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df[date_column] <= end_date]
        
        print(f"Filtered to {len(df):,} tweets in date range")
    except Exception as e:
        print(f"Error filtering by date: {str(e)}")
    
    return df


def sample_large_dataset(df, sample_size=100000, random_state=42):
    """
    Sample from large dataset
    
    Args:
        df: Input dataframe
        sample_size: Number of samples to take
        random_state: Random seed
        
    Returns:
        Sampled dataframe
    """
    if len(df) <= sample_size:
        return df
    
    print(f"Sampling {sample_size:,} tweets from {len(df):,} total tweets")
    sampled_df = df.sample(n=sample_size, random_state=random_state)
    
    return sampled_df
