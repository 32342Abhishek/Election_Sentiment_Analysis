"""
Configuration file for Twitter Sentiment Analysis on Elections
"""
import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Indian States and Union Territories
# 28 States + 8 Union Territories = 36 Total
INDIAN_STATES = [
    # 28 States
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    # 8 Union Territories
    'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu',
    'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
]

# State abbreviations mapping
STATE_ABBREVIATIONS = {
    'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh',
    'AS': 'Assam',
    'BR': 'Bihar',
    'CG': 'Chhattisgarh',
    'GA': 'Goa',
    'GJ': 'Gujarat',
    'HR': 'Haryana',
    'HP': 'Himachal Pradesh',
    'JH': 'Jharkhand',
    'KA': 'Karnataka',
    'KL': 'Kerala',
    'MP': 'Madhya Pradesh',
    'MH': 'Maharashtra',
    'MN': 'Manipur',
    'ML': 'Meghalaya',
    'MZ': 'Mizoram',
    'NL': 'Nagaland',
    'OD': 'Odisha',
    'PB': 'Punjab',
    'RJ': 'Rajasthan',
    'SK': 'Sikkim',
    'TN': 'Tamil Nadu',
    'TS': 'Telangana',
    'TR': 'Tripura',
    'UP': 'Uttar Pradesh',
    'UK': 'Uttarakhand',
    'WB': 'West Bengal',
    'DL': 'Delhi',
    'JK': 'Jammu and Kashmir'
}

# Major cities to state mapping
CITY_TO_STATE = {
    # Andhra Pradesh
    'Hyderabad': 'Telangana',
    'Visakhapatnam': 'Andhra Pradesh',
    'Vijayawada': 'Andhra Pradesh',
    'Guntur': 'Andhra Pradesh',
    'Tirupati': 'Andhra Pradesh',
    
    # Telangana
    'Warangal': 'Telangana',
    'Nizamabad': 'Telangana',
    
    # Karnataka
    'Bangalore': 'Karnataka',
    'Bengaluru': 'Karnataka',
    'Mysore': 'Karnataka',
    'Mangalore': 'Karnataka',
    'Hubli': 'Karnataka',
    
    # Tamil Nadu
    'Chennai': 'Tamil Nadu',
    'Coimbatore': 'Tamil Nadu',
    'Madurai': 'Tamil Nadu',
    'Tiruchirappalli': 'Tamil Nadu',
    'Salem': 'Tamil Nadu',
    
    # Maharashtra
    'Mumbai': 'Maharashtra',
    'Pune': 'Maharashtra',
    'Nagpur': 'Maharashtra',
    'Nashik': 'Maharashtra',
    'Aurangabad': 'Maharashtra',
    
    # Gujarat
    'Ahmedabad': 'Gujarat',
    'Surat': 'Gujarat',
    'Vadodara': 'Gujarat',
    'Rajkot': 'Gujarat',
    
    # Kerala
    'Kochi': 'Kerala',
    'Thiruvananthapuram': 'Kerala',
    'Kozhikode': 'Kerala',
    'Thrissur': 'Kerala',
    
    # West Bengal
    'Kolkata': 'West Bengal',
    'Howrah': 'West Bengal',
    'Durgapur': 'West Bengal',
    
    # Uttar Pradesh
    'Lucknow': 'Uttar Pradesh',
    'Kanpur': 'Uttar Pradesh',
    'Varanasi': 'Uttar Pradesh',
    'Agra': 'Uttar Pradesh',
    'Meerut': 'Uttar Pradesh',
    'Noida': 'Uttar Pradesh',
    'Ghaziabad': 'Uttar Pradesh',
    
    # Madhya Pradesh
    'Bhopal': 'Madhya Pradesh',
    'Indore': 'Madhya Pradesh',
    'Gwalior': 'Madhya Pradesh',
    'Jabalpur': 'Madhya Pradesh',
    
    # Rajasthan
    'Jaipur': 'Rajasthan',
    'Jodhpur': 'Rajasthan',
    'Udaipur': 'Rajasthan',
    'Kota': 'Rajasthan',
    
    # Delhi
    'New Delhi': 'Delhi',
    'Delhi': 'Delhi',
    
    # Punjab
    'Chandigarh': 'Punjab',
    'Ludhiana': 'Punjab',
    'Amritsar': 'Punjab',
    'Jalandhar': 'Punjab',
    
    # Haryana
    'Gurugram': 'Haryana',
    'Gurgaon': 'Haryana',
    'Faridabad': 'Haryana',
    
    # Bihar
    'Patna': 'Bihar',
    'Gaya': 'Bihar',
    'Bhagalpur': 'Bihar',
    
    # Jharkhand
    'Ranchi': 'Jharkhand',
    'Jamshedpur': 'Jharkhand',
    'Dhanbad': 'Jharkhand',
    
    # Odisha
    'Bhubaneswar': 'Odisha',
    'Cuttack': 'Odisha',
    'Rourkela': 'Odisha',
    
    # Assam
    'Guwahati': 'Assam',
    'Dispur': 'Assam',
    
    # Chhattisgarh
    'Raipur': 'Chhattisgarh',
    'Bhilai': 'Chhattisgarh',
    
    # Uttarakhand
    'Dehradun': 'Uttarakhand',
    'Haridwar': 'Uttarakhand',
    
    # Himachal Pradesh
    'Shimla': 'Himachal Pradesh',
    
    # Goa
    'Panaji': 'Goa',
}

# Sentiment Analysis Parameters
SENTIMENT_THRESHOLD = {
    'positive': 0.05,
    'negative': -0.05
}

# Election Keywords
ELECTION_KEYWORDS = [
    'election', 'vote', 'voting', 'poll', 'ballot', 'campaign',
    'candidate', 'party', 'BJP', 'Congress', 'AAP', 'TMC',
    'DMK', 'AIADMK', 'TRS', 'BRS', 'SP', 'BSP', 'JDU', 'RJD',
    'PM', 'CM', 'minister', 'parliament', 'assembly', 'MP', 'MLA',
    'governance', 'democracy', 'manifesto', 'rally', 'nomination'
]

# Text Preprocessing Parameters
STOP_WORDS_LANGUAGES = ['english', 'hindi']
MIN_TWEET_LENGTH = 10
MAX_TWEET_LENGTH = 500

# Twitter API Configuration (placeholder - use environment variables)
TWITTER_API_CONFIG = {
    'api_key': os.getenv('TWITTER_API_KEY', ''),
    'api_secret': os.getenv('TWITTER_API_SECRET', ''),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN', ''),
    'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET', ''),
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN', '')
}

# Model Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
