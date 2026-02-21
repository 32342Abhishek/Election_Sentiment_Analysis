# 🗳️ Election Sentiment Intelligence System

**An advanced AI-powered Election Sentiment Intelligence System designed to analyze public opinion related to Indian elections using social media data.**

This professional platform supports researchers, analysts, policymakers, and campaign strategists in understanding public opinion trends through comprehensive sentiment analysis across all 36 Indian states and union territories.

## 🌟 System Capabilities

### **Core Intelligence Features**

#### 1. **Advanced Sentiment Analysis** 🤖
- Multi-model ensemble approach (VADER + TextBlob + ML)
- Classify tweets into Positive, Negative, and Neutral categories
- Calculate sentiment distribution percentages
- Compute national and state-level sentiment scores
- Sentiment Index: -1.0 (highly negative) to +1.0 (highly positive)

#### 2. **Comprehensive State-wise Analysis** 🗺️
- Aggregate sentiment for all 36 Indian states and UTs
- Generate sentiment index scores for each state
- Identify top positive and top negative states
- Detect swing states with changing sentiment patterns
- Regional analysis (North, South, East, West, Northeast, Central)

#### 3. **Time Series & Trend Analysis** 📈
- Analyze sentiment changes over time
- Detect peaks and sudden shifts in public opinion
- Visualize historical trends and patterns
- Identify critical time periods and events
- Daily/weekly sentiment tracking

#### 4. **Interactive Web Dashboard** 💻
- Real-time sentiment visualization
- Regional and state-level filtering
- Interactive charts and gauges
- Engagement metrics tracking
- Top performing tweets analysis
- Export capabilities

#### 5. **Data Coverage** 📊
- **294,170 analyzed tweets** from Kaggle datasets
- **100% state coverage** across all 36 states/UTs
- **Multiple data sources**: Political leaders, elections, public sentiment
- **Geographic accuracy**: 100+ city-to-state mappings

## ✨ NEW: Enhanced Features (v2.0)

### **Visual Intelligence** ☁️
- **Word Cloud Analysis** - Visualize trending keywords by sentiment
- **Hashtag Tracking** - Monitor viral hashtags and campaign messaging
- **India Heatmap** - Interactive geographic sentiment visualization
- **Customizable Views** - Filter by sentiment, adjust word counts, select regions

### **Political Leader Analysis** 👥
- **Direct Comparison** - Compare sentiment across Modi, Rahul Gandhi, Kejriwal
- **70,000 tweets per leader** - Comprehensive politician-level analysis
- **Sentiment Metrics** - Track individual leader popularity and perception
- **Visual Charts** - Side-by-side grouped bar chart comparisons

### **Professional Export & Reporting** 📥
- **Multiple Formats** - CSV, Excel, and Text summary reports
- **Excel Reports** - Multi-sheet workbooks with data + state summaries
- **Executive Summaries** - One-page text reports with key insights
- **Smart Naming** - Auto-generated descriptive filenames

### **Enhanced User Experience** 🎨
- **Expandable Sections** - Clean, organized interface with collapsible panels
- **Interactive Controls** - Sliders, dropdowns, and filters for customization
- **Information Cards** - Contextual help and explanations throughout
- **Professional Visuals** - Publication-ready charts and visualizations

> 📖 **See [ENHANCEMENTS.md](ENHANCEMENTS.md) for complete feature documentation**

### **Professional Web Interface**
- 📊 **Interactive Dashboard** - Real-time sentiment analysis with Streamlit
- 🗺️ **Regional & State Filters** - Select specific states or regions to analyze
- 📈 **Dynamic Visualizations** - Interactive charts, gauges, and trend analysis
- 💬 **Engagement Metrics** - Track likes, retweets, and replies by sentiment
- 📋 **Detailed Statistics** - Comprehensive state-wise breakdown

### **Advanced Analytics**
- 🤖 **Multi-Model Sentiment Analysis** 
  - VADER (Social media optimized)
  - TextBlob (Polarity & Subjectivity)
  - Machine Learning Ensemble (Logistic Regression, Random Forest, Naive Bayes)
- 🎯 **Sentiment Index Calculation** - Quantitative measure of public sentiment
- 📊 **Regional Comparisons** - North, South, East, West, Northeast, Central
- ⏱️ **Trend Analysis** - Sentiment changes over time

### **Comprehensive Coverage**
- 🇮🇳 **All 36 Indian States & UTs** - Complete geographic coverage
- 📍 **100+ City Mappings** - Accurate state extraction from location data
- 🔢 **Large-Scale Analysis** - Process 720,000+ tweets (20,000 per state minimum)
- 🌐 **Multi-Source Data** - Twitter API, Kaggle datasets, CSV files, or sample data

## 🚀 Quick Start

### **Prerequisites**
```bash
Python 3.8 or higher
pip (Python package installer)
```

### **Installation**

1. **Clone or download the project**
```bash
cd Learnathone
```

2. **Install dependencies**
```bash
py -m pip install -r requirements.txt
```

3. **Download NLTK data**
```bash
py -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger omw-1.4
```

### **Run the Application**

#### **Step 1: Generate Analysis Data**
```bash
# Generate 75,000 sample tweets (default)
py main.py --data-source sample --num-samples 75000

# Or generate 100,000 tweets for more comprehensive analysis
py main.py --data-source sample --num-samples 100000
```

#### **Step 2: Launch Web Application**

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
chmod +x run_app.sh
./run_app.sh
```

**Or manually:**
```bash
streamlit run app.py
```

The web application will automatically open in your browser at `http://localhost:8501`

## 📊 Intelligence & Insights Provided

The Election Sentiment Intelligence System delivers comprehensive analysis across multiple dimensions:

### **1. Sentiment Classification & Scoring** ✅
- Positive, Negative, and Neutral tweet classification
- Overall sentiment index calculation (-1 to +1 scale)
- Confidence scores for each prediction
- Multi-model ensemble for accuracy

### **2. Geographic Intelligence** ✅
- State-wise sentiment aggregation
- Regional comparison (6 regions across India)
- Identification of high-positive and high-negative states
- City-level location mapping to states

### **3. Engagement Analysis** ✅
- Average likes, retweets, and replies by sentiment
- Top performing tweets identification
- Viral content tracking
- User interaction patterns

### **4. Temporal Analysis** ✅
- Sentiment trends over time
- Historical pattern identification
- Peak detection and event correlation
- Time-series visualization

### **5. Comparative Analysis** ✅
- State-to-state sentiment comparison
- Regional performance benchmarking
- Leader-wise data availability (Modi, Rahul Gandhi, Kejriwal)
- Engagement metrics comparison

### **Future Enhancements** 🚀
See [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md) for planned features:
- Predictive election outcome modeling
- Anomaly and crisis detection
- Automated strategic recommendations
- Topic modeling and trending issues
- PDF report generation

---

## 📊 Using the Dashboard

### **Navigation**

1. **Sidebar Controls**
   - 🔄 **Refresh Data** - Reload latest analysis
   - 📍 **Region Filter** - Select All India, North, South, East, West, Northeast, or Central
   - 🗺️ **State Filter** - Choose specific state or all states
   - ☑️ **Analysis Options** - Toggle trends, engagement, and comparison views

2. **Main Dashboard**
   - **Overview Metrics** - Total tweets, sentiment percentages, sentiment index
   - **Sentiment Gauge** - Visual indicator of overall sentiment (-100 to +100)
   - **Pie Chart** - Distribution of positive, neutral, and negative tweets
   - **State Comparison** - Bar chart comparing top states by tweet volume
   - **Trend Analysis** - Sentiment changes over time
   - **Engagement Analysis** - Average likes, retweets, and replies by sentiment
   - **Top Tweets** - Most liked, retweeted, and replied tweets

3. **Interactive Features**
   - Hover over charts for detailed information
   - Click legend items to show/hide data series
   - Zoom and pan on charts
   - Sort and filter tables
   - Export visualizations

## 🏗️ Project Structure

```
Learnathone/
├── app.py                      # Streamlit web application (MAIN INTERFACE)
├── main.py                     # Analysis pipeline
├── run_app.bat                 # Windows launcher
├── run_app.sh                  # Linux/Mac launcher
├── requirements.txt            # Python dependencies
├── config/
│   ├── __init__.py
│   └── config.py              # All 36 states, regions, cities
├── src/
│   ├── __init__.py
│   ├── data_collection.py     # Twitter API, sample data generation
│   ├── data_preprocessing.py  # Text cleaning, tokenization
│   ├── state_extraction.py    # Geographic identification
│   ├── sentiment_analysis.py  # VADER, TextBlob, ML models
│   ├── state_aggregation.py   # Regional analysis
│   └── visualization.py       # Chart generation (legacy)
├── data/
│   ├── raw/                   # Raw tweet data
│   └── processed/             # Cleaned and analyzed data
├── outputs/
│   ├── *.csv                  # Analysis results
│   └── visualizations/        # Generated charts (legacy)
└── models/                    # Trained ML models

```

## 📖 Detailed Usage

### **Data Sources**

#### **Current Implementation: Kaggle Datasets**
The system uses real Twitter data from multiple Kaggle sources:

```bash
# Process real data from 6 Kaggle datasets (294,170 tweets)
py main.py --data-source real --num-samples 500000
```

**Dataset Sources**:
- LokSabha Election 2024 Tweets (1,000 tweets)
- Assembly Elections 2022 (43,000 tweets)
- General Election Tweets (50,000 tweets)
- Narendra Modi Political Data (70,000 tweets)
- Rahul Gandhi Political Data (70,000 tweets)
- Arvind Kejriwal Political Data (70,000 tweets)

**Total**: 304,271 raw tweets → 294,170 after cleaning and deduplication

#### **Alternative Data Sources**

##### **1. Sample Data (For Testing)**
```bash
py main.py --data-source sample --num-samples 75000
```
Generates realistic election tweets with locations, timestamps, and engagement metrics.

#### **2. Twitter API (Real Data)**
Set up environment variables:
```bash
TWITTER_BEARER_TOKEN=your_token_here
```
Then run:
```bash
py main.py --data-source twitter_api
```

#### **3. CSV File**
```bash
py main.py --data-source csv --csv-path "path/to/tweets.csv"
```

#### **4. Kaggle Dataset**
Configure `kaggle.json` and run:
```bash
py main.py --data-source kaggle
```

### **Understanding the Results**

#### **Sentiment Index**
- Range: -1.0 to +1.0
- Calculation: `(% Positive - % Negative) / 100`
- **> 0.2**: Strong positive sentiment
- **-0.2 to 0.2**: Neutral/mixed sentiment
- **< -0.2**: Strong negative sentiment

#### **Regional Grouping**
- **North**: Delhi, Haryana, HP, J&K, Ladakh, Punjab, Rajasthan, UP, Uttarakhand, Chandigarh
- **South**: AP, Karnataka, Kerala, TN, Telangana, Puducherry, Lakshadweep, A&N Islands
- **East**: Bihar, Jharkhand, Odisha, West Bengal
- **West**: Goa, Gujarat, Maharashtra, Dadra & Nagar Haveli and Daman & Diu
- **Northeast**: Arunachal, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura
- **Central**: Chhattisgarh, Madhya Pradesh

## 🔧 Advanced Configuration

### **Customize Analysis**

Edit `config/config.py` to:
- Add more cities to `CITY_TO_STATE` mapping
- Modify election keywords
- Change sentiment thresholds
- Update regional groupings

### **Adjust Sample Size**

For different analysis scales:
```bash
# Quick test (5,000 tweets)
py main.py --data-source sample --num-samples 5000

# Standard analysis (75,000 tweets)
py main.py --data-source sample --num-samples 75000

# Comprehensive analysis (100,000 tweets)
py main.py --data-source sample --num-samples 100000
```

## 📈 Output Files

After running the analysis, you'll find:

- **data/raw/sample_election_tweets.csv** - Original tweet data
- **data/processed/preprocessed_tweets.csv** - Cleaned tweets
- **data/processed/tweets_with_sentiment.csv** - Tweets with sentiment scores
- **outputs/state_sentiment_summary.csv** - State-wise statistics
- **outputs/state_sentiment_aggregation.csv** - Detailed aggregation
- **outputs/analysis_insights.txt** - Key findings and interpretation

## 🛠️ Technologies Used

- **Python 3.12+** - Core language
- **Streamlit** - Web application framework
- **Pandas & NumPy** - Data processing
- **NLTK & TextBlob** - Natural language processing
- **VADER** - Social media sentiment analysis
- **Scikit-learn** - Machine learning models
- **Plotly** - Interactive visualizations
- **Tweepy** - Twitter API integration

## 🎯 Key Capabilities

✅ **294,170 real tweets analyzed** (Kaggle data sources)  
✅ **Real-time interactive web dashboard**  
✅ **State-wise sentiment intelligence** (36/36 covered)  
✅ **Multi-model AI sentiment analysis** (VADER + TextBlob + ML Ensemble)  
✅ **Regional filtering and comparison**  
✅ **Engagement metrics tracking** (likes, retweets, replies)  
✅ **Trend analysis over time**  
✅ **Professional Plotly visualizations**  
✅ **Sentiment gauge and distribution charts**  
✅ **Top tweets by engagement**  
✅ **CSV export functionality**  
✅ **Complete India coverage** (28 States + 8 UTs)  

## 📚 Additional Documentation

- **[PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md)** - Detailed technical specification, implementation status, and development roadmap
- **[LICENSE](LICENSE)** - MIT License details  

## 📝 Notes

- **First Run**: Analysis pipeline takes 5-15 minutes for 75,000 tweets (depending on your hardware)
- **Browser**: Works best on Chrome, Firefox, or Edge
- **Port**: Default Streamlit port is 8501 (configurable)
- **Data Refresh**: Click "Refresh Data" in sidebar to reload latest analysis

## 🐛 Troubleshooting

**Issue**: "No data available" error in dashboard  
**Solution**: Run `py main.py` first to generate analysis data

**Issue**: Streamlit not found  
**Solution**: Run `py -m pip install streamlit`

**Issue**: NLTK data missing  
**Solution**: Run `py -m nltk.downloader punkt stopwords wordnet`

**Issue**: Port already in use  
**Solution**: Run `streamlit run app.py --server.port 8502`

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a complete, production-ready application for election sentiment analysis.

---

**Built with ❤️ for analyzing public sentiment on Indian elections**
