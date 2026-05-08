import feedparser
import re
from transformers import pipeline

# Load FinBERT once on startup — takes ~30 seconds first time
print("Loading FinBERT model...")
sentiment_pipeline = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    top_k=1
)
print("FinBERT loaded.")

# Map bank names to search-friendly terms
BANK_SEARCH_NAMES = {
    'HDFC BANK': 'HDFC Bank',
    'ICICI BANK': 'ICICI Bank',
    'STATE BANK OF INDIA': 'SBI Bank',
    'YES BANK': 'Yes Bank',
    'AXIS BANK': 'Axis Bank',
    'KOTAK MAHINDRA BANK': 'Kotak Mahindra Bank',
    'PUNJAB NATIONAL BANK': 'PNB Bank',
    'BANK OF BARODA': 'Bank of Baroda',
    'CANARA BANK': 'Canara Bank',
    'INDUSIND BANK': 'IndusInd Bank',
    'BANDHAN BANK': 'Bandhan Bank',
    'FEDERAL BANK': 'Federal Bank India',
    'RBL BANK': 'RBL Bank',
    'IDFC FIRST BANK': 'IDFC First Bank',
    'AU SMALL FINANCE BANK': 'AU Small Finance Bank',
    'IDBI BANK': 'IDBI Bank',
    'UNION BANK OF INDIA': 'Union Bank India',
    'BANK OF INDIA': 'Bank of India',
    'CENTRAL BANK OF INDIA': 'Central Bank India',
    'UCO BANK': 'UCO Bank',
    'INDIAN OVERSEAS BANK': 'Indian Overseas Bank',
    'INDIAN BANK': 'Indian Bank',
    'BANK OF MAHARASHTRA': 'Bank of Maharashtra',
    'PUNJAB AND SIND BANK': 'Punjab Sind Bank',
    'LAKSHMI VILAS BANK': 'Lakshmi Vilas Bank',
    'KARNATAKA BANK': 'Karnataka Bank',
    'KARUR VYSYA BANK': 'Karur Vysya Bank',
    'SOUTH INDIAN BANK': 'South Indian Bank',
    'CITY UNION BANK': 'City Union Bank',
    'DCB BANK': 'DCB Bank',
    'CSB BANK': 'CSB Bank',
    'DHANLAXMI BANK': 'Dhanlaxmi Bank',
    'TAMILNAD MERCANTILE BANK': 'Tamilnad Mercantile Bank',
    'EQUITAS SMALL FINANCE BANK': 'Equitas Small Finance Bank',
    'UJJIVAN SMALL FINANCE BANK': 'Ujjivan Small Finance Bank',
    'ESAF SMALL FINANCE BANK': 'ESAF Small Finance Bank',
    'SURYODAY SMALL FINANCE BANK': 'Suryoday Small Finance Bank',
    'UTKARSH SMALL FINANCE BANK': 'Utkarsh Small Finance Bank',
}

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:512]  # FinBERT max token limit

def get_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0][0]
        label = result['label'].capitalize()
        confidence = round(result['score'] * 100, 1)
        return label, confidence
    except:
        return 'Neutral', 50.0

def get_news_and_sentiment(bank_name):
    search_name = BANK_SEARCH_NAMES.get(bank_name, bank_name.title())
    
    # Google News RSS — free, no API key
    query = search_name.replace(' ', '+') + '+India+banking'
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        feed = feedparser.parse(url)
        entries = feed.entries[:5]  # top 5 headlines
        
        if not entries:
            return {
                'bank': bank_name,
                'headlines': [],
                'overall_sentiment': 'Neutral',
                'sentiment_score': 0.0,
                'error': 'No news found'
            }
        
        headlines = []
        sentiment_scores = []
        
        for entry in entries:
            title = clean_text(entry.get('title', ''))
            source = entry.get('source', {}).get('title', 'Unknown')
            published = entry.get('published', '')
            
            # Run FinBERT on headline
            sentiment_label, confidence = get_sentiment(title)
            
            # Convert to numeric score: Positive=+1, Neutral=0, Negative=-1
            if sentiment_label == 'Positive':
                score = confidence / 100
            elif sentiment_label == 'Negative':
                score = -(confidence / 100)
            else:
                score = 0.0
            
            sentiment_scores.append(score)
            
            headlines.append({
                'title': title,
                'source': source,
                'published': published,
                'sentiment': sentiment_label,
                'confidence': confidence
            })
        
        # Overall sentiment
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        if avg_score > 0.2:
            overall = 'Positive'
        elif avg_score < -0.2:
            overall = 'Negative'
        else:
            overall = 'Mixed'
        
        return {
            'bank': bank_name,
            'headlines': headlines,
            'overall_sentiment': overall,
            'sentiment_score': round(avg_score, 3),
        }
    
    except Exception as e:
        return {
            'bank': bank_name,
            'headlines': [],
            'overall_sentiment': 'Neutral',
            'sentiment_score': 0.0,
            'error': str(e)
        }