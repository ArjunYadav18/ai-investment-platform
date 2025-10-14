# src/sentiment_analysis.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newsapi import NewsApiClient
import pandas as pd
import os

# --- Configuration ---
NEWS_API_KEY = "cd6160d0fd88428396386bed3906f75e"
TICKERS = ['AAPL', 'MSFT', 'JPM', 'PG', 'JNJ', 'XOM']

def initialize_sentiment_model():
    """
    Initializes the FinBERT model and tokenizer from Hugging Face.
    This will download the model the first time it's run.
    """
    print("Initializing FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    print("Model initialized successfully.")
    return model, tokenizer

def fetch_recent_news(api_key, ticker):
    """
    Fetches recent news headlines for a given stock ticker using NewsAPI.
    """
    print(f"Fetching news for {ticker}...")
    try:
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=ticker,
                                          language='en',
                                          sort_by='publishedAt',
                                          page_size=20) # Fetch 20 most recent articles
        
        headlines = [article['title'] for article in articles['articles']]
        print(f"Found {len(headlines)} headlines for {ticker}.")
        return headlines
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def analyze_sentiment(headlines, model, tokenizer):
    """
    Analyzes the sentiment of a list of headlines using the FinBERT model.
    Returns a single sentiment score (positive - negative).
    """
    if not headlines:
        return 0.0

    total_positive = 0
    total_negative = 0
    
    with torch.no_grad():
        for headline in headlines:
            inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            positive_score = scores[0][0].item()
            negative_score = scores[0][1].item()
            
            total_positive += positive_score
            total_negative += negative_score

    num_headlines = len(headlines)
    final_score = (total_positive - total_negative) / num_headlines
    return final_score

if __name__ == "__main__":
    finbert_model, finbert_tokenizer = initialize_sentiment_model()
    
    sentiment_scores = {}
    for ticker in TICKERS:
        news_headlines = fetch_recent_news(NEWS_API_KEY, ticker)
        sentiment = analyze_sentiment(news_headlines, finbert_model, finbert_tokenizer)
        sentiment_scores[ticker] = sentiment
        print(f"--- {ticker} Sentiment Score: {sentiment:.4f} ---")
        
    sentiment_df = pd.DataFrame.from_dict(sentiment_scores, orient='index', columns=['sentiment_score'])
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'data', 'sentiment_scores.csv')
    sentiment_df.to_csv(output_path)
    
    print("\nSentiment analysis complete.")
    print(f"Scores saved to {output_path}")