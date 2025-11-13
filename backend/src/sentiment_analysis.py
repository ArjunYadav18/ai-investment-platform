import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newsapi import NewsApiClient
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer:
    """
    A class to handle FinBERT sentiment analysis.
    It loads the model once and provides a method to get scores.
    """
    def __init__(self):
        """
        Initializes the FinBERT model, tokenizer, and NewsAPI client.
        """
        print("Initializing FinBERT model (this may take a moment)...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print("FinBERT Model initialized successfully.")

        self.api_key = os.environ.get("NEWS_API_KEY")
        if not self.api_key:
            raise RuntimeError("NEWS_API_KEY not found in .env file.")
            
        self.newsapi = NewsApiClient(api_key=self.api_key)

    def _fetch_recent_news(self, ticker: str) -> list:
        """
        Fetches recent news headlines for a given stock ticker.
        """
        print(f"Fetching news for {ticker}...")
        try:
            articles = self.newsapi.get_everything(q=ticker,
                                                  language='en',
                                                  sort_by='publishedAt',
                                                  page_size=20)
            
            headlines = [article['title'] for article in articles.get('articles', [])]
            print(f"Found {len(headlines)} headlines for {ticker}.")
            return headlines
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []

    def _analyze_sentiment(self, headlines: list) -> float:
        """
        Analyzes a list of headlines and returns a single aggregated score.
        Score is (avg_positive - avg_negative).
        """
        if not headlines:
            return 0.0

        total_positive = 0
        total_negative = 0
        
        with torch.no_grad():
            for headline in headlines:
                inputs = self.tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)

                positive_score = scores[0][0].item()
                negative_score = scores[0][1].item()
                
                total_positive += positive_score
                total_negative += negative_score

        num_headlines = len(headlines)
        final_score = (total_positive - total_negative) / num_headlines
        return final_score

    def get_sentiment_scores(self, tickers: list[str]) -> pd.Series:
        """
        Main public method.
        Gets sentiment scores for a list of tickers and returns them as a pandas Series.
        """
        sentiment_scores = {}
        for ticker in tickers:
            headlines = self._fetch_recent_news(ticker)
            sentiment = self._analyze_sentiment(headlines)
            sentiment_scores[ticker] = sentiment
            print(f"--- {ticker} Sentiment Score: {sentiment:.4f} ---")
            
        return pd.Series(sentiment_scores)