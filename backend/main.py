import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Import the SentimentAnalyzer for AI-enhanced portfolio optimization
try:
    from backend.src.sentiment_analysis import SentimentAnalyzer
    SENTIMENT_ANALYZER = None  # Will be initialized lazily when needed
except ImportError as e:
    print(f"Warning: Could not import SentimentAnalyzer: {e}. Portfolio optimization will use historical data only.")

# Import the FinancialReportSummarizer for generating briefs
try:
    from backend.src.financial_report_summarization import FinancialReportSummarizer
    REPORT_SUMMARIZER = None  # Will be initialized lazily when needed
except ImportError as e:
    print(f"Warning: Could not import FinancialReportSummarizer: {e}. Brief generation will not be available.")

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API from environment variable only (never hardcode keys)
GEMINI_CONFIGURED = False
gemini_key = os.environ.get("GEMINI_API_KEY")
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        GEMINI_CONFIGURED = True
        print("Gemini API configured successfully.")
    except Exception as e:
        # Non-fatal: allow app to start but log the error.
        print(f"Failed to configure Gemini API: {e}")
else:
    print("Warning: GEMINI_API_KEY not set. Gemini API endpoints will return 503 until configured.")


class ChatRequest(BaseModel):
    ticker: str
    prompt: str
    brief_summary: str

class PitchRequest(BaseModel):
    kpis: dict
    portfolio_composition: list

class OptimizeRequest(BaseModel):
    tickers: list[str] = ['AAPL', 'MSFT', 'JPM', 'PG', 'JNJ', 'XOM']
    years: int = 5

def get_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculates portfolio performance metrics."""
    portfolio_return = np.sum(mean_returns * weights) * 252 # Annualized return
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) # Annualized volatility
    return portfolio_return, portfolio_std_dev

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """Calculates the negative Sharpe ratio (to be minimized)."""
    p_return, p_std_dev = get_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std_dev

def find_max_sharpe_ratio_portfolio(mean_returns, cov_matrix):
    """
    Uses scipy's optimizer to find the portfolio with the maximum Sharpe ratio.
    This replaces the need for Excel Solver.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    # Constraint: sum of weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds for each weight (0 to 1)
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    # Initial guess (equal weights)
    initial_guess = num_assets * [1. / num_assets,]
    
    # Run the optimization
    result = minimize(negative_sharpe_ratio, initial_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x # Returns the optimal weights

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "AI Financial Platform Backend is running"}

# Endpoint evolved from your ai-investment-platform project
@app.post("/api/optimize-portfolio")
async def optimize_portfolio(request: OptimizeRequest):
    """
    Takes a list of tickers and returns the optimal portfolio based on MPT.
    Enhanced with AI sentiment analysis to adjust expected returns.
    """
    try:
        try:
            import yfinance as yf
        except ImportError:
            raise HTTPException(status_code=500, detail=(
                "Missing dependency 'yfinance'. Please install backend requirements: `pip install -r requirements.txt`"
            ))

        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.years * 365)
        
        data_df = yf.download(request.tickers, start=start_date, end=end_date)['Close']
        if getattr(data_df, 'empty', False):
            raise ValueError("No data fetched. Check ticker symbols.")

        # 1. Get historical returns
        returns = data_df.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # 2. Try to get AI-powered sentiment scores to adjust returns
        adjusted_mean_returns = mean_returns.copy()
        try:
            global SENTIMENT_ANALYZER
            if SENTIMENT_ANALYZER is None:
                print("Initializing SentimentAnalyzer...")
                SENTIMENT_ANALYZER = SentimentAnalyzer()
            
            print("Fetching sentiment scores for portfolio optimization...")
            sentiment_scores = SENTIMENT_ANALYZER.get_sentiment_scores(request.tickers)
            
            # Adjust returns based on sentiment using a scaling factor
            # Formula: Adjusted Return = Historical Return + (Sentiment Score * Scaling Factor)
            scaling_factor = 0.25  # Conservative adjustment factor
            
            # Align sentiment scores with mean_returns (ensure same ticker order)
            aligned_sentiment = sentiment_scores.reindex(mean_returns.index, fill_value=0.0)
            
            # Annualize returns, apply sentiment adjustment, then convert back to daily
            annualized_mean_returns = mean_returns * 252
            annualized_adjusted_returns = annualized_mean_returns + (aligned_sentiment * scaling_factor)
            adjusted_mean_returns = annualized_adjusted_returns / 252
            
            print("Original Annualized Returns:\n", annualized_mean_returns)
            print("Adjusted Annualized Returns (with sentiment):\n", annualized_adjusted_returns)
            
        except Exception as sentiment_error:
            print(f"Warning: Could not fetch sentiment scores, using historical returns only: {sentiment_error}")
            # Fall back to historical returns if sentiment fails
            adjusted_mean_returns = mean_returns

        # 3. Find the optimal portfolio using adjusted returns
        optimal_weights_array = find_max_sharpe_ratio_portfolio(adjusted_mean_returns, cov_matrix)
        
        # 4. Calculate final performance metrics
        final_return, final_volatility = get_portfolio_performance(optimal_weights_array, adjusted_mean_returns, cov_matrix)
        final_sharpe = (final_return) / final_volatility  # Assuming risk-free rate = 0

        # Format the results
        optimal_weights_dict = {ticker: weight for ticker, weight in zip(request.tickers, optimal_weights_array)}
        
        return {
            "optimal_weights": optimal_weights_dict,
            "kpis": {
                "return": final_return,
                "volatility": final_volatility,
                "sharpe_ratio": final_sharpe
            }
        }

    except Exception as e:
        print(f"Error during optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints from the finai project (unchanged for now)
@app.post("/api/generate/chat-response")
async def generate_chat_response(request: ChatRequest):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        prompt = f"""
        You are a senior investment analyst. Based on the following summary of an investment brief for {request.ticker}, 
        answer the user's question concisely and professionally.

        Investment Brief Summary:
        "{request.brief_summary}"

        User's Question:
        "{request.prompt}"

        Your Answer:
        """
        response = model.generate_content(prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/pitch")
async def generate_pitch(request: PitchRequest):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        composition_str = ", ".join([f"{item['name']} ({item['weight']}%)" for item in request.portfolio_composition])
        prompt = f"""
        You are a financial advisor. Write a brief, confident, and professional client pitch for an investment portfolio with the following characteristics. 
        Focus on the balance between growth and stability.

        Key Performance Indicators:
        - Annualized Return: {request.kpis.get('annualizedReturn', 'N/A')}
        - Annual Volatility: {request.kpis.get('annualVolatility', 'N/A')}
        - Sharpe Ratio: {request.kpis.get('sharpeRatio', 'N/A')}

        Portfolio Composition:
        {composition_str}

        Client Pitch (2-3 paragraphs):
        """
        response = model.generate_content(prompt)
        return {"pitch": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-brief/{ticker}")
async def generate_brief(ticker: str):
    """
    Generates an investment brief for a given ticker by:
    1. Fetching the latest 10-K filing from SEC
    2. Extracting the MD&A section
    3. Summarizing it using T5 model
    
    Returns a JSON object with the brief summary.
    """
    try:
        global REPORT_SUMMARIZER
        if REPORT_SUMMARIZER is None:
            print("Initializing FinancialReportSummarizer...")
            REPORT_SUMMARIZER = FinancialReportSummarizer()
        
        print(f"Generating brief for {ticker.upper()}...")
        summary = REPORT_SUMMARIZER.get_10k_summary(ticker.upper())
        
        return {
            "ticker": ticker.upper(),
            "summary": summary,
            "status": "success"
        }
    except Exception as e:
        print(f"Error generating brief for {ticker}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate brief for {ticker}: {str(e)}"
        )