import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import financial libraries
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- Application Setup ---

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # The default port for Vite React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the Gemini API
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("Gemini API configured successfully.")
except KeyError:
    raise RuntimeError("GEMINI_API_KEY not found in .env file. Please create a .env file and add your API key.")

# --- Pydantic Models for Request Bodies ---

class ChatRequest(BaseModel):
    ticker: str
    prompt: str
    brief_summary: str

class PitchRequest(BaseModel):
    kpis: dict
    portfolio_composition: list

class OptimizeRequest(BaseModel):
    tickers: list[str] = ['AAPL', 'MSFT', 'JPM', 'PG', 'JNJ', 'XOM'] # Default tickers from your project
    years: int = 5

# --- Portfolio Optimization Logic (Evolved from your data science project) ---

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
    """
    try:
        # 1. Fetch data (logic from your data_processing.py)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.years * 365)
        
        data_df = yf.download(request.tickers, start=start_date, end=end_date)['Close']
        if data_df.empty:
            raise ValueError("No data fetched. Check ticker symbols.")

        # 2. Calculate inputs (logic from your data_processing.py)
        returns = data_df.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # 3. Find the optimal portfolio (replaces Excel Solver)
        optimal_weights_array = find_max_sharpe_ratio_portfolio(mean_returns, cov_matrix)
        
        # 4. Calculate final performance metrics
        final_return, final_volatility = get_portfolio_performance(optimal_weights_array, mean_returns, cov_matrix)
        final_sharpe = (final_return) / final_volatility # Assuming risk-free rate = 0

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
