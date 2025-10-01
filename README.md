AI-Enhanced Investment Decision Platform

Project Overview
This project is a comprehensive platform designed to augment traditional quantitative portfolio optimization with modern AI-driven insights. It moves beyond purely historical data by integrating real-time market sentiment from financial news and qualitative insights from official company reports (10-K filings).

The platform constructs a baseline optimal portfolio using Modern Portfolio Theory and then enhances it through two layers of Natural Language Processing (NLP):

Sentiment Analysis: Uses FinBERT to score news sentiment, creating a forward-looking adjustment to expected returns.

Report Summarization: Uses a Transformer-based model (T5) to distill complex "Management's Discussion and Analysis" sections of 10-K reports into concise, actionable insights.

The final output is a recommended portfolio and a dashboard that clearly visualizes the impact of the AI enhancements.

Tech Stack
Programming Language: Python 3.10+

Data Analysis & Quant: Pandas, NumPy, SciPy

AI / NLP: Hugging Face Transformers, PyTorch

Data Sources: yfinance (stock data), newsapi-python (news), sec-api (filings)

Optimization & Dashboard: Microsoft Excel (Solver)

Features
Quantitative Portfolio Optimization: Calculates the optimal portfolio allocation to maximize the Sharpe Ratio based on historical returns and volatility.

NLP-Powered News Sentiment Analysis: Fetches recent financial news for a basket of stocks and uses FinBERT to classify sentiment, generating a score to adjust expected returns.

Transformer-Based Report Summarization: Extracts and summarizes key sections from SEC 10-K filings to provide qualitative insights on management's outlook, risks, and performance drivers.

Integrated Dashboard: A comprehensive Excel dashboard that visualizes the baseline portfolio, the final AI-adjusted portfolio, news sentiment scores, and key takeaways from financial reports.
