
# TweetStorm to Wall Street: Sentiment–Powered Stock Predictions

A real-time stock forecasting pipeline that combines **news sentiment analysis** and **time-series modeling** to predict stock price movements every 5 minutes. The system utilizes AWS cloud services, **FinBERT** for financial sentiment scoring, and **SARIMAX** for time-series forecasting.

---

## Project Overview

This project tracks seven major tech stocks (AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA), merges historical stock price data with financial news sentiment, and delivers real-time trend predictions. The pipeline supports automated ingestion, transformation, modeling, and visualization using AWS infrastructure.

---

## Data Engineering Lifecycle

| Stage                | Description |
|----------------------|-------------|
| **Data Collection**  | Historical stock data (1-year daily, 60-day intraday) via yFinance and financial news headlines via FinancialModelingPrep API. |
| **Ingestion**        | Scripts run on EC2 to push raw/processed data to Amazon S3 and DynamoDB. |
| **Storage**          | Amazon S3 for long-term storage (CSV), DynamoDB for real-time updates and predictions. |
| **Transformation**   | FinBERT for sentiment scoring, plus technical indicators (EMA, volatility, lagged prices). |
| **Model Inference**  | Real-time SARIMAX-based predictions using trained models from S3, triggered every 5 minutes. |
| **Analytics**        | Daily retraining of SARIMAX models and updating prediction logs in DynamoDB. |

---

## Machine Learning Approach

We apply **SARIMAX (Seasonal ARIMA with Exogenous Variables)** for forecasting short-term stock movements. Models are retrained daily on a rolling 60-day window using 5-minute interval data.

### Features used:
- Exponential Moving Averages (5, 12)
- 30-minute rolling volatility
- Percentage price change
- Lagged closing prices
- Day of the week
- Faded FinBERT sentiment score

---

## AWS Services Used

| Service         | Role |
|----------------|------|
| **EC2**         | Runs scripts for data ingestion, training, and prediction |
| **S3**          | Stores CSVs and serialized model artifacts |
| **DynamoDB**    | Real-time data store for prices and predictions |
| **SageMaker**   | Used for one-time sentiment scoring via FinBERT |
| **QuickSight**  | Dashboard to monitor trends, predictions, and news sentiment |

---

## Dashboard Capabilities

The **Amazon QuickSight** dashboard provides:

- Historical stock trends (1 year)
- Real-time prices (60-day intraday)
- Latest headlines and sentiment scores
- 15/30/60-minute forecasted price movements
- Trend indicators (Up / Down)

---

## Project Structure

```
TweetStorm-to-Wall-Street/
├── README.md
├── scripts/
│   ├── fetch_stock_1yr.py
│   ├── fetch_news_1yr.py
│   ├── fetch_stock_60d.py
│   ├── preprocess_sentiment.py
│   ├── train_sarimax_models.py
│   └── real_time_pipeline.py
├── models/
├── dashboards/
├── architecture/
└── data/
```

---

## Optimization Techniques

- Efficient DynamoDB reads/writes every 5 minutes
- Model caching in S3 to avoid retraining
- News fetcher avoids duplicate headlines
- Sentiment fading logic reduces compute
- Lightweight EC2 jobs for cost efficiency

---

## Future Improvements

- Optimize for Cost and Performance
- Try Finance-Specific Forecasting Models
- Introduce Drift Detection and Alerts
- Expand Sentiment Sources
- Improve Dashboard Interactivity

---

## References

- [yFinance](https://github.com/ranaroussi/yfinance)
- [Financial Modeling Prep API](https://site.financialmodelingprep.com/)
- [FinBERT on Hugging Face](https://huggingface.co/yiyanghkust/finbert-tone)
- [SARIMAX Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

---

## Contributors

- **Vineetkumar Vijaybhai Patel** – Data Engineering, Pipeline Automation  
- **Utsav Bharatbhai Goti** – ML Modeling, Pipeline Automation  
- **Gagandeep Thind** – SageMaker Integration, Dashboard Development  
- **Rehan Chanegaon** – Sentiment Analysis, Feature Engineering  

---

## License

This project is licensed under the **MIT License**. Use freely with attribution.
