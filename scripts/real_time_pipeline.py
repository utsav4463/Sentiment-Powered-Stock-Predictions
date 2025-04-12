# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import yfinance as yf
import pandas as pd
import boto3
import requests
import torch
import joblib
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from decimal import Decimal
import pytz
import os
from sklearn.preprocessing import StandardScaler
 
# AWS Setup
session = boto3.Session()
dynamodb = session.resource("dynamodb")
data_table = dynamodb.Table("realtimedata")
tracker_table = dynamodb.Table("news_tracker")
sentiment_table = dynamodb.Table("sentiment_tracker")
pred_table = dynamodb.Table("realtime_predictions")
s3 = boto3.client("s3")
 
# Constants
STOCKS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
API_KEY = "hDdU807PlusyraTBnNgkkm2gFuPtkZ9F"
S3_BUCKET = "stock-market-data-uofc"
MODEL_PREFIX = "models/sarimax"
LOCAL_MODEL_PATH = "/tmp"
 
# FinBERT Setup
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
 
# Utility Functions
def is_market_open():
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    return now.weekday() < 5 and datetime.strptime("09:30", "%H:%M").time() <= now.time() <= datetime.strptime("16:00", "%H:%M").time()
 
def fetch_stock_data():
    if not is_market_open():
        return pd.DataFrame()
    all_data = []
    for stock in STOCKS:
        df = yf.Ticker(stock).history(period="1d", interval="5m")
        if df.empty: continue
        df.reset_index(inplace=True)
        df["Stock"] = stock
        df.rename(columns={"Datetime": "Date"}, inplace=True)
        all_data.append(df[["Stock", "Date", "Open", "High", "Low", "Close", "Volume"]])
    return pd.concat(all_data, ignore_index=True)
 
def fetch_news():
    all_news = []
    for stock in STOCKS:
        try:
            item = tracker_table.get_item(Key={"Stock": stock, "LastPublished": "latest"}).get("Item")
            last_date = pd.to_datetime(item["LastDate"])
            last_title = item["LastTitle"]
        except:
            last_date = datetime.utcnow() - timedelta(days=1)
            last_title = ""
        res = requests.get("https://financialmodelingprep.com/api/v3/stock_news", params={
            "tickers": stock,
            "from": last_date.strftime('%Y-%m-%d'),
            "to": datetime.utcnow().strftime('%Y-%m-%d'),
            "limit": 50,
            "apikey": API_KEY
        })
        if res.status_code == 200:
            articles = res.json()
            new_articles = [a for a in articles if pd.to_datetime(a["publishedDate"]) > last_date or a["title"] != last_title]
            if new_articles:
                latest = pd.to_datetime(new_articles[0]["publishedDate"])
                tracker_table.put_item(Item={
                    "Stock": stock,
                    "LastPublished": "latest",
                    "LastDate": latest.isoformat(),
                    "LastTitle": new_articles[0]["title"]
                })
                all_news.extend(new_articles)
    return pd.DataFrame(all_news)
 
def get_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)
            return -1 * probs[0][0].item() + probs[0][2].item()
    except:
        return 0.0
 
def get_faded_score(stock, latest_news, score, fade_factor=0.9):
    try:
        entry = sentiment_table.get_item(Key={"Stock": stock}).get("Item")
        prev_news = entry.get("LastNews")
        prev_score = float(entry.get("LastScore"))
        repeat_count = int(entry.get("RepeatCount"))
        if latest_news == prev_news:
            repeat_count += 1
            faded_score = prev_score * (fade_factor ** repeat_count)
        else:
            repeat_count = 0
            faded_score = score
    except:
        faded_score = score
        repeat_count = 0
    sentiment_table.put_item(Item={
        "Stock": stock,
        "LastNews": latest_news,
        "LastScore": Decimal(str(round(score, 4))),
        "RepeatCount": repeat_count
    })
    return round(faded_score, 4)
 
def add_features(df):
    df["EMA_5"] = df["Close"].ewm(span=5).mean()
    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["Volatility_30min"] = df["Close"].rolling(window=6).std()
    df["Price_Change_Pct"] = df["Close"].pct_change()
    df["Lag_Close"] = df["Close"].shift(1)
    df["Weekday"] = df["Date"].dt.weekday
    return df
 
def load_model_from_s3(stock):
    local_path = os.path.join(LOCAL_MODEL_PATH, f"{stock}.pkl")
    s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/{stock}.pkl", local_path)
    return joblib.load(local_path)
 
def make_prediction_row(stock, latest_row, model_obj):
    features = model_obj["features"]
    scaler = model_obj["scaler"]
    model = model_obj["model"]
 
    X = scaler.transform(latest_row[features])
    pred_15 = model.forecast(steps=3, exog=X.repeat(3, axis=0)).mean()
    pred_30 = model.forecast(steps=6, exog=X.repeat(6, axis=0)).mean()
    pred_60 = model.forecast(steps=12, exog=X.repeat(12, axis=0)).mean()
 
    return {
        "Stock": stock,
        "Timestamp": latest_row["Date"].iloc[0].isoformat(),
        "Current_Close": round(float(latest_row["Close"]), 4),
        "Pred_15min": round(float(pred_15), 4),
        "Pred_30min": round(float(pred_30), 4),
        "Pred_60min": round(float(pred_60), 4)
    }
 
def process_all(stock_df, news_df):
    news_df = news_df.sort_values("publishedDate")
    sentiment_scores = {}
    for stock in STOCKS:
        latest_news = news_df[news_df["symbol"] == stock].sort_values("publishedDate").tail(1)
        if latest_news.empty:
            sentiment_scores[stock] = Decimal("0.0")
            continue
        text = latest_news.iloc[0]["text"]
        raw_score = get_sentiment(text)
        faded = get_faded_score(stock, text, raw_score)
        sentiment_scores[stock] = Decimal(str(faded))
 
    prediction_rows = []
    for stock in STOCKS:
        sdf = stock_df[stock_df["Stock"] == stock].copy()
        if sdf.empty: continue
        score = sentiment_scores.get(stock, Decimal("0.0"))
        sdf["faded_sentiment_score"] = float(score)
        sdf = add_features(sdf)
        sdf.dropna(inplace=True)
        if sdf.empty: continue
 
        # Use last row for Dynamo + Prediction
        latest = sdf.tail(1).copy()
        item = {
            "Stock": stock,
            "Timestamp": latest["Date"].iloc[0].isoformat(),
            "Open": Decimal(str(latest["Open"].iloc[0])),
            "High": Decimal(str(latest["High"].iloc[0])),
            "Low": Decimal(str(latest["Low"].iloc[0])),
            "Close": Decimal(str(latest["Close"].iloc[0])),
            "Volume": int(latest["Volume"].iloc[0]),
            "faded_sentiment_score": Decimal(str(latest["faded_sentiment_score"].iloc[0]))
        }
        data_table.put_item(Item=item)
 
        try:
            model_obj = load_model_from_s3(stock)
            pred_row = make_prediction_row(stock, latest, model_obj)
            pred_table.put_item(Item={k: (Decimal(str(v)) if isinstance(v, float) else v) for k, v in pred_row.items()})
        except:
            continue
 
# Main
if __name__ == "__main__":
    print("🚀 Ingestion + Prediction job started")
    stock_data = fetch_stock_data()
    if not stock_data.empty:
        news_data = fetch_news()
        process_all(stock_data, news_data)
        print("✅ Data ingested & predictions made")
    else:
        print("⚠️ Market closed or no stock data")
