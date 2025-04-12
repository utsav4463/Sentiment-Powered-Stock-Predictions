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
import pandas as pd
import boto3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from tqdm import tqdm
from decimal import Decimal
 
# --- AWS Configuration ---
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("realtimedata")
 
# --- S3 Configuration ---
s3 = boto3.client("s3")
bucket = "stock-market-data-uofc"
stock_key = "historical-data/stock_60d_5m.csv"
news_key = "news-data/news_history_1yr.csv"
 
# --- Load Data ---
stock_df = pd.read_csv(s3.get_object(Bucket=bucket, Key=stock_key)["Body"])
news_df = pd.read_csv(s3.get_object(Bucket=bucket, Key=news_key)["Body"])
 
# --- Preprocessing ---
stock_df["Date"] = pd.to_datetime(stock_df["Date"], utc=True)
stock_df["Date"] = stock_df["Date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
stock_df["Date"] = stock_df["Date"].dt.floor("min")
 
news_df["publishedDate"] = pd.to_datetime(news_df["publishedDate"], errors="coerce")
news_df["publishedDate"] = news_df["publishedDate"].dt.floor("min")
news_df.rename(columns={"symbol": "Stock"}, inplace=True)
 
# --- Filter news in stock date range ---
min_date = stock_df["Date"].min().date()
max_date = stock_df["Date"].max().date()
news_df = news_df[
    (news_df["publishedDate"].dt.date >= min_date) &
    (news_df["publishedDate"].dt.date <= max_date)
]
 
# --- Assign latest news to each stock row ---
stock_df.sort_values(["Stock", "Date"], inplace=True)
news_df.sort_values(["Stock", "publishedDate"], inplace=True)
 
assigned_news = []
for symbol, group in stock_df.groupby("Stock"):
    stock_times = group["Date"].values
    news_times = news_df[news_df["Stock"] == symbol][["publishedDate", "text"]].values
    latest_news = None
    news_index = 0
    for stock_time in stock_times:
        while news_index < len(news_times) and news_times[news_index][0] <= stock_time:
            latest_news = news_times[news_index][1]
            news_index += 1
        assigned_news.append(latest_news)
 
stock_df["latest_news"] = assigned_news
 
# --- Sentiment Analysis (FinBERT) ---
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
 
def get_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1)
            scores = probs[0].tolist()
            return -1 * scores[0] + 0 * scores[1] + 1 * scores[2]
    except:
        return 0.0
 
unique_news = stock_df["latest_news"].dropna().unique()
news_sentiment_map = {news: get_sentiment(news) for news in tqdm(unique_news)}
 
# --- Sentiment Fading ---
faded_scores = []
fade_factor = 0.9
prev_news = None
prev_score = 0.0
repeat_count = 0
 
for news in stock_df["latest_news"]:
    if news != prev_news:
        sentiment = news_sentiment_map.get(news, 0.0)
        repeat_count = 0
    else:
        repeat_count += 1
        sentiment = prev_score * (fade_factor ** repeat_count)
    faded_scores.append(round(sentiment, 4))
    prev_news = news
    prev_score = news_sentiment_map.get(news, 0.0)
 
stock_df["faded_sentiment_score"] = faded_scores
 
# --- Upload to DynamoDB ---
print("Uploading 60-day data to DynamoDB...")
for _, row in stock_df.iterrows():
    try:
        table.put_item(Item={
            "Stock": row["Stock"],
            "Timestamp": row["Date"].strftime("%Y-%m-%dT%H:%M:%S"),
            "Open": Decimal(str(row["Open"])),
            "High": Decimal(str(row["High"])),
            "Low": Decimal(str(row["Low"])),
            "Close": Decimal(str(row["Close"])),
            "Volume": int(row["Volume"]),
            "faded_sentiment_score": Decimal(str(row["faded_sentiment_score"]))
        })
    except Exception as e:
        print(f"Error inserting row: {e}")
 
print("✅ Historical 60-day data uploaded.")
