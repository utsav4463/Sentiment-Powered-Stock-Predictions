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
mport requests
import pandas as pd
from datetime import datetime, timedelta
import time
import boto3
from io import StringIO
 
# AWS S3 Setup
#S3_BUCKET = "data608-2025-stock-market-data"
#S3_KEY = "news-data/historical_news_1yr.csv"
S3_BUCKET = "stock-market-data-uofc"
S3_KEY = "news-data/news_history_1yr.csv"
s3_client = boto3.client("s3")
 
# API Setup
API_KEY = 'hDdU807PlusyraTBnNgkkm2gFuPtkZ9F'
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
tickers_str = ','.join(tickers)
 
def load_existing_news():
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        existing_df = pd.read_csv(response['Body'], parse_dates=["publishedDate"])
        print(f"Loaded existing news data: {len(existing_df)} articles")
        return existing_df
    except s3_client.exceptions.NoSuchKey:
        print("No existing news file found in S3.")
        return None
 
def upload_to_s3(df):
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    s3_client.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buffer.getvalue())
    print(f"Updated news data uploaded to S3: {S3_KEY}")
 
def fetch_news(from_date, to_date):
    page = 0
    limit = 100
    all_news = []
 
    print(f"Fetching news from {from_date} to {to_date}")
    while True:
        page += 1
        print(f"Fetching page {page}")
        url = "https://financialmodelingprep.com/api/v3/stock_news"
        params = {
            "tickers": tickers_str,
            "from": from_date,
            "to": to_date,
            "limit": limit,
            "page": page,
            "apikey": API_KEY
        }
 
        response = requests.get(url, params=params)
        if response.status_code == 200:
            news_batch = response.json()
            if not news_batch:
                break
            all_news.extend(news_batch)
            time.sleep(1)
        else:
            print(f"Failed with status {response.status_code}")
            break
 
    if all_news:
        df = pd.DataFrame(all_news)
        df["publishedDate"] = pd.to_datetime(df["publishedDate"])
        df = df.sort_values(by=["symbol", "publishedDate"])
        return df
    else:
        print("No news returned.")
        return pd.DataFrame()
 
# Main logic
existing_news = load_existing_news()
 
if existing_news is not None:
    last_date = existing_news["publishedDate"].max()
    start_date = last_date.strftime('%Y-%m-%d')
    print(f"Updating from last saved date: {start_date}")
else:
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    print(f"First-time fetch from: {start_date}")
    
end_date = datetime.now().strftime('%Y-%m-%d')
 
# Fetch new data
new_news_df = fetch_news(start_date, end_date)
 
# Merge & upload
if not new_news_df.empty:
    if existing_news is not None:
        combined_df = pd.concat([existing_news, new_news_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["symbol", "publishedDate", "title"], inplace=True)
    else:
        combined_df = new_news_df
 
    upload_to_s3(combined_df)
else:
    print("News already up to date. No new records.")
