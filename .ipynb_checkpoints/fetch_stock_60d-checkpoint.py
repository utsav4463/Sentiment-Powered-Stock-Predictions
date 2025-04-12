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
from datetime import datetime, timedelta
from datetime import timezone
import pytz
start_time = datetime.now(pytz.timezone("America/New_York")).replace(tzinfo=None) - timedelta(days=60)
 
 
# AWS S3 Configuration
#S3_BUCKET = "data608-2025-stock-market-data"
#bucket = "stock-market-data-uofc"
#stock_key = "historical-data/stock_60d_5m.csv"
S3_BUCKET = "stock-market-data-uofc"
S3_KEY = "historical-data/stock_60d_5m.csv"
s3_client = boto3.client("s3")
 
# Stocks to fetch
stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
 
# Function to fetch 5-minute interval data for the last 60 days
def fetch_stock_data(start_datetime):
    all_data = []
    for stock in stocks:
        print(f"Fetching {stock} from {start_datetime}...")
        ticker = yf.Ticker(stock)
        history = ticker.history(period="60d", interval="5m")
        history.reset_index(inplace=True)
        history["Stock"] = stock
        history = history.rename(columns={"Datetime": "Date"})
        history["Date"] = history["Date"].dt.tz_localize(None)  # Remove timezone info
        history = history[["Stock", "Date", "Open", "High", "Low", "Close", "Volume"]]
        history = history[history["Date"] > start_datetime]
        all_data.append(history)
    return pd.concat(all_data, ignore_index=True)
 
# Load existing data from S3
def load_existing_data():
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        existing_df = pd.read_csv(response['Body'], parse_dates=["Date"])
        print(f"Loaded existing data: {len(existing_df)} rows")
        return existing_df
    except s3_client.exceptions.NoSuchKey:
        print("No existing data found in S3.")
        return None
 
# Upload DataFrame to S3
def upload_to_s3(df):
    csv_buffer = df.to_csv(index=False)
    s3_client.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=csv_buffer)
    print(f"Uploaded updated dataset to S3: {S3_KEY}")
 
# Main logic
existing_df = load_existing_data()
 
if existing_df is not None:
    last_time = existing_df["Date"].max()
    new_data = fetch_stock_data(start_datetime=last_time)
    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    combined_df.drop_duplicates(subset=["Stock", "Date"], inplace=True)
else:
    print("First-time fetch for last 60 days (5m)...")
    start_time = datetime.now() - timedelta(days=60)
    combined_df = fetch_stock_data(start_datetime=start_time)
 
upload_to_s3(combined_df)
