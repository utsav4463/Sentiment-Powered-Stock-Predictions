import yfinance as yf
import pandas as pd
import boto3
from datetime import datetime
 
# AWS S3 Configuration
S3_BUCKET = "stock-market-data-uofc"
S3_KEY = "historical-data/stock_history_1y.csv"
#S3_BUCKET = "data608-2025-stock-market-data"
#S3_KEY = "historical-stock-data/stock_1y_1d.csv"
s3_client = boto3.client("s3")
 
# List of stocks to fetch
stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
 
# Fetch stock data from a given start date
def fetch_stock_data(start_date):
    all_stock_data = []
    for stock in stocks:
        print(f"Fetching {stock} from {start_date}...")
        ticker = yf.Ticker(stock)
        history = ticker.history(start=start_date, interval="1d")
        history.reset_index(inplace=True)
        history["Stock"] = stock
        history = history[["Stock", "Date", "Open", "High", "Low", "Close", "Volume"]]
        all_stock_data.append(history)
    return pd.concat(all_stock_data, ignore_index=True)
 
# Load data from S3
def load_existing_data():
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        existing_data = pd.read_csv(response['Body'], parse_dates=["Date"])
        print(f"Loaded existing data: {len(existing_data)} rows")
        return existing_data
    except s3_client.exceptions.NoSuchKey:
        print("No existing data found. Fetching fresh...")
        return None
 
# Save DataFrame to S3
def upload_to_s3(df):
    csv_buffer = df.to_csv(index=False)
    s3_client.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=csv_buffer)
    print(f"Uploaded to S3: {S3_KEY}")
 
# Main logic
existing_df = load_existing_data()
 
if existing_df is not None:
    last_date = existing_df["Date"].max().date()
    next_date = last_date + pd.Timedelta(days=1)
    new_data = fetch_stock_data(start_date=next_date)
    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    combined_df.drop_duplicates(subset=["Stock", "Date"], inplace=True)
else:
    start_date = (datetime.now() - pd.Timedelta(days=365)).date()
    combined_df = fetch_stock_data(start_date=start_date)
 
upload_to_s3(combined_df)