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
import numpy as np
import boto3
import joblib
import os
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from decimal import Decimal
import warnings
 
warnings.filterwarnings("ignore")
 
# --- Config ---
LOCAL_MODEL_DIR = "/tmp/trained_models"
S3_BUCKET = "stock-market-data-uofc"
S3_PREFIX = "models/sarimax"
DYNAMO_TABLE = "realtimedata"
 
# Create local model directory
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
 
# --- Load Data from DynamoDB ---
def load_data_from_dynamodb():
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(DYNAMO_TABLE)
 
    response = table.scan()
    items = response["Items"]
 
    # Keep paginating if needed
    while "LastEvaluatedKey" in response:
        response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        items.extend(response["Items"])
 
    df = pd.DataFrame(items)
 
    # Convert to correct types
    df["Date"] = pd.to_datetime(df["Timestamp"])
    for col in ["Open", "High", "Low", "Close", "faded_sentiment_score"]:
        df[col] = df[col].astype(float)
    df["Volume"] = df["Volume"].astype(int)
    return df
 
# --- Feature Engineering ---
def add_intraday_features(df):
    df = df.copy()
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["Volatility_30min"] = df["Close"].rolling(window=6).std()
    df["Price_Change_Pct"] = df["Close"].pct_change()
    df["Lag_Close"] = df["Close"].shift(1)
    df["Weekday"] = df["Date"].dt.weekday
    return df
 
# --- Train & Upload Model ---
def train_and_save_model(df, stock, s3_client):
    stock_df = df[df["Stock"] == stock].copy()
    stock_df = add_intraday_features(stock_df)
    stock_df.set_index("Date", inplace=True)
    stock_df.sort_index(inplace=True)
 
    feature_cols = [
        "Volume",
        "faded_sentiment_score",
        "EMA_5",
        "EMA_12",
        "Volatility_30min",
        "Price_Change_Pct",
        "Lag_Close",
        "Weekday"
    ]
 
    ts_df = stock_df[["Close"] + feature_cols].dropna()
    if len(ts_df) < 100:
        return
 
    endog = ts_df["Close"]
    exog = ts_df[feature_cols]
 
    scaler = StandardScaler()
    exog_scaled = pd.DataFrame(scaler.fit_transform(exog), columns=feature_cols, index=exog.index)
 
    try:
        arima_model = auto_arima(
            endog, exogenous=exog_scaled,
            seasonal=False, stepwise=True,
            suppress_warnings=True, max_order=6, maxiter=30
        )
        order = arima_model.order
    except:
        order = (2, 1, 2)
 
    try:
        model = SARIMAX(endog, exog=exog_scaled, order=order, enforce_stationarity=False)
        result = model.fit(disp=False, maxiter=50)
 
        local_path = os.path.join(LOCAL_MODEL_DIR, f"{stock}.pkl")
        joblib.dump({
            "model": result,
            "scaler": scaler,
            "features": feature_cols,
            "order": order
        }, local_path)
 
        s3_key = f"{S3_PREFIX}/{stock}.pkl"
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    except:
        pass
 
# --- Main ---
if __name__ == "__main__":
    print("🚀 Loading data from DynamoDB...")
    df = load_data_from_dynamodb()
    s3 = boto3.client("s3")
    stocks = df["Stock"].unique()
    for stock in stocks:
        train_and_save_model(df, stock, s3)
    print("✅ Model training & upload complete.")
