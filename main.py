import pandas as pd

train_data = pd.read_parquet("C:/Users/Floater PC/OneDrive - Hult Students/Documents/GitHub/jane-street-real-time-market-data-forecasting/train.parquet")

print(train_data.head())
print(train_data.info())  # Get information on data types and missing values
print(train_data.describe())  # Get statistical summary

