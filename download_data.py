import requests
import json


def download_data(symbol: str, year_month: str) -> None:
    print("download_data")
    print(f"Downloading data for {symbol} in {year_month}.")
    # Download data from API
    print(f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey=7L9DZ2E475H0PMVE&month={year_month}&outputsize=full")
    req = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey=7L9DZ2E475H0PMVE&month={year_month}&outputsize=full")
    data = req.json()
    with open(f"{symbol}_{year_month}.json", "w") as f:
        json.dump(data, f)


download_data("NVDA", "2024-10")
download_data("NVDA", "2024-09")
download_data("NVDA", "2024-08")
