import requests
import json


def download_data_tiingo(symbol: str, year_month: str) -> None:
    print("download_data_tiingo")
    print(f"Downloading data for {symbol} in {year_month}.")
    # Download data from API
    # print(f"https://api.tiingo.com/iex/{symbol}/prices?startDate={year_month}&resampleFreq=1min&&resampleFreq=1min&token=6d85abf36badf7c156fde99d822d251f1e973583")
    # req = requests.get(f"https://api.tiingo.com/iex/{symbol}/prices?startDate={year_month}&resampleFreq=10min&token=6d85abf36badf7c156fde99d822d251f1e973583")
    req = requests.get(
        f"https://api.tiingo.com/tiingo/crypto/prices?tickers=btcusd&resampleFreq=10min&token=6d85abf36badf7c156fde99d822d251f1e973583")
    data = req.json()
    with open(f"{symbol}_{year_month}.json", "w") as f:
        json.dump(data, f)


# download_data_tiingo("BTC", "2024-08-01")
