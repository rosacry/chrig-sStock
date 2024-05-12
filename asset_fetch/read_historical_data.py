import pandas as pd
from datetime import datetime
import yfinance as yf

def fetch_tickers_from_file(file_path):
    df = pd.read_csv(file_path)
    # Ensure the 'symbol' column is treated as string and drop NaN values
    df['symbol'] = df['symbol'].dropna().astype(str)
    return df['symbol'].tolist()

def download_stock_data(tickers, start_year):
    today_date = datetime.now().strftime('%Y-%m-%d')
    data = pd.DataFrame()
    for ticker in tickers:
        print(f"Downloading data for {ticker} from {start_year} to today ({today_date})")
        try:
            stock_data = yf.download(ticker, start=f"{start_year}-01-01", end=today_date)
            stock_data['Symbol'] = ticker
            if not stock_data.empty:
                data = pd.concat([data, stock_data])
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")
    return data

def main():
    iex_tickers = fetch_tickers_from_file('asset_fetch/tickers/iex-tickers.csv')
    start_year = 2000
    current_year = datetime.now().year

    for year in range(start_year, current_year + 1):
        all_stock_data = download_stock_data(iex_tickers, year)
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        all_stock_data = all_stock_data.reset_index()
        all_stock_data = all_stock_data[columns]
        all_stock_data.to_csv(f"asset_fetch/stock_data/stock_data_{year}_to_{current_year}.csv", index=False)
        print(f"Saved data to asset_fetch/stock_data/stock_data_{year}_to_{current_year}.csv")

if __name__ == "__main__":
    main()
