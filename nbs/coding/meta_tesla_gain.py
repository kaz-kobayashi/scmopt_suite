# filename: meta_tesla_gain.py
import yfinance as yf
import datetime

# Get today's date
today = datetime.date.today()
print("Today's date:", today)

# Define the ticker symbols for META and TESLA stocks
tickers = ['META', 'TSLA']

# Download historical data for both stocks using yfinance
data = yf.download(tickers, start='2021-01-01', end=today.strftime('%Y-%m-%d'))

# Calculate the year-to-date gain for each stock
for ticker in tickers:
    print(f"Year-to-date gain for {ticker}: {data[ticker].pct_change()[0]} %")