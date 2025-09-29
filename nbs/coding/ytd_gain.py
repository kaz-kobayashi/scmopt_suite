# filename: ytd_gain.py
import yfinance as yf
import datetime

# Get today's date
today = datetime.date.today()
print("Today's date:", today)

# Define the ticker symbols for META and TESLA stocks
tickers = ['FB', 'TSLA']

# Download historical data for the given tickers using yfinance
data = yf.download(tickers, start='2021-01-01', auto_adjust=True)

# Calculate year-to-date gain for each stock
for ticker in tickers:
    ytd_gain = data[ticker].iloc[-1]['Return'] * 100
    print(f"{ticker.upper()} Year-to-Date Gain: {ytd_gain:.2f}%")