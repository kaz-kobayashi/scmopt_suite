# filename: tesla_gain.py
import yfinance as yf
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Get today's date
today = datetime.date.today()
print("Today's date:", today)

# Download historical data for TESLA stock
tesla = yf.Ticker("TSLA")
data = tesla.history(period="max")

# Convert today's date to tz-aware datetime
today_aware = pd.Timestamp(year=today.year, month=today.month, day=today.day, tz='UTC')

# Filter the data to get the year-to-date data
start_date = pd.to_datetime(pd.Timestamp(year=today.year, month=today.month, day=1)).tz_localize('UTC').tz_convert('UTC')
end_date = today_aware.tz_convert('UTC')
filtered_data = data[['Close']][(data.index >= start_date) & (data.index <= end_date)]

# Calculate year-to-date gain
ytd_gain = (filtered_data.iloc[-1]['Close'] / filtered_data.iloc[0]['Close']) - 1
print("Year-to-date gain for TESLA:", ytd_gain)

# Plot the chart of TESLA's stock price change YTD
plt.figure(figsize=(12,6))
plt.plot(filtered_data.index, filtered_data['Close'])
plt.title('Tesla Stock Price Change YTD')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid()
plt.savefig('stock_price_ytd.png')
plt.show()