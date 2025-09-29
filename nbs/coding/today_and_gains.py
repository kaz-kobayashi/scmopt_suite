# filename: today_and_gains.py
import datetime as dt
import csv

# Get today's date
today = dt.date.today()
print(f"Today is {today}")

# Read META stock prices from CSV file
meta_file = open('META.csv', newline='')
meta_reader = csv.reader(meta_file, delimiter=',')
next(meta_reader)  # Skip header line
meta_prices = []
for row in meta_reader:
    date_str, price = row
    date = dt.datetime.strptime(date_str, '%Y-%m-%d').date()
    if date <= today:
        meta_prices.append(float(price))
meta_gain = sum(meta_prices) / len(meta_prices) - float(meta_prices[0])
print(f"META's year-to-date gain is {meta_gain:.2%}")

# Read TESLA stock prices from CSV file
tesla_file = open('TESLA.csv', newline='')
tesla_reader = csv.reader(tesla_file, delimiter=',')
next(tesla_reader)  # Skip header line
tesla_prices = []
for row in tesla_reader:
    date_str, price = row
    date = dt.datetime.strptime(date_str, '%Y-%m-%d').date()
    if date <= today:
        tesla_prices.append(float(price))
tesla_gain = sum(tesla_prices) / len(tesla_prices) - float(tesla_prices[0])
print(f"TESLA's year-to-date gain is {tesla_gain:.2%}")

# Compare the gains
if meta_gain > tesla_gain:
    print("META has a higher year-to-date gain than TESLA.")
elif meta_gain < tesla_gain:
    print("TESLA has a higher year-to-date gain than META.")
else:
    print("Both META and TESLA have the same year-to-date gain.")

TERMINATE