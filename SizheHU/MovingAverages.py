import numpy as np
import pandas as pd
# from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf


# Double Crossover
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 1, 1)
TSLA_data = yf.download('TSLA', start_date, end_date)

# TSLA_data[['Adj Close']].plot(figsize=(15,10))
TSLA_data['10_avg'] = TSLA_data['Adj Close'].rolling(window=20).mean()
TSLA_data['50_avg'] = TSLA_data['Adj Close'].rolling(window=50).mean()
new_data = TSLA_data[['Adj Close', '10_avg', '50_avg']].dropna()
new_data.rename(columns={'Adj Close': 'close'}, inplace=True)
new_data['comparison'] = np.where(
new_data['10_avg'] > new_data['50_avg'], 1.0, 0.0)
new_data['signal'] = new_data['comparison'].diff()

# A buy signal is produced when the shorter average crosses above the longer.
# A sell signal is produced when the shorter averages moves below the longer average

if_plot = True
if (if_plot):
    plt.figure(figsize=(17, 7))
    plt.title('Simple Moving Averages')
    plt.xlabel('Days')
    plt.ylabel('Closing Prices')
    new_data['close'].plot(color='k', label='Close Price')
    new_data['10_avg'].plot(color='g', label='20-day Avg Close Price')
    new_data['50_avg'].plot(color='y', label='50-day Avg Close Price')
    plt.plot(new_data[new_data['signal'] == 1].index,
             new_data['10_avg'][new_data['signal'] == 1],
             'o', markersize=5, color='r', label='Buy')
    plt.plot(new_data[new_data['signal'] == -1].index,
             new_data['10_avg'][new_data['signal'] == -1],
             'o', markersize=5, color='b', label='Sell')
    plt.legend()
    plt.show()


init_num_shares = 1000
init_value = 0

cur_value = init_value
cur_num_shares = init_num_shares
new_data = new_data.reset_index()
for index, row in new_data.iterrows():
#     print(row['signal'])
    if row['signal'] == -1:
        # Sell
        cur_value = cur_num_shares * row['close']
        cur_num_shares = 0
        print(f"At time {row['Date']} Sell. Value now: {cur_value}")
    if row['signal'] == 1:
        # Buy
        cur_num_shares = cur_value / row['close']
        cur_value = 0
        print(f"At time {row['Date']} Buy.")
