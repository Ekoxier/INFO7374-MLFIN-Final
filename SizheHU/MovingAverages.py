import numpy as np
import pandas as pd
# from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf

start_date = datetime(2020,1,1)
end_date = datetime(2023,1,1)
TSLA_data = yf.download('TSLA',start_date ,end_date)
# TSLA_data[['Adj Close']].plot(figsize=(15,10))
TSLA_data['10_avg'] = TSLA_data['Adj Close'].rolling(window = 20).mean()
TSLA_data['50_avg'] = TSLA_data['Adj Close'].rolling(window = 50).mean()
new_data = TSLA_data[['Adj Close','10_avg','50_avg']].dropna()
new_data.rename(columns = {'Adj Close':'close'}, inplace = True)
new_data['comparison'] = np.where(new_data['10_avg'] > new_data['50_avg'], 1.0, 0.0)
new_data['signal'] = new_data['comparison'].diff()
# print(new_data)

# A buy signal is produced when the shorter average crosses above the longer.
# A sell signal is produced when the shorter averages moves below the longer average
print(new_data)
if_plot = True
if (if_plot):
    plt.figure(figsize = (10,6))
    new_data['close'].plot(color = 'k', label= 'Daily close price') 
    new_data['10_avg'].plot(color = 'g',label = '20-day avg close price') 
    new_data['50_avg'].plot(color = 'y',label = '50-day avg close price')
    plt.plot(new_data[new_data['signal'] == 1].index, 
         new_data['10_avg'][new_data['signal'] == 1], 
         'o', markersize = 5, color = 'r', label = 'buy')
    plt.plot(new_data[new_data['signal'] == -1].index, 
            new_data['10_avg'][new_data['signal'] == -1], 
            'o', markersize = 5, color = 'b', label= 'sell')
    plt.legend()
    plt.show()