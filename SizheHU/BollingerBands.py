import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

def get_sma(prices, rate):
    return prices.rolling(rate).mean()

def get_bollinger_bands(prices, rate=20):
    sma = get_sma(prices, rate)
    std = prices.rolling(rate).std()
    bollinger_up = sma + std * 2 # Calculate top band
    bollinger_down = sma - std * 2 # Calculate bottom band
    return bollinger_up, bollinger_down

start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 1, 1)
tsla_data = yf.download('TSLA', start_date, end_date)

closing_prices = tsla_data['Adj Close']

bollinger_up, bollinger_down = get_bollinger_bands(closing_prices)
final_data = pd.DataFrame({'Close':closing_prices,'Upper':bollinger_up,'Lower':bollinger_down}).dropna()
final_data['buy_signal'] = np.where(final_data['Close'] <= final_data['Lower'],1,0)
final_data['sell_signal'] = np.where(final_data['Close'] >= final_data['Upper'],1,0)
print(final_data)
plt.figure(figsize=(17,7))
plt.title('Bollinger Bands')
plt.xlabel('Days')
plt.ylabel('Closing Prices')
plt.plot(final_data['Close'],'k', label='Close Price')
plt.plot(final_data['Upper'], 'g',label='Upper')
plt.plot(final_data['Lower'], 'y',label='Lower')
plt.plot(final_data[final_data['buy_signal'] == 1].index, 
         final_data['Close'][final_data['buy_signal'] == 1],'o', markersize=3, color='r', label='Buy')
plt.plot(final_data[final_data['sell_signal'] == 1].index, 
         final_data['Close'][final_data['sell_signal'] == 1],'o', markersize=3, color='b', label='Sell')
plt.legend()
plt.show()