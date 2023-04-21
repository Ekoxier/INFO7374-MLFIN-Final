import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
import getFamaFrenchFactors as gff
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
import os
# importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout


print("ww")
# Define the ticker symbol for Tesla
ticker = 'TSLA'

# Define the start and end dates for the data
start_date = '2019-01-01'
end_date = '2022-01-01'

# Download the data from Yahoo Finance
tsla = yf.download(ticker, start=start_date, end=end_date)


# Import ADS_INDEX
from datetime import datetime
date_parser = lambda x: datetime.strptime(x, '%Y:%m:%d')
ads_data = pd.read_excel(f'{os.getcwd()}/Project/ads_index_010622.xlsx', parse_dates=['Date'], date_parser=date_parser)
new_ads_data = ads_data[['Date', 'ADS_INDEX_010622']].set_index('Date')

# Calculate OBV
tsla['daily_return'] = tsla['Adj Close'].pct_change()
tsla['direction'] = np.where(tsla['daily_return'] >= 0, 1, -1)
tsla['direction'][0] = 0
tsla['vol_adjusted'] = tsla['Volume'] * tsla['direction']
tsla['OBV'] = tsla['vol_adjusted'].cumsum()


sp500 = yf.download('^GSPC', start=start_date, end=end_date)
aapl = yf.download('AAPL', start=start_date, end=end_date)
amzn = yf.download('AMZN', start=start_date, end=end_date)
goog = yf.download('GOOG', start=start_date, end=end_date)
cma = yf.download('CMA', start=start_date, end=end_date)
btc = yf.download('BTC-USD', start=start_date, end=end_date)
eth = yf.download('ETH-USD', start=start_date, end=end_date)
xrp = yf.download('XRP-USD', start=start_date, end=end_date)
ltc = yf.download('LTC-USD', start=start_date, end=end_date)
ada = yf.download('ADA-USD', start=start_date, end=end_date)
vix = yf.download('^VIX', start=start_date, end=end_date)

# Calculate additional features
tsla['mom_5_20'] = (tsla['Close'] / tsla['Close'].shift(5)) - 1
tsla['mom_20_100'] = (tsla['Close'] / tsla['Close'].shift(20)) - 1
tsla['mom_60_200'] = (tsla['Close'] / tsla['Close'].shift(60)) - 1

bkcn_df = yf.download("BKCN", start_date, end_date)
shsz300_df = yf.download("000300.SS", start_date, end_date)
rsi_indicator = RSIIndicator(close=tsla['Adj Close'], window=14)
macd_indicator = MACD(close=tsla['Adj Close'])

tsla['rsi'] = rsi_indicator.rsi()
tsla['macd'] = macd_indicator.macd()

print(shsz300_df.head())
print(bkcn_df.head())

# Calculate Fama French 3 factors
ff_data = gff.famaFrench3Factor(frequency='m') 
ff_data.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
ff_data.set_index('Date',inplace=True)
ff_data = ff_data.resample('D').interpolate()




# Reset index and convert all dates to same timezones, so they become mergable
for x in [sp500,aapl,amzn,goog,tsla,cma,btc,eth,xrp,ltc,ada,ff_data,vix,new_ads_data,shsz300_df,bkcn_df]:
    x.reset_index(inplace=True)
    x['Date'] =  pd.to_datetime(x['Date']).dt.date

tsla = ff_data.merge(tsla,on='Date')
tsla = new_ads_data.merge(tsla,on='Date')
    
## CORRECTION - Need to merge FF data according to date here

df_regressor = pd.DataFrame({
    'SP_500_Adj_Close':sp500['Adj Close'].shift(1),
    'AAPL_Adj_Close':aapl['Adj Close'].shift(1),
    'AMZN_Adj_Close':amzn['Adj Close'].shift(1),
    'GOOG_Adj_Close':goog['Adj Close'].shift(1),
    'CMA_Adj_Close':cma['Adj Close'].shift(1),
    'BTC_Adj_Close':btc['Adj Close'].shift(1),
    'ETH_Adj_Close':eth['Adj Close'].shift(1),
    'XRP_Adj_Close':xrp['Adj Close'].shift(1),
    'LTC_Adj_Close':ltc['Adj Close'].shift(1),
    'ADA_Adj_Close':ada['Adj Close'].shift(1),
    'Fama_French_Mkt_RF':tsla['Mkt-RF'].shift(1),
    'Fama_French_SMB' : tsla['SMB'].shift(1),
    'Fama_French_HML' : tsla['HML'].shift(1),
    'OBV': tsla['OBV'].shift(1),
    'mom_5_20':  tsla['mom_5_20'].shift(1),
    'mom_20_100':  tsla['mom_20_100'].shift(1),
    'mom_60_200':  tsla['mom_60_200'].shift(1),
     'TSLA_CLOSE': tsla['Adj Close'],
    'VIX_IDX':vix['Adj Close'].shift(1),
    'avg_close_20_days_': tsla['Adj Close'].rolling(window=20).mean().shift(1),
    'avg_Close_50_days':tsla['Adj Close'].rolling(window=50).mean().shift(1),
    'ADS_INDEX': tsla['ADS_INDEX_010622'].shift(1),
#     'bkcn_Adj_Close':bkcn_df['Adj Close'].shift(1),
    'shsz300_df':shsz300_df['Adj Close'].shift(1),
    
    'TSLA_RSI': tsla['rsi'].shift(1),
    'TSLA_MACD':tsla['macd'].shift(1)
})

# Remove any rows with missing data
df_regressor.dropna(inplace=True)
print(df_regressor.head)
print(df_regressor.shape)


scaler = StandardScaler()
X = scaler.fit_transform(df_regressor)
# Split the data into features (X) and target (y)
y = df_regressor['TSLA_CLOSE']
df_regressor = df_regressor.drop(columns=['TSLA_CLOSE'],axis=1)
X = df_regressor

# initializing the RNN
regressor = Sequential()

print("wwww")
# adding first RNN layer and dropout regulatization
regressor.add(
    SimpleRNN(units = 50, 
              activation = "tanh", 
              return_sequences = True, 
              input_shape = (X_train.shape[1],1))
             )

regressor.add(
    Dropout(0.2)
             )


# adding second RNN layer and dropout regulatization

regressor.add(
    SimpleRNN(units = 50, 
              activation = "tanh", 
              return_sequences = True)
             )

regressor.add(
    Dropout(0.2)
             )

# adding third RNN layer and dropout regulatization

regressor.add(
    SimpleRNN(units = 50, 
              activation = "tanh", 
              return_sequences = True)
             )

regressor.add(
    Dropout(0.2)
             )

# adding fourth RNN layer and dropout regulatization

regressor.add(
    SimpleRNN(units = 50)
             )

regressor.add(
    Dropout(0.2)
             )

# adding the output layer
regressor.add(Dense(units = 1))

# compiling RNN
regressor.compile(
    optimizer = "adam", 
    loss = "mean_squared_error",
    metrics = ["accuracy"])

# fitting the RNN
history = regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)