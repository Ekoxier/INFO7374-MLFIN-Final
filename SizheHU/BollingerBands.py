import plotly.io as pio
import plotly.graph_objects as go

import numpy as np
import pandas as pd
# from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf

start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 1, 1)
TSLA_data = yf.download('TSLA', start_date, end_date)
TSLA_close = TSLA_data['Adj Close']
TSLA_close = TSLA_close.rename(columns={'Adj Close': 'Close'})
TSLA_sma = TSLA_data['Adj Close'].rolling(window=20).mean().dropna()
TSLA_rstd = TSLA_data['Adj Close'].rolling(window=20).std().dropna()

upper_band = TSLA_sma + 2 * TSLA_rstd
lower_band = TSLA_sma - 2 * TSLA_rstd

upper_band = upper_band.rename(columns={'Adj Close': 'upper'})
lower_band = lower_band.rename(columns={'Adj Close': 'lower'})
bb = TSLA_close.join(upper_band).join(lower_band)
bb = bb.dropna()

buyers = bb[bb['Close'] <= bb['lower']]
sellers = bb[bb['Close'] >= bb['upper']]

# Plotting
if_plot = True
if(if_plot):
    plt.scatter(x=lower_band.index, 
                            y=lower_band['lower'], 
                            name='Lower Band', 
                            line_color='rgba(173,204,255,0.2)'
                            )
    plt.scatter(x=upper_band.index, 
                            y=upper_band['upper'], 
                            name='Upper Band', 
                            fill='tonexty', 
                            fillcolor='rgba(173,204,255,0.2)', 
                            line_color='rgba(173,204,255,0.2)'
                            )
    plt.scatter(x=df.index, 
                            y=df['Close'], 
                            name='Close', 
                            line_color='#636EFA'
                            )
    pio.templates.default = "plotly_dark"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lower_band.index, 
                            y=lower_band['lower'], 
                            name='Lower Band', 
                            line_color='rgba(173,204,255,0.2)'
                            ))
    fig.add_trace(go.Scatter(x=upper_band.index, 
                            y=upper_band['upper'], 
                            name='Upper Band', 
                            fill='tonexty', 
                            fillcolor='rgba(173,204,255,0.2)', 
                            line_color='rgba(173,204,255,0.2)'
                            ))
    fig.add_trace(go.Scatter(x=df.index, 
                            y=df['Close'], 
                            name='Close', 
                            line_color='#636EFA'
                            ))
    fig.add_trace(go.Scatter(x=sma.index, 
                            y=sma['Close'], 
                            name='SMA', 
                            line_color='#FECB52'
                            ))
    fig.add_trace(go.Scatter(x=buyers.index, 
                            y=buyers['Close'], 
                            name='Buyers', 
                            mode='markers',
                            marker=dict(
                                color='#00CC96',
                                size=10,
                                )
                            ))
    fig.add_trace(go.Scatter(x=sellers.index, 
                            y=sellers['Close'], 
                            name='Sellers', 
                            mode='markers', 
                            marker=dict(
                                color='#EF553B',
                                size=10,
                                )
                            ))
    fig.show()


current_value = bb['Close'][-1]
upper_bound = bb['upper'][-1]
lower_bound = bb['lower'][-1]
if current_value > upper_bound:
    message = 'Sell indicator'
elif current_value < lower_bound:
    message = 'Buy indicator'
else:
    message = False