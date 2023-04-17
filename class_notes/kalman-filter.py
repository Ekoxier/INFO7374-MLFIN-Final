from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import poly1d
from datetime import datetime
import matplotlib.pyplot as plt

tsla_df = yf.download('TSLA',
                      start='2014-01-01',
                      end='2019-12-31',
                      progress=False)
df = tsla_df[['Adj Close']]
kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance = 1,
                  transition_covariance = 0.0001)
mean, cov = kf.filter(df['Adj Close'].values)
mean, std = mean.squeeze(), np.std(cov.squeeze())
plt.figure(figsize=(12,6))
plt.plot(df['Adj Close'].values - mean, 'red')
plt.title("Kalman filtered price fluctuation")
plt.ylabel("Deviation from the mean ($)")
plt.xlabel("Days")
