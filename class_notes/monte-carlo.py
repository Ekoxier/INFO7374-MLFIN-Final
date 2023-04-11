# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 22:08:38 2018

MONTE CARLO

@author: Yizhen Zhao
"""
"Data"
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_datareader import DataReader
from datetime import datetime
import yfinance as yf

start_date = datetime(2021,1,1)
end_date = datetime(2021,4,30)

AAPL = yf.download('AAPL',start_date ,end_date)
X = AAPL['Adj Close'].values
mu = np.mean(X)
se = np.std(X)
print('SAMPLE MEAN',mu)
print('SAMPLE STD',se)
sns.kdeplot(data=X, linewidth=4)

T = X.shape[0]
M = 1000
mu_mc = np.zeros(M)
se_mc = np.zeros(M)
t_stat_mc = np.zeros(M)
y_mc = np.zeros([T, M])
y_mc_std = np.zeros(M)
for i in range(0, M):
    # simulate historical apple price (doesn't have to be true)
    y_mc[:,i] = mu + se*np.random.normal(0,4,T)
    mu_mc[i] = np.mean(y_mc[:,i])
    se_mc[i] = np.std(y_mc[:,i])/np.sqrt(T) # std of mu_mc
    y_mc_std[i] = np.std(y_mc[:,i])

mu_mc = np.sort(mu_mc)
se_mc = np.sort(se_mc)
y_mc_std = np.sort(y_mc_std)
print("confidence interval of mu_mc:", mu_mc[25], mu_mc[975])
print("confidence interval of std(mu_mc):", se_mc[25], se_mc[975])
print("confidence interval of std(y_mc):", se_mc[25]*np.sqrt(T), se_mc[975]*np.sqrt(T))
print("confidence interval of std(y_mc):", y_mc_std[25], y_mc_std[975])
Y_MC = np.mean(y_mc, axis = 1)
sns.kdeplot(data=Y_MC,linewidth=4)

