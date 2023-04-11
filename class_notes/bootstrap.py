# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 20:45:32 2018

BOOTSTRAP

@author: Yizhen Zhao
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_datareader import DataReader
from datetime import datetime
import scipy.stats as ss
import yfinance as yf

start_date = datetime(2021,1,1)
end_date = datetime(2021,4,30)

AAPL = yf.download('AAPL',start_date ,end_date)
AAPL['Adj Close'].describe() # summary statistics
X = AAPL['Adj Close'].values
print('SAMPLE MEAN',np.mean(X))
print('SAMPLE STD',np.std(X))
print('SAMPLE SKEWNESS: ',ss.skew(X, axis=0, bias=True))
print('SAMPLE KURTOSIS: ',ss.kurtosis(X, axis=0, bias=True)+3)
# NOTE: 
# Calculations are corrected for statistical bias, if bias set to False.
# Python reports excess kurtosis, by -3 (kurt of normal dist)
# CASE 1: A distribution with a negative kurtosis value indicates that 
# the distribution has lighter tails than the normal distribution. 
# CASE 2: If a distribution has positive kurtosis, it is said to be leptokurtic, 
# which means that it has a sharper peak and heavier tails compared to a normal distribution.

# Empirical Distribution
sns.kdeplot(data=X,linewidth=4)

"Bootstrap"
T= X.shape[0]
B = 1000 # 5000, 100000 [250 9750]
mu_boot = np.zeros(B)
se_boot = np.zeros(B)
x_boot_std = np.zeros(B)
for i in range(0, B):
     x_boot = X[np.random.choice(T,T)]
     mu_boot[i] = np.mean(x_boot)
     se_boot[i] = np.std(x_boot)/np.sqrt(T) # std of mu_boot
     x_boot_std[i] = np.std(x_boot) # std of x_boot
     """"CLT: std(x_boost) = sqrt(T)*std(mu_boot)"""
mu_boot = np.sort(mu_boot)
se_boot = np.sort(se_boot)
xboot_std = np.sort(x_boot_std)

print("95% confidence interval of mu_boot:", mu_boot[25], mu_boot[975])
print("95% confidence interval of std_boot of mu :", se_boot[25], se_boot[975])
print("95% confidence interval of APPL std:", se_boot[25]*np.sqrt(T), se_boot[975]*np.sqrt(T))
print("95% confidence interval of APPL std:", xboot_std[25], xboot_std[975])