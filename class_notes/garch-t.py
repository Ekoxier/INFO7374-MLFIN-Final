# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 09:04:31 2019

@author: yizhe
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:43:49 2019
GARCH-t MODEL
@author: Yizhen Zhao
"""
import pandas as pd
import numpy as np
import scipy.special as ss
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf


def GARCH_t(Y):
 "Initialize Params:"
 mu = param0[0]
 omega = param0[1]
 alpha = param0[2]
 beta = param0[3]
 nv = param0[4]
 
 T = Y.shape[0]
 GARCH_t = np.zeros(T) 
 sigma2 = np.zeros(T)   
 F = np.zeros(T)   
 v = np.zeros(T)   
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    F[t] = Y[t] - mu-np.sqrt(sigma2[t])*np.random.standard_t(nv,1)
    v[t] = sigma2[t]
    GARCH_t[t] = np.log(ss.gamma((nv+1)/2))-np.log(np.sqrt(nv*np.pi))-\
                    np.log(ss.gamma(nv/2))-((nv+1)/2)*np.log(1+((F[t]**2)/v[t])/nv)     
 
 Likelihood = np.sum(GARCH_t[1:-1])  
 return Likelihood


def GARCH_PROD_t(params, Y0, T):
 mu = params[0]
 omega = params[1]
 alpha = params[2]
 beta = params[3]
 nv = params[4]
 Y = np.zeros(T)  
 sigma2 = np.zeros(T)
 Y[0] = Y0
 sigma2[0] = 0.0001
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    Y[t] = mu+np.sqrt(sigma2[t])*np.random.standard_t(nv,1) 
 return Y    


# 2. Real Data
TSLA = yf.download('TSLA', datetime(2021,1,1), datetime(2021,6,30))
# Y = TSLA['Adj Close'].values
Y = np.diff(np.log(TSLA['Adj Close'].values))
T = Y.shape[0]

param0 = np.array([np.mean(Y), np.var(Y)/10, 0.3, 0.3, 10])
param_star = minimize(GARCH_t, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
Y_GARCH_t = GARCH_PROD_t(param_star.x, Y[0], T)
timevec = np.linspace(1,T,T)
plt.plot(timevec, Y,'b',timevec, Y_GARCH_t,'r')

RMSE = np.sqrt(np.mean((Y_GARCH_t - Y)**2))
print('RMSE values is:', RMSE)