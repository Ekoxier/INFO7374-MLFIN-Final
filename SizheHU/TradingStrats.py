
import numpy as np

'BUY HOLD'
def trading_rule_1(p):
    T = len(p)
    signal = np.zeros(T);
    signal[0] = 1
    signal[T-1] = -1
    return signal

'LONG SHORT'
def trading_rule_2(p_hat, p):
    T = len(p)
    signal = np.zeros(T)
    for t in range(T):
        if(t == 0):
            continue
        if p_hat[t-1] > p[t-1] and p_hat[t] < p[t]:
            # FORCAST > OPEN: LONG
            signal[t] = 1
        elif p_hat[t-1] < p[t-1] and p_hat[t] > p[t]:
            # FORECAST < OPEN: SHORT
            signal[t] = -1
    return signal

'DAY TRADE'
def trading_rule_3(p_hat, p):
    T = len(p)
    signal = np.zeros(T)
    for t in range(T):
        if p_hat(t)> p(t):
        # FORCAST > OPEN: LONG
            signal[t] = 1
        elif p_hat(t)< p(t): 
            #FORECAST < OPEN: SHORT
            signal[t] = -1
    return signal