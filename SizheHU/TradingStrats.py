import numpy as np

def trading_strat_day_trade(Y_predicted,Y):
    T = Y.size()
    signal =  np.zeros(T)
    for t in range(0, T):
        if Y_predicted[t] > Y[t-1]:
            signal[t] = 1  # long signal
        elif Y_predicted[t] < Y[t-1]:
            signal[t] = -1  # short signal
    return signal


def trading_strat_long_short(Y_predicted,Y):
    T = Y.size()
    signal =  np.zeros(T)
    position = np.zeros(T)
    for t in range(0, T):
        if Y_predicted[t] > Y[t]:
            signal[t] = 1  
        elif Y_predicted[t] < Y[t]:
            signal[t] = -1  
    for t in range(0, T):
        if t == 0:
            position[t] = signal[t]
        elif signal[t] != signal[t-1]: 
            # take the first long/short signal as position
            position[t] = signal[t] 
    return position