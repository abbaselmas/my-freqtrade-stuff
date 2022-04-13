import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from pandas import DataFrame, Series


"""
Misc. Helper Functions
"""
def same_length(bigger, shorter):
    return np.concatenate((np.full((bigger.shape[0] - shorter.shape[0]), np.nan), shorter))

"""
Maths
"""
def linear_growth(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    """
    Simple linear growth function. Grows from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (end - start) / (end_time - start_time)

    return min(end, start + (rate * time))

def linear_decay(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    """
    Simple linear decay function. Decays from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (start - end) / (end_time - start_time)

    return max(end, start - (rate * time))

"""
TA Indicators
"""

def zema(dataframe, period, field='close'):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/overlap_studies.py#L79
    Modified slightly to use ta.EMA instead of technical ema
    """
    df = dataframe.copy()

    df['ema1'] = ta.EMA(df[field], timeperiod=period)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=period)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']

    return df['zema']

def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]

def mastreak(dataframe: DataFrame, period: int = 4, field='close') -> Series:
    """
    MA Streak
    Port of: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
    """    
    df = dataframe.copy()

    avgval = zema(df, period, field)

    arr = np.diff(avgval)
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                    -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))

    res = same_length(df['close'], streak)

    return res

def pcc(dataframe: DataFrame, period: int = 20, mult: int = 2):
    """
    Percent Change Channel
    PCC is like KC unless it uses percentage changes in price to set channel distance.
    https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
    """
    df = dataframe.copy()

    df['previous_close'] = df['close'].shift()

    df['close_change'] = (df['close'] - df['previous_close']) / df['previous_close'] * 100
    df['high_change'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_change'] = (df['low'] - df['close']) / df['close'] * 100

    df['delta'] = df['high_change'] - df['low_change']

    mid = zema(df, period, 'close_change')
    rangema = zema(df, period, 'delta')

    upper = mid + rangema * mult
    lower = mid - rangema * mult

    return upper, rangema, lower

def SSLChannels(dataframe, length=10, mode='sma'):
    """
    Source: https://www.tradingview.com/script/xzIoaIJC-SSL-channel/
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L1025
    Usage:
        dataframe['sslDown'], dataframe['sslUp'] = SSLChannels(dataframe, 10)
    """
    if mode not in ('sma'):
        raise ValueError(f"Mode {mode} not supported yet")

    df = dataframe.copy()

    if mode == 'sma':
        df['smaHigh'] = df['high'].rolling(length).mean()
        df['smaLow'] = df['low'].rolling(length).mean()

    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1,
                         np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()

    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

def WaveTrend(dataframe, chlen=10, avg=21, smalen=4):
    """
    WaveTrend Ocillator by LazyBear
    https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
    """
    df = dataframe.copy()

    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['esa'] = ta.EMA(df['hlc3'], timeperiod=chlen)
    df['d'] = ta.EMA((df['hlc3'] - df['esa']).abs(), timeperiod=chlen)
    df['ci'] = (df['hlc3'] - df['esa']) / (0.015 * df['d'])
    df['tci'] = ta.EMA(df['ci'], timeperiod=avg)

    df['wt1'] = df['tci']
    df['wt2'] = ta.SMA(df['wt1'], timeperiod=smalen)
    df['wt1-wt2'] = df['wt1'] - df['wt2']

    return df['wt1'], df['wt2']

def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b*b*b
    c2 = 3*b*b+3*b*b*b
    c3 = -6*b*b-3*b-3*b*b*b
    c4 = 1+3*b+b*b*b+3*b*b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc

def Kalman(dataframe: DataFrame):
    """
    One-dimensional steady-state Kalman filter. It is obtained from the
    alpha filter.
    """
    df = dataframe.copy()
    src = dataframe['close']
    tr = ta.TRANGE(dataframe)
    df['value1'] = 0.2 * (src - src.shift(1, fill_value=0)) 
    df['value1'] = df['value1'].rolling(2, min_periods=1).apply(lambda x: x.iloc[0] + 0.8 * x.iloc[-1])
    df['value2'] = 0.1 * (tr)
    df['value2'] = df['value2'].rolling(2, min_periods=1).apply(lambda x: x.iloc[0] + 0.8 * x.iloc[-1])

    L = abs(df['value1'] / df['value2'])
    # Alpha filter
    alpha = (-L ** 2.0 + np.sqrt(L ** 4.0 + 16.0 * L ** 2.0)) / 8.0

    df['value3'] =  alpha * src
    df['value3'] = (1 - alpha) * df['value3'].shift(1, fill_value=0)
    
    return df['value3'].fillna(0)

# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma1_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif

# PMAX
def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx

# smoothed Heiken Ashi
def HA(dataframe, smoothing=None):
    df = dataframe.copy()

    df['HA_Close']=(df['open'] + df['high'] + df['low'] + df['close'])/4

    df.reset_index(inplace=True)

    ha_open = [ (df['open'][0] + df['close'][0]) / 2 ]
    [ ha_open.append((ha_open[i] + df['HA_Close'].values[i]) / 2) for i in range(0, len(df)-1) ]
    df['HA_Open'] = ha_open

    df.set_index('index', inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','high']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','low']].min(axis=1)

    if smoothing is not None:
        sml = abs(int(smoothing))
        if sml > 0:
            df['Smooth_HA_O']=ta.EMA(df['HA_Open'], sml)
            df['Smooth_HA_C']=ta.EMA(df['HA_Close'], sml)
            df['Smooth_HA_H']=ta.EMA(df['HA_High'], sml)
            df['Smooth_HA_L']=ta.EMA(df['HA_Low'], sml)
            
    return df

def pump_warning(dataframe, perc=15):
    df = dataframe.copy()    
    df["change"] = df["high"] - df["low"]
    df["test1"] = (df["close"] > df["open"])
    df["test2"] = ((df["change"]/df["low"]) > (perc/100))
    df["result"] = (df["test1"] & df["test2"]).astype('int')
    return df['result']

def tv_wma(dataframe, length = 9, field="close") -> DataFrame:
    """
    Source: Tradingview "Moving Average Weighted"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : WMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_wma'
    """

    norm = 0
    sum = 0

    for i in range(1, length - 1):
        weight = (length - i) * length
        norm = norm + weight
        sum = sum + dataframe[field].shift(i) * weight

    dataframe["tv_wma"] = (sum / norm) if norm > 0 else 0
    return dataframe["tv_wma"]

def tv_hma(dataframe, length = 9, field="close") -> DataFrame:
    """
    Source: Tradingview "Hull Moving Average"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : HMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_hma'
    """

    dataframe["h"] = 2 * tv_wma(dataframe, math.floor(length / 2), field) - tv_wma(dataframe, length, field)

    dataframe["tv_hma"] = tv_wma(dataframe, math.floor(math.sqrt(length)), "h")
    # dataframe.drop("h", inplace=True, axis=1)

    return dataframe["tv_hma"]

def normalize(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

# VWAP bands
def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def top_percent_change(dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price
        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')

def zlema2(dataframe, fast):
    df = dataframe.copy()
    zema1=ta.EMA(df['close'], fast)
    zema2=ta.EMA(zema1, fast)
    d1=zema1-zema2
    df['zlema2']=zema1+d1
    return df['zlema2']

"""
Moving Average Cross
Port of: https://www.tradingview.com/script/PcWAuplI-Moving-Average-Cross/
"""
def mac(self, dataframe: DataFrame, fast: int = 20, slow: int = 50) -> Series:

    dataframe = dataframe.copy()

    # Fast MAs
    upper_fast = ta.EMA(dataframe['high'], timeperiod=fast)
    lower_fast = ta.EMA(dataframe['low'], timeperiod=fast)

    # Slow MAs
    upper_slow = ta.EMA(dataframe['high'], timeperiod=slow)
    lower_slow = ta.EMA(dataframe['low'], timeperiod=slow)

    # Crosses
    crosses_lf_us = qtpylib.crossed_above(lower_fast, upper_slow) | qtpylib.crossed_below(lower_fast, upper_slow)
    crosses_uf_ls = qtpylib.crossed_above(upper_fast, lower_slow) | qtpylib.crossed_below(upper_fast, lower_slow)

    dir_1 = np.where(crosses_lf_us, 1, np.nan)
    dir_2 = np.where(crosses_uf_ls, -1, np.nan)

    dir = np.where(dir_1 == 1, dir_1, np.nan)
    dir = np.where(dir_2 == -1, dir_2, dir_1)

    res = Series(dir).fillna(method='ffill').to_numpy()

    return res

"""
MA Streak
Port of: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
"""
def ma_streak(self, dataframe: DataFrame, period: int = 4, source_type='close') -> Series:
    
    dataframe = dataframe.copy()

    avgval = self.zlema(dataframe[source_type], period)

    arr = np.diff(avgval)
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                    -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))

    res = np.concatenate((np.full((dataframe.shape[0] - streak.shape[0]), np.nan), streak))

    return res

"""
Percent Change Channel
PCC is like KC unless it uses percentage changes in price to set channel distance.
https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
"""
def pcc(self, dataframe: DataFrame, period: int = 20, mult: int = 2):

    PercentChangeChannel = namedtuple('PercentChangeChannel', ['upperband', 'middleband', 'lowerband'])

    dataframe = dataframe.copy()

    close = dataframe['close']
    previous_close = close.shift()
    low = dataframe['low']
    high = dataframe['high']

    close_change = (close - previous_close) / previous_close * 100
    high_change = (high - close) / close * 100
    low_change = (low - close) / close * 100

    mid = self.zlema(close_change, period)
    rangema = self.zlema(high_change - low_change, period)

    upper = mid + rangema * mult
    lower = mid - rangema * mult

    return PercentChangeChannel(upper, rangema, lower)

def zlema(self, series: Series, period):
    ema1 = ta.EMA(series, period)
    ema2 = ta.EMA(ema1, period)
    d = ema1 - ema2
    zlema = ema1 + d
    return zlema

def getHigherLows(data: np.array, order=5, K=2):
    """
    Finds consecutive higher lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be higher.
    """
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are higher than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] < lows[i-1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

def getLowerHighs(data: np.array, order=5, K=2):
    """
    Finds consecutive lower highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be lower.
    """
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are lower than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] > highs[i-1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

def getHigherHighs(data: np.array, order=5, K=2):
    """
    Finds consecutive higher highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be higher.
    """
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=5)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are higher than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] < highs[i-1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

def getLowerLows(data: np.array, order=5, K=2):
    """
    Finds consecutive lower lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be lower.
    """
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are lower than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] > lows[i-1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

def getHHIndex(data: np.array, order=5, K=2):
    extrema = getHigherHighs(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[np.where(idx<len(data))]

def getLHIndex(data: np.array, order=5, K=2):
    extrema = getLowerHighs(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[np.where(idx<len(data))]

def getLLIndex(data: np.array, order=5, K=2):
    extrema = getLowerLows(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[np.where(idx<len(data))]

def getHLIndex(data: np.array, order=5, K=2):
    extrema = getHigherLows(data, order, K)
    idx = np.array([i[-1] + order for i in extrema])
    return idx[np.where(idx<len(data))]

def getPeaks(data, key='close', order=5, K=2):
    vals = data[key].values
    hh_idx = getHHIndex(vals, order, K)
    lh_idx = getLHIndex(vals, order, K)
    ll_idx = getLLIndex(vals, order, K)
    hl_idx = getHLIndex(vals, order, K)
    data[f'{key}_highs'] = np.nan
    data[f'{key}_highs'][hh_idx] = 1
    data[f'{key}_highs'][lh_idx] = -1
    data[f'{key}_highs'] = data[f'{key}_highs'].ffill().fillna(0)
    data[f'{key}_lows'] = np.nan
    data[f'{key}_lows'][ll_idx] = 1
    data[f'{key}_lows'][hl_idx] = -1
    data[f'{key}_lows'] = data[f'{key}_highs'].ffill().fillna(0)
    return data

# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma

# Modified Elder Ray Index
def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend

def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.
        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def vmap_b(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']