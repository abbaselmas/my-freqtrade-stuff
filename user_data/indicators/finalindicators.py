
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from audioop import mul
from curses import window
from lib2to3.pgen2.pgen import DFAState
from re import match
from signal import signal
from tokenize import maybe
from unicodedata import decimal
import warnings
import sys
from certifi import where
from h11 import Data
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from numpy import nan as npNaN
from pandas import Series
from pandas.core.window.rolling import Window

import talib.abstract as ta  # noqa
from talib import MA_Type
import freqtrade.vendor.qtpylib.indicators as qtpylib
import freqtrade.vendor.qtpylib as qt
import technical.indicators as indicators
from numpy import cos as npCos
from numpy import exp as npExp
from numpy import pi as npPi
from numpy import sin as npSin
from numpy import sqrt as npSqrt

# =============================================
# check min, python version
if sys.version_info < (3, 4):
    raise SystemError("QTPyLib requires Python version >= 3.4")

# =============================================
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# =============================================


def vidya(dataframe, period=9, field='close', select=True):

    if f"vidya-ma-{period}" in dataframe.columns.values:
        return dataframe[f"vidya-ma-{period}"].to_numpy()

    dataframe['hl2'] = qtpylib.mid_price(dataframe)

    first_index = period if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()
    df = dataframe.copy()
    alpha = 2 / (period + 1)
    df["momm"] = df[field].diff()
    df["m1"] = np.where(df["momm"] > 0, df["momm"], 0.0)
    df["m2"] = np.where(df["momm"] > 0, 0.0, -df["momm"])

    df["sm1"] = df["m1"].rolling(9, min_periods=1).sum()
    df["sm2"] = df["m2"].rolling(9, min_periods=1).sum()

    df["chandeMO"] = 100 * (df["sm1"] - df["sm2"]) / (df["sm1"] + df["sm2"])
    if select:
        df["k"] = abs(df["chandeMO"]) / 100
    else:
        df["k"] = df[field].rolling(period).std()
    df.fillna(0.0, inplace=True)

    df["VIDYA"] = 0.0
    for i in range(first_index+1, len(df)):

        df["VIDYA"].iat[i] = (
            alpha * df["k"].iat[i] * df[field].iat[i]
            + (1 - alpha * df["k"].iat[i]) * df["VIDYA"].iat[i - 1]
        )

    return df["VIDYA"]


def EVWMA(dataframe: DataFrame, period: int = 20, field='close'):
    """
    The eVWMA can be looked at as an approximation to the
    average price paid per share in the last n periods.
    :period: Specifies the number of Periods used for eVWMA calculation
    """

    if f"vidya-ma-{period}" in dataframe.columns.values:
        return dataframe[f"vidya-ma-{period}"].to_numpy()

    if(field == 'hl2'):
        dataframe['hl2'] = qtpylib.mid_price(dataframe)

    vol_sum = (
        dataframe["volume"].rolling(window=period).sum()
    )  # floating shares in last N periods

    x = (vol_sum - dataframe["volume"]) / vol_sum
    y = (dataframe["volume"] * dataframe[field]) / vol_sum

    evwma = [0]

    #  evwma = (evma[-1] * (vol_sum - volume)/vol_sum) + (volume * price / vol_sum)
    for x, y in zip(x.fillna(0).iteritems(), y.iteritems()):
        if x[1] == 0 or y[1] == 0:
            evwma.append(0)
        else:
            evwma.append(evwma[-1] * x[1] + y[1])

    return evwma[1:]


def vidya_fast(dataframe, period, field='close'):

    if f"vidya-ma-{period}" in dataframe.columns.values:
        return dataframe[f"vidya-ma-{period}"].to_numpy()

    if(field == 'hl2'):
        dataframe['hl2'] = qtpylib.mid_price(dataframe)
    elif(field == 'hlc4'):
        dataframe['hlc4'] = (dataframe["high"] + dataframe["low"] +
                             dataframe["close"] + dataframe["open"]) / 4

    first_index = period if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()
    src = dataframe[field].to_numpy()

    mom = np.diff(src)
    mom = np.insert(mom, 0, 0, axis=0)
    m1 = np.where(mom > 0.0, mom, 0.0)
    m2 = np.where(mom > 0.0, 0.0, -mom)

    sm1 = rolling_sum(m1, window=9, min_periods=1)
    sm2 = rolling_sum(m2, window=9, min_periods=1)

    cmo = np.true_divide(np.subtract(sm1, sm2), np.add(sm1, sm2))
    cmo2 = np.absolute(cmo)
    cmo3 = np.nan_to_num(cmo2)
    # cmo3 = np.pad(cmo2, (src.size - cmo2.size, 0), 'constant')

    vidya = np.full_like(src, np.nan)
    alpha: float = 2 / (period + 1)

    for i in range(first_index + 1, len(vidya)):
        vcmo = 0.0 if np.isnan(cmo3[i]) else cmo3[i]
        vsrc = 0.0 if np.isnan(src[i]) else src[i]
        vidyaprev = 0.0 if np.isnan(vidya[i-1]) else vidya[i-1]
        vidya[i] = ((alpha * vcmo * vsrc) + (1 - alpha * vcmo) * vidyaprev)

    vidya = np.nan_to_num(vidya)
    return vidya


def rolling_sum(series, window=14, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).sum()
    except Exception as e:  # noqa: F841
        return pd.Series(series).rolling(window=window, min_periods=min_periods).sum()


def vma(dataframe, period=30, field='close'):

    if f"vma-{period}" in dataframe.columns.values:
        return dataframe[f"vma-{period}"].to_numpy()

    if(field == 'hl2'):
        dataframe['hl2'] = qtpylib.mid_price(dataframe)

    first_index = period if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    src = dataframe[field].to_numpy()

    k = 1.0 / period

    def calcVMA(src, k, first_index):

        iS = np.full_like(src, np.nan)
        pdmSP = 0
        mdmSP = 0
        pdiSP = 0
        mdiSP = 0

        for i in range(first_index + 1, len(src)):
            pdm = src[i] - src[i-1] if src[i] - src[i-1] >= 0 else 0.0
            mdm = src[i-1] - src[i] if src[i-1] - src[i] >= 0 else 0.0
            pdmS = (((1-k) * pdmSP) + k*pdm)
            mdmS = (((1-k) * mdmSP) + k*mdm)
            s = pdmS + mdmS
            pdi = pdmS / s
            mdi = mdmS / s
            pdiS = (((1-k) * pdiSP) + k*pdi)
            mdiS = (((1-k) * mdiSP) + k*mdi)
            d = (pdiS - mdiS) if (pdiS - mdiS) >= 0.0 else ((pdiS - mdiS) * -1)
            s1 = pdiS + mdiS
            iS[i] = (((1-k) * (0.0 if np.isnan(iS[i-1]) else iS[i-1])) + k*d/s1)
            pdmSP = pdmS
            mdmSP = mdmS
            pdiSP = pdiS
            mdiSP = mdiS
        return iS

    iS = calcVMA(src, k, first_index)
    hhv = qtpylib.rolling_min(iS, period, min_periods=1)
    llv = qtpylib.rolling_max(iS, period, min_periods=1)
    d1 = np.subtract(hhv, llv)
    vI = np.true_divide(np.subtract(iS, llv), d1)
    vma = np.full_like(src, np.nan)
    for i in range(first_index+1, len(vma)):
        vma[i] = ((1-(k * vI[i])) * (0.0 if np.isnan(vma[i-1]) else vma[i-1]) + k*vI[i]*src[i])

    return vma


def ott(dataframe: DataFrame, window, percent, field='close', matype='var') -> DataFrame:

    df = dataframe.copy()

    if matype == 'var':
        df['Var'] = vidya(df, window, field=field)
    elif matype == 'evwma':
        df['Var'] == EVWMA(df, window, field)
    else:
        df['Var'] = vidya(df, window, field=field)

    df['fark'] = df['Var'] * percent * 0.01
    df['newlongstop'] = df['Var'] - df['fark']
    df['newshortstop'] = df['Var'] + df['fark']
    df['longstop'] = 0.0
    df['shortstop'] = 999999999999999999

    def maxlongstop():
        df.loc[(df['newlongstop'] > df['longstop'].shift(1)), 'longstop'] = df['newlongstop']
        df.loc[(df['longstop'].shift(1) > df['newlongstop']), 'longstop'] = df['longstop'].shift(1)

        return df['longstop']

    def minshortstop():
        df.loc[(df['newshortstop'] < df['shortstop'].shift(1)), 'shortstop'] = df['newshortstop']
        df.loc[(df['shortstop'].shift(1) < df['newshortstop']),
               'shortstop'] = df['shortstop'].shift(1)

        return df['shortstop']

    df['longstop'] = np.where(df['Var'] > df['longstop'].shift(1), maxlongstop(), df['newlongstop'])
    df['shortstop'] = np.where(df['Var'] < df['shortstop'].shift(1),
                               minshortstop(), df['newshortstop'])
    df['xlongstop'] = np.where((df['Var'].shift(1) > df['longstop'].shift(1))
                               & (df['Var'] < df['longstop'].shift(1)), 1, 0)
    df['xshortstop'] = np.where((df['Var'].shift(1) < df['shortstop'].shift(1))
                                & (df['Var'] > df['shortstop'].shift(1)), 1, 0)
    df['trend'] = 0
    df['dir'] = 0

    df['trend'] = np.where((df['xshortstop'] == 1), 1, np.where(
        df['xlongstop'] == 1, -1, df['trend'].shift(1)))
    df['dir'] = np.where(df['xshortstop'] == 1, 1, np.where(
        df['xlongstop'] == 1, -1, df['dir'].shift(1).fillna(1)))

    df['MT'] = np.where(df['dir'] == 1, df['longstop'], df['shortstop'])
    df['OTT'] = np.where(df['Var'] > df['MT'], (df['MT'] * (200 + percent) /
                         200), (df['MT'] * (200 - percent) / 200))
    df['OTT'] = df['OTT'].shift(2)

    return df['Var'], df['OTT']


def ott_fast(dataframe: DataFrame, window, percent, field='close', matype='var'):

    df = dataframe.copy()

    if(field == 'hl2'):
        dataframe['hl2'] = qtpylib.mid_price(dataframe)
    elif(field == 'hlc4'):
        dataframe['hlc4'] = (dataframe["high"] + dataframe["low"] +
                             dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, window, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop

        return np.concatenate((np.full(2, 0), ott[:-2]))

    return var, calcOtt(var, percent, window, first_index)



def ott_parallel(dataframe: DataFrame, window, percent, field='close', matype='var'):

    df = dataframe.copy()

    if(field == 'hl2'):
        dataframe['hl2'] = qtpylib.mid_price(dataframe)
    # elif(field == 'hlc4'):
        # dataframe['hlc4'] = (dataframe["high"] + dataframe["low"] + dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, window, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop

        return np.concatenate((np.full(2, 0), ott[:-2]))

    return var, calcOtt(var, percent, window, first_index), window, percent


def ott_fast_smt(dataframe: DataFrame, window, percent, smoothing=0.0008, field='close', matype='var'):

    df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()

    if(field == 'hl2'):
        df['hl2'] = qtpylib.mid_price(dataframe)
    elif(field == 'hlc4'):
        df['hlc4'] = (dataframe["high"] + dataframe["low"] +
                      dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, window, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop
        return np.concatenate((np.full(2, 0), ott[:-2]))

    ott_ret = calcOtt(var, percent, window, first_index)
    ott_up = ott_ret * (1.00 + smoothing)
    ott_down = ott_ret * (1.00 - smoothing)
    return var, ott_up, ott_down


def ott_parallel_smt(dataframe: DataFrame, window: int, percent: float, smoothing: float = 0.0008, field='close', matype='var'):

    df = dataframe.copy()

    if(field == 'hl2'):
        dataframe['hl2'] = qtpylib.mid_price(dataframe)
    # elif(field == 'hlc4'):
        # dataframe['hlc4'] = (dataframe["high"] + dataframe["low"] + dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop

        return np.concatenate((np.full(2, 0), ott[:-2]))

    ott_ret = calcOtt(var, percent, first_index)
    ott_up = ott_ret * (1.00 + smoothing)
    ott_down = ott_ret * (1.00 - smoothing)
    return var, ott_up, ott_down, window, percent, smoothing


def StochVar(df: DataFrame, fastk_period=520, fastd_period=350, smoothing=0.3, plus=1.0, matype='var'):

    my_df = df[['open', 'high', 'low', 'close']].copy()

    my_df['fastk'] = qtpylib.stoch(my_df, window=fastk_period, d=111, k=3, fast=True)['fast_k']
    # my_df['fastk'], fastd = ta.STOCHF(my_df['high'], my_df['low'], my_df['close'],fastk_period=fastk_period, fastd_period=111, fastd_matype=1)
    if matype == 'var':
        my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    elif matype == 'evwma':
        my_df['fastk_var'] = EVWMA(my_df, fastd_period, field='fastk')
    elif matype == 'vwap':
        my_df['fastk_var'] = qtpylib.rolling_vwap(my_df, fastd_period)
    elif matype == 'vma':
        my_df['fastk_var'] = vma(my_df, fastd_period, field='fastk')
    else:
        my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    plusval = (1000.00 * plus)
    my_df['fastk_src'] = my_df['fastk_var'] + plusval
    stosk, stott = ott_fast(my_df, 2, smoothing, field='fastk_src')
    return my_df['fastk_src'], stott


def STOTTParallel(df: DataFrame, fastk_period=520, fastd_period=350, smoothing=0.3, plus=1.0, matype='var'):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'] = qtpylib.stoch(my_df, window=fastk_period, d=111, k=3, fast=True)['fast_k']
    # my_df['fastk'], fastd = ta.STOCHF(my_df['high'], my_df['low'], my_df['close'],fastk_period=fastk_period, fastd_period=1, fastd_matype=0)
    if matype == 'var':
        my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    elif matype == 'evwma':
        my_df['fastk_var'] = EVWMA(my_df, fastd_period, field='fastk')
    elif matype == 'vwap':
        my_df['fastk_var'] = qtpylib.rolling_vwap(my_df, fastd_period)
    elif matype == 'vma':
        my_df['fastk_var'] = vma(my_df, fastd_period, field='fastk')
    else:
        my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    plusval = (1000.00 * plus)
    my_df['fastk_src'] = my_df['fastk_var'] + plusval
    stosk, stott = ott_fast(my_df, 2, smoothing, field='fastk_src')
    return (my_df['fastk'] + plusval), stott, fastk_period, fastd_period, smoothing


def StochVMA(df: DataFrame, fastk_period=520, fastd_period=350, smoothing=0.3, plus=1.0):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'], fastd = ta.STOCHF(
        my_df['high'], my_df['low'], my_df['close'], fastk_period=fastk_period, fastd_period=1, fastd_matype=0)
    my_df['fastk_var'] = vma(my_df, fastd_period, field='fastk')
    plusval = (1000.00 * plus)
    my_df['fastk_src'] = my_df['fastk_var'] + plusval
    stosk, stott = ott_fast(my_df, 2, smoothing, field='fastk_src')
    return stosk, stott


def nATR(dataframe: DataFrame, atr_period=10):

    tr_val = (qtpylib.true_range(dataframe))
    alpha = (1.0 / atr_period)
    return tr_val.ewm(alpha=alpha).mean()


def PMax(dataframe: DataFrame, atr_period=10, atr_mult=3.0, ma_length: int = 10, field='close', matype='var'):

    if matype == 'var':
        ma = vidya_fast(dataframe, ma_length, field=field)
    elif matype == 'evwma':
        ma = EVWMA(dataframe, ma_length, field)
    elif matype == 'vma':
        ma = vma(dataframe, ma_length, field)
    else:
        ma = vidya_fast(dataframe, ma_length, field=field)

    natr = nATR(dataframe, atr_period).to_numpy()

    longv = ma - (natr * atr_mult)
    shortv = ma + (natr * atr_mult)
    dir = np.ones_like(dataframe.close, dtype=np.int8)

    def calcDir(ma_length: int, dir: np.ndarray, ma: np.ndarray, short: np.ndarray, longv: np.ndarray):
        for i in range(ma_length, len(dir)):
            longv[i] = max(longv[i], longv[i-1]) if ma[i] > longv[i-1] else longv[i]
            short[i] = min(short[i], short[i-1]) if ma[i] < short[i-1] else short[i]
            if dir[i-1] == -1 and ma[i] > short[i-1]:
                dir[i] = 1
            elif dir[i-1] == 1 and ma[i] < longv[i-1]:
                dir[i] = -1
            else:
                dir[i] = dir[i-1]
        return short, longv, dir

    shortr, longr, dirr = calcDir(ma_length, dir, ma, shortv, longv)
    PMax = np.where((dirr == -1), shortr, longr)

    return ma, PMax

def PMaxSignal(dataframe: DataFrame, atr_period=10, atr_mult=3.0, ma_length: int = 10, field='close', matype='var'):

    if matype == 'var':
        ma = vidya_fast(dataframe, ma_length, field=field)
    elif matype == 'evwma':
        ma = EVWMA(dataframe, ma_length, field)
    elif matype == 'vma':
        ma = vma(dataframe, ma_length, field)
    else:
        ma = vidya_fast(dataframe, ma_length, field=field)

    natr = nATR(dataframe, atr_period).to_numpy()

    longv = ma - (natr * atr_mult)
    shortv = ma + (natr * atr_mult)
    dir = np.ones_like(dataframe.close, dtype=np.int8)

    def calcDir(ma_length: int, dir: np.ndarray, ma: np.ndarray, short: np.ndarray, longv: np.ndarray):
        for i in range(ma_length, len(dir)):
            longv[i] = max(longv[i], longv[i-1]) if ma[i] > longv[i-1] else longv[i]
            short[i] = min(short[i], short[i-1]) if ma[i] < short[i-1] else short[i]
            if dir[i-1] == -1 and ma[i] > short[i-1]:
                dir[i] = 1
            elif dir[i-1] == 1 and ma[i] < longv[i-1]:
                dir[i] = -1
            else:
                dir[i] = dir[i-1]
        return short, longv, dir

    shortr, longr, dirr = calcDir(ma_length, dir, ma, shortv, longv)
    PMax = np.where((dirr == -1), shortr, longr)
    signal = np.where(ma > PMax, 'buy', 'sell')
    return signal


def PMaxVMA(dataframe: DataFrame, atr_period=10, atr_mult=3.0, ma_length: int = 10, field='close'):
    ma = vma(dataframe, ma_length, field=field)
    natr = nATR(dataframe, atr_period).to_numpy()

    longv = ma - (natr * atr_mult)
    shortv = ma + (natr * atr_mult)
    dir = np.ones_like(dataframe.close, dtype=np.int8)

    def calcDir(ma_length: int, dir: np.ndarray, ma: np.ndarray, short: np.ndarray, longv: np.ndarray):
        for i in range(ma_length, len(dir)):
            longv[i] = max(longv[i], longv[i-1]) if ma[i] > longv[i-1] else longv[i]
            short[i] = min(short[i], short[i-1]) if ma[i] < short[i-1] else short[i]
            if dir[i-1] == -1 and ma[i] > short[i-1]:
                dir[i] = 1
            elif dir[i-1] == 1 and ma[i] < longv[i-1]:
                dir[i] = -1
            else:
                dir[i] = dir[i-1]
        return short, longv, dir

    shortr, longr, dirr = calcDir(ma_length, dir, ma, shortv, longv)
    PMax = np.where((dirr == -1), shortr, longr)

    return ma, PMax


def StochPmax(df: DataFrame, fastk_period=520, fastd_period=350, smoothing=0.3, plus=1.0):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'], fastd = ta.STOCHF(
        my_df['high'], my_df['low'], my_df['close'], fastk_period=fastk_period, fastd_period=1, fastd_matype=0)
    my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    plusval = (1000.00 * plus)
    my_df['fastk_src'] = my_df['fastk_var'] + plusval

    natr = nATR(my_df, 30).to_numpy()

    ma = vidya_fast(my_df, 2, field='fastk_src')
    longv = ma - (natr * smoothing)
    shortv = ma + (natr * smoothing)
    dir = np.ones_like(df.close, dtype=np.int8)

    def calcDir(ma_length: int, dir: np.ndarray, ma: np.ndarray, short: np.ndarray, longv: np.ndarray):
        for i in range(ma_length, len(dir)):
            longv[i] = max(longv[i], longv[i-1]) if ma[i] > longv[i-1] else longv[i]
            short[i] = min(short[i], short[i-1]) if ma[i] < short[i-1] else short[i]
            if dir[i-1] == -1 and ma[i] > short[i-1]:
                dir[i] = 1
            elif dir[i-1] == 1 and ma[i] < longv[i-1]:
                dir[i] = -1
            else:
                dir[i] = dir[i-1]
        return short, longv, dir

    shortr, longr, dirr = calcDir(fastd_period, dir, ma, shortv, longv)
    PMax = np.where((dirr == -1), shortr, longr)

    return ma, PMax


def IFStochVar(df: DataFrame, fastk_period=520, fastd_period=350):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'], fastd = ta.STOCHF(
        my_df['high'], my_df['low'], my_df['close'], fastk_period=fastk_period, fastd_period=1, fastd_matype=0)
    my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    my_df['fastk_src'] = 0.1 * (my_df['fastk_var'] - 50.0)

    # Vidya End
    return (np.exp(2 * my_df['fastk_src'])-1) / (np.exp(2 * my_df['fastk_src']) + 1)


def IFStochVarNorm(df: DataFrame, fastk_period=520, fastd_period=350):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'], fastd = ta.STOCHF(
        my_df['high'], my_df['low'], my_df['close'], fastk_period=fastk_period, fastd_period=1, fastd_matype=0)
    my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    my_df['fastk_src'] = 0.1 * (my_df['fastk_var'] - 50.0)

    return 50.0 * ((np.exp(2 * my_df['fastk_src'])-1) / (np.exp(2 * my_df['fastk_src']) + 1) + 1.0)


def IFStoch(df: DataFrame, fastk_period=520, fastd_period=350):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'], my_df['fastd'] = ta.STOCHF(
        my_df['high'], my_df['low'], my_df['close'], fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=2)
    # my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    my_df['fastk_src'] = 0.1 * (my_df['fastd'] - 50.0)

    # Vidya End
    return (np.exp(2 * my_df['fastk_src'])-1) / (np.exp(2 * my_df['fastk_src']) + 1)



def IFStochVMA(df: DataFrame, fastk_period=520, fastd_period=350):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'], fastd = ta.STOCHF(
        my_df['high'], my_df['low'], my_df['close'], fastk_period=fastk_period, fastd_period=1, fastd_matype=0)
    my_df['fastk_var'] = vma(my_df, fastd_period, field='fastk')
    my_df['fastk_src'] = 0.1 * (my_df['fastk_var'] - 50.0)

    return (np.exp(2 * my_df['fastk_src'])-1) / (np.exp(2 * my_df['fastk_src']) + 1)


def Tillson(dataframe: DataFrame, period=50, vf=0.6):
    e1 = ta.EMA(dataframe, period)
    e2 = ta.EMA(e1, period)
    e3 = ta.EMA(e2, period)
    e4 = ta.EMA(e3, period)
    e5 = ta.EMA(e4, period)
    e6 = ta.EMA(e5, period)
    c1 = -vf * vf * vf
    c2 = 3 * vf * vf + 3 * vf * vf * vf
    c3 = -6 * vf * vf - 3 * vf - 3 * vf * vf * vf
    c4 = 1 + 3 * vf + vf * vf * vf + 3 * vf * vf
    T = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    return T


def ebsw(df: DataFrame, length=None, bars=None,  **kwargs):
    """Indicator: Even Better SineWave (EBSW)"""
    # Validate arguments
    length = int(length) if length and length > 38 else 40
    bars = int(bars) if bars and bars > 0 else 10
    close = df['close']

    if close is None:
        return

    # variables
    alpha1 = HP = 0  # alpha and HighPass
    a1 = b1 = c1 = c2 = c3 = 0
    Filt = Pwr = Wave = 0

    lastClose = lastHP = 0
    FilterHist = [0, 0]   # Filter history

    # Calculate Result
    m = close.size
    result = [npNaN for _ in range(0, length - 1)] + [0]
    for i in range(length, m):
        # HighPass filter cyclic components whose periods are shorter than Duration input
        alpha1 = (1 - npSin(360 / length)) / npCos(360 / length)
        HP = 0.5 * (1 + alpha1) * (close[i] - lastClose) + alpha1 * lastHP

        # Smooth with a Super Smoother Filter from equation 3-3
        a1 = npExp(-npSqrt(2) * npPi / bars)
        b1 = 2 * a1 * npCos(npSqrt(2) * 180 / bars)
        c2 = b1
        c3 = -1 * a1 * a1
        c1 = 1 - c2 - c3
        Filt = c1 * (HP + lastHP) / 2 + c2 * FilterHist[1] + c3 * FilterHist[0]
        # Filt = float("{:.8f}".format(float(Filt))) # to fix for small scientific notations, the big ones fail

        # 3 Bar average of Wave amplitude and power
        Wave = (Filt + FilterHist[1] + FilterHist[0]) / 3
        Pwr = (Filt * Filt + FilterHist[1] * FilterHist[1] + FilterHist[0] * FilterHist[0]) / 3

        # Normalize the Average Wave to Square Root of the Average Power
        Wave = Wave / npSqrt(Pwr)

        # update storage, result
        FilterHist.append(Filt)  # append new Filt value
        FilterHist.pop(0)  # remove first element of list (left) -> updating/trim
        lastHP = HP
        lastClose = close[i]
        result.append(Wave)

    ebsw = Series(result, index=close.index)

    # Handle fills
    if "fillna" in kwargs:
        ebsw.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ebsw.fillna(method=kwargs["fill_method"], inplace=True)

    return ebsw


def hhv(df: DataFrame, period=20, shft=1):
    return df['high'].rolling(period).max().shift(shft)


def hhv_signal(df: DataFrame, period=20, shft=1):
    hhv2 = hhv(df, period, shft)
    signal = np.where(df['high'] > hhv2, 'buy', 'sell')
    return signal


def llv(df: DataFrame, period=20, shft=1):
    return df['low'].rolling(period).min().shift(shft)


def llv_signal(df: DataFrame, period=20, shft=1):
    llv2 = llv(df, period, shft)
    signal = np.where(df['low'] < llv2, 'sell', 'buy')
    return signal


def hott(df: DataFrame, period=20, shft=0, perc=0.6):
    my_df = df[['high']].copy()
    my_df['hhv'] = hhv(my_df, period=period, shft=shft)
    var, hott = ott_fast(my_df, 2, percent=perc, field='hhv')
    return hott


def hott_signal(df: DataFrame, period=20, shft=0, perc=0.6):
    hott2 = hott(df, period, shft, perc)
    signal = np.where(df['high'] > hott2, 'buy', 'sell')
    return signal


def lott(df: DataFrame, period=20, shft=0, perc=0.6):
    my_df = df[['low']].copy()
    my_df['llv'] = llv(my_df, period=period, shft=shft)
    var, lott = ott_fast(my_df, 2, percent=perc, field='llv')
    return lott


def lott_signal(df: DataFrame, period=20, shft=0, perc=0.6):
    lott2 = lott(df, period, shft, perc)
    signal = np.where(df['low'] < lott2, 'sell', 'buy')
    return signal

def ott_signal_check(dataframe: DataFrame, window, percent, field='close', matype='var'):

    if field in (['open', 'high', 'low', 'close', 'volume']):
        df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()
    else:
        df = dataframe.copy()

    if(field == 'hl2'):
        df['hl2'] = qtpylib.mid_price(dataframe)
    elif(field == 'hlc4'):
        df['hlc4'] = (dataframe["high"] + dataframe["low"] +
                      dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, window, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop

        return np.concatenate((np.full(2, 0), ott[:-2]))

    ott = calcOtt(var, percent, window, first_index)


    return var, ott

def ott_signal(dataframe: DataFrame, window, percent, field='close', matype='var'):

    if field in (['open', 'high', 'low', 'close', 'volume']):
        df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()
    else:
        df = dataframe.copy()

    if(field == 'hl2'):
        df['hl2'] = qtpylib.mid_price(dataframe)
    elif(field == 'hlc4'):
        df['hlc4'] = (dataframe["high"] + dataframe["low"] +
                      dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, window, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop

        return np.concatenate((np.full(2, 0), ott[:-2]))

    ott = calcOtt(var, percent, window, first_index)

    signal = np.where(var > ott, 'buy', 'sell')

    return signal

def rott(dataframe: DataFrame, window, percent, percent2, field='close', matype='var'):

    if field in (['open', 'high', 'low', 'close', 'volume']):
        df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()
    else:
        df = dataframe.copy()

    if(field == 'hl2'):
        df['hl2'] = qtpylib.mid_price(dataframe)
    elif(field == 'hlc4'):
        df['hlc4'] = (dataframe["high"] + dataframe["low"] +
                      dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, window, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop

        return np.concatenate((np.full(2, 0), ott[:-2]))

    ott = calcOtt(var, percent, window, first_index)
    ott2 = calcOtt(var, percent2, window, first_index)

    # rott = (ott2 + (2*ott2 - ott).shift(100))/2

    rott = (ott2 +  np.concatenate((np.full(100, np.nan), (2*ott2 - ott)[:-100])))/2

    # (
    # OTT(C,opt1,opt3)
    # +
    # REF(
    #     OTT(C,opt1,opt3)-
    #     (
    #         OTT(C,opt1,opt2)-OTT(C,opt1,opt3)
    #     ),
    #     -100
    #     )
    # )

    return var, ott, rott


def ott_bolge_signal(dataframe: DataFrame, window, percent, percent2, field='close', matype='var'):

    if field in (['open', 'high', 'low', 'close', 'volume']):
        df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()
    else:
        df = dataframe.copy()

    if(field == 'hl2'):
        df['hl2'] = qtpylib.mid_price(dataframe)
    elif(field == 'hlc4'):
        df['hlc4'] = (dataframe["high"] + dataframe["low"] +
                      dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, window, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop

        return np.concatenate((np.full(2, 0), ott[:-2]))

    ott = calcOtt(var, percent, window, first_index)
    ott2 = calcOtt(var, percent2, window, first_index)

    # rott = (ott2 + (2*ott2 - ott).shift(100))/2

    rott = (ott2 +  np.concatenate((np.full(100, np.nan), (2*ott2 - ott)[:-100])))/2

    # (
    # OTT(C,opt1,opt3)
    # +
    # REF(
    #     OTT(C,opt1,opt3)-
    #     (
    #         OTT(C,opt1,opt2)-OTT(C,opt1,opt3)
    #     ),
    #     -100
    #     )
    # )



    # MAV > OTT:
    #     MAV > ROTT : Bolge1 (Trend Long)  (Trend LongExit) - (NoTrendShortExit)
    #     MAV < ROTT : Bolge2 (Trend Long)  (Trend LongExit) (Firsatci ShortEntry) (NoTrendShortExit)
    # MAV < OTT:
    #     MAV > ROTT : Bolge3 (Firsatci Long) (NoTrend LongExit) -- (Short Entry) - (ShortExit)
    #     MAV < ROTT : Bolge4  (NoTrendLongExit)      -- (ShortEntry)  - (ShortExit)  

    signal = np.where(var > ott, np.where(var > rott, 1, 2), np.where(var>rott, 3, 4))

    return signal

def ott_smt_signal(dataframe: DataFrame, window, percent, smoothing=0.0008, field='close', matype='var'):

    df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()

    if(field == 'hl2'):
        dataframe['hl2'] = qtpylib.mid_price(dataframe)
    elif(field == 'hlc4'):
        dataframe['hlc4'] = (dataframe["high"] + dataframe["low"] +
                             dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, window, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop
        return np.concatenate((np.full(2, 0), ott[:-2]))

    ott_ret = calcOtt(var, percent, window, first_index)
    ott_up = ott_ret * (1.00 + smoothing)
    ott_down = ott_ret * (1.00 - smoothing)
    up_signal = np.where(var > ott_up, 'buy', 'sell')
    down_signal = np.where(var < ott_down, 'sell', 'buy')
    return up_signal, down_signal


def StochVar_Signal(df: DataFrame, fastk_period=520, fastd_period=350, smoothing=0.3, plus=1.0, matype='var'):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'] = qtpylib.stoch(my_df, window=fastk_period, d=111, k=2, fast=True)['fast_k']
    # my_df['fastk'], fastd = ta.STOCHF(my_df['high'], my_df['low'], my_df['close'],fastk_period=fastk_period, fastd_period=1, fastd_matype=0)
    if matype == 'var':
        my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    elif matype == 'evwma':
        my_df['fastk_var'] = EVWMA(my_df, fastd_period, field='fastk')
    elif matype == 'vwap':
        my_df['fastk_var'] = qtpylib.rolling_vwap(my_df, fastd_period)
    elif matype == 'vma':
        my_df['fastk_var'] = vma(my_df, fastd_period, field='fastk')
    else:
        my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    plusval = (1000.00 * plus)
    my_df['fastk_src'] = my_df['fastk_var'] + plusval
    _, stott = ott_fast(my_df, 2, smoothing, field='fastk_src')
    signal = np.where(my_df['fastk_src'] > stott, 'buy', 'sell')
    return signal

def tott_combine_buy(df:DataFrame, window, percent, smoothing=0.0008, field='close', matype='var',fastk_period=520, fastd_period=350, fsmoothing=0.3, plus=1.0, hhvlen=20, hott_length=20, hott_perc=0.6 ):

   up, _ = ott_smt_signal(df, window, percent, smoothing, field, matype)
   stsk = StochVar_Signal(df, fastk_period, fastd_period, fsmoothing, plus, matype)
   hhv = hhv_signal(df, hhvlen)
   hott = hott_signal(df, hott_length, 0, hott_perc)
   return np.where((up == 'buy') & (stsk == 'buy') & (hhv == 'buy') & (hott == 'buy'), 'buy', 'sell')

def tott_combine_sell(df:DataFrame, window, percent, smoothing=0.0008, field='close', matype='var',fastk_period=520, fastd_period=350, fsmoothing=0.3, plus=1.0, llvlen=20, lott_length=20, lott_perc=0.6 ):

   _, down = ott_smt_signal(df, window, percent, smoothing, field, matype)
   stsk = StochVar_Signal(df, fastk_period, fastd_period, fsmoothing, plus, matype)
   llv = llv_signal(df, llvlen)
   lott = lott_signal(df, lott_length, 0, lott_perc)
   return np.where((down == 'sell') & (stsk == 'sell') &( llv == 'sell') &( lott == 'sell'), 'sell', 'buy')


def StochVarCross_Signal(df: DataFrame, fastk_period=520, fastd_period=350, smoothing=0.3, plus=1.0, matype='var'):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'] = qtpylib.stoch(my_df, window=fastk_period, d=111, k=2, fast=True)['fast_k']
    # my_df['fastk'], fastd = ta.STOCHF(my_df['high'], my_df['low'], my_df['close'],fastk_period=fastk_period, fastd_period=1, fastd_matype=0)
    if matype == 'var':
        my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    elif matype == 'evwma':
        my_df['fastk_var'] = EVWMA(my_df, fastd_period, field='fastk')
    elif matype == 'vwap':
        my_df['fastk_var'] = qtpylib.rolling_vwap(my_df, fastd_period)
    elif matype == 'vma':
        my_df['fastk_var'] = vma(my_df, fastd_period, field='fastk')
    else:
        my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    plusval = (1000.00 * plus)
    my_df['fastk_src'] = my_df['fastk_var'] + plusval
    stosk, stott = ott_fast(my_df, 2, smoothing, field='fastk_src')

    my_df['buycheck'] = (qtpylib.crossed_above(my_df['fastk_src'], stott)).rolling(3).max() == True 
    my_df['sellcheck'] = (qtpylib.crossed_below(my_df['fastk_src'], stott)).rolling(3).max() == True 
    signal= np.where(my_df['buycheck']== True,  'buy', np.where(my_df['sellcheck']== True,'sell', None))

    return signal

def AlphaTrend(df: DataFrame, multiplier=5.0, period: int = 14):

    # coeff:=input("ATR Multiplier",0,20,1);
    # AP:=input("Common Period",1,500,14);
    # momentum:=input("Hesaplama MFI=1 RSI=2",1,2,1);
    # mom:=if(momentum=1,MFI(AP),RSI(AP));
    # upT:=L-ATR(AP)*coeff;
    # downT:=H+ATR(AP)*coeff;
    # AlphaTrend:=If(mom>=50,If(upT<PREV,PREV,upT),If(downT>PREV,PREV,downT));
    # ref(AlphaTrend,-2);
    # AlphaTrend

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    my_df['tr'] = qtpylib.true_range(df)
    my_df['atr'] = vidya_fast(my_df, period, field='tr')

    # my_df['atr'] = qtpylib.atr(df, window=period, exp=False)

    upT = my_df['low'] - my_df['atr'] * multiplier
    downT = my_df['high'] + my_df['atr'] * multiplier
    my_df['mfi'] = ta.MFI(my_df, period)

    first_index = period if my_df['mfi'].first_valid_index(
    ) is None else my_df['mfi'].first_valid_index()

    def calcAt(mfi, upT, downT, first_index):

        at = np.full_like(mfi, 0)
        atPrev = 0.0
        for i in range(first_index + 1, len(mfi)):
            if(mfi[i] >= 50):
                if(upT[i] < atPrev):
                    at[i] = atPrev
                else:
                    at[i] = upT[i]
            else:
                if(downT[i] > atPrev):
                    at[i] = atPrev
                else:
                    at[i] = downT[i]

            atPrev = at[i]
        return at

    my_df['at'] = calcAt(my_df['mfi'], upT, downT, first_index)
    my_df['at_sig'] = np.concatenate((np.full(2, 0), my_df['at'][:-2]))

    return my_df['at'],  my_df['at_sig']


def numpy_fill(arr: DataFrame):
    '''Solution provided by Divakar.'''
    mask = pd.isna(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def Alpha_Signal(df: DataFrame, multiplier=5.0, period: int = 14):

    # coeff:=input("ATR Multiplier",0,20,1);
    # AP:=input("Common Period",1,500,14);
    # momentum:=input("Hesaplama MFI=1 RSI=2",1,2,1);
    # mom:=if(momentum=1,MFI(AP),RSI(AP));
    # upT:=L-ATR(AP)*coeff;
    # downT:=H+ATR(AP)*coeff;
    # AlphaTrend:=If(mom>=50,If(upT<PREV,PREV,upT),If(downT>PREV,PREV,downT));
    # ref(AlphaTrend,-2);
    # AlphaTrend

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()


    # my_df['tr'] = qtpylib.true_range(df)
    # my_df['atr'] = vidya_fast(my_df, period, field='tr')

    my_df['atr'] = qtpylib.atr(df, window=period, exp=False)

    upT = my_df['low'] - my_df['atr'] * multiplier
    downT = my_df['high'] + my_df['atr'] * multiplier
    my_df['mfi'] = ta.MFI(my_df, period)

    first_index = period if my_df['mfi'].first_valid_index(
    ) is None else my_df['mfi'].first_valid_index()

    def calcAt(mfi, upT, downT, first_index):

        at = np.full_like(mfi, 0)
        atPrev = 0.0
        for i in range(first_index + 1, len(mfi)):
            if(mfi[i] >= 50):
                if(upT[i] < atPrev):
                    at[i] = atPrev
                else:
                    at[i] = upT[i]
            else:
                if(downT[i] > atPrev):
                    at[i] = atPrev
                else:
                    at[i] = downT[i]

            atPrev = at[i]
        return at


    my_df['at'] = calcAt(my_df['mfi'], upT, downT, first_index)

    #crossover-version
    my_df['at_sig'] = np.concatenate((np.full(2, 0), my_df['at'][:-2]))

    my_df['up'] = qtpylib.crossed_above(my_df['at'], my_df['at_sig'])
    my_df['down'] = qtpylib.crossed_below(my_df['at'], my_df['at_sig'])

    my_df['sig'] = np.where(my_df['down'] == True, 'sell', np.where(
        (my_df['up'] == True), 'buy', None))
    my_df['sig'].fillna(method='ffill', inplace=True)
    # my_df['sig'] = np.where(my_df['close']< my_df['at'], 'sell', my_df['sig'])
    
    #upper-lower version
    # my_df['sig'] = np.where(my_df['close']< my_df['at'], 'sell', 'buy')      

    return my_df['sig']


def Alpha_SignalMA(df: DataFrame, multiplier=5.0, period: int = 14):

    # coeff:=input("ATR Multiplier",0,20,1);
    # AP:=input("Common Period",1,500,14);
    # momentum:=input("Hesaplama MFI=1 RSI=2",1,2,1);
    # mom:=if(momentum=1,MFI(AP),RSI(AP));
    # upT:=L-ATR(AP)*coeff;
    # downT:=H+ATR(AP)*coeff;
    # AlphaTrend:=If(mom>=50,If(upT<PREV,PREV,upT),If(downT>PREV,PREV,downT));
    # ref(AlphaTrend,-2);
    # AlphaTrend

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    my_df['MA'] = vidya_fast(my_df, 5, 'close')

    # my_df['tr'] = qtpylib.true_range(df)
    # my_df['atr'] = vidya_fast(my_df, period, field='tr')

    my_df['atr'] = qtpylib.atr(df, window=period, exp=False)

    upT = my_df['low'] - my_df['atr'] * multiplier
    downT = my_df['high'] + my_df['atr'] * multiplier
    my_df['mfi'] = ta.MFI(my_df, period)

    first_index = period if my_df['mfi'].first_valid_index(
    ) is None else my_df['mfi'].first_valid_index()

    def calcAt(mfi, upT, downT, first_index):

        at = np.full_like(mfi, 0)
        atPrev = 0.0
        for i in range(first_index + 1, len(mfi)):
            if(mfi[i] >= 50):
                if(upT[i] < atPrev):
                    at[i] = atPrev
                else:
                    at[i] = upT[i]
            else:
                if(downT[i] > atPrev):
                    at[i] = atPrev
                else:
                    at[i] = downT[i]

            atPrev = at[i]
        return at


    my_df['at'] = calcAt(my_df['mfi'], upT, downT, first_index)

    #crossover-version
    my_df['at_sig'] = np.concatenate((np.full(2, 0), my_df['at'][:-2]))

    my_df['up'] = qtpylib.crossed_above(my_df['at'], my_df['at_sig'])
    my_df['down'] = qtpylib.crossed_below(my_df['at'], my_df['at_sig'])

    my_df['sig'] = np.where(my_df['down'] == True, 'sell', np.where(
        (my_df['up'] == True), 'buy', None))
    my_df['sig'].fillna(method='ffill', inplace=True)
    my_df['sig'] = np.where((my_df['sig'] == 'buy') & (my_df['MA']< my_df['at']), 'sell', my_df['sig'])
    
    #upper-lower version
    # my_df['sig'] = np.where(my_df['close']< my_df['at'], 'sell', 'buy')      

    return my_df['sig']

def AlphaTrend_Signal(df: DataFrame, multiplier=5.0, period: int = 14):

    # coeff:=input("ATR Multiplier",0,20,1);
    # AP:=input("Common Period",1,500,14);
    # momentum:=input("Hesaplama MFI=1 RSI=2",1,2,1);
    # mom:=if(momentum=1,MFI(AP),RSI(AP));
    # upT:=L-ATR(AP)*coeff;
    # downT:=H+ATR(AP)*coeff;
    # AlphaTrend:=If(mom>=50,If(upT<PREV,PREV,upT),If(downT>PREV,PREV,downT));
    # ref(AlphaTrend,-2);
    # AlphaTrend

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # my_df['tr'] = qtpylib.true_range(df)
    # my_df['atr'] = vidya_fast(my_df, period, field='tr')

    my_df['atr'] = qtpylib.atr(df, window=period, exp=False)

    upT = my_df['low'] - my_df['atr'] * multiplier
    downT = my_df['high'] + my_df['atr'] * multiplier
    my_df['mfi'] = ta.MFI(my_df, period)

    first_index = period if my_df['mfi'].first_valid_index(
    ) is None else my_df['mfi'].first_valid_index()

    def calcAt(mfi, upT, downT, first_index):

        at = np.full_like(mfi, 0)
        atPrev = 0.0
        for i in range(first_index + 1, len(mfi)):
            if(mfi[i] >= 50):
                if(upT[i] < atPrev):
                    at[i] = atPrev
                else:
                    at[i] = upT[i]
            else:
                if(downT[i] > atPrev):
                    at[i] = atPrev
                else:
                    at[i] = downT[i]

            atPrev = at[i]
        return at


    my_df['at'] = calcAt(my_df['mfi'], upT, downT, first_index)

    #crossover-version
    my_df['at_sig'] = np.concatenate((np.full(2, 0), my_df['at'][:-2]))

    my_df['up'] = qtpylib.crossed_above(my_df['at'], my_df['at_sig'])
    my_df['down'] = qtpylib.crossed_below(my_df['at'], my_df['at_sig'])

    my_df['sig'] = np.where(my_df['down'] == True, 'sell', np.where(
        (my_df['up'] == True), 'buy', None))
    my_df['sig'].fillna(method='ffill', inplace=True)
    my_df['sig'] = np.where(my_df['close']< my_df['at'], 'sell', my_df['sig'])
    
    #upper-lower version
    # my_df['sig'] = np.where(my_df['close']<= my_df['at'], 'sell', 'buy')      

    return my_df['sig']

def IFStochSignal(df: DataFrame, fastk_period=520, fastd_period=350):

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    my_df['fastk'], my_df['fastd'] = ta.STOCHF(
        my_df['high'], my_df['low'], my_df['close'], fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=2)
    # my_df['fastk_var'] = vidya_fast(my_df, fastd_period, field='fastk')
    my_df['fastk_src'] = 0.1 * (my_df['fastd'] - 50.0)

    my_df['ifst'] = (np.exp(2 * my_df['fastk_src'])-1) / (np.exp(2 * my_df['fastk_src']) + 1)

    my_df['up'] = qtpylib.crossed_above(my_df['ifst'], -0.50)
    my_df['up'] = qtpylib.crossed_above(my_df['ifst'], 0.50)
    my_df['down'] = qtpylib.crossed_below(my_df['ifst'], 0.50)
    my_df['down'] = qtpylib.crossed_below(my_df['ifst'], -0.50)
    my_df['sig'] = np.where(my_df['down'] == True, 'sell', np.where((my_df['up'] == True), 'buy', None))
    my_df['sig'].fillna(method='ffill', inplace=True)
    return my_df['sig']

def AlphaMainTrend_Signal(df: DataFrame, multiplier=5.0, period: int = 14):

    # coeff:=input("ATR Multiplier",0,20,1);
    # AP:=input("Common Period",1,500,14);
    # momentum:=input("Hesaplama MFI=1 RSI=2",1,2,1);
    # mom:=if(momentum=1,MFI(AP),RSI(AP));
    # upT:=L-ATR(AP)*coeff;
    # downT:=H+ATR(AP)*coeff;
    # AlphaTrend:=If(mom>=50,If(upT<PREV,PREV,upT),If(downT>PREV,PREV,downT));
    # ref(AlphaTrend,-2);
    # AlphaTrend

    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # my_df['tr'] = qtpylib.true_range(df)
    # my_df['atr'] = vidya_fast(my_df, period, field='tr')

    my_df['atr'] = qtpylib.atr(df, window=period, exp=False)

    upT = my_df['low'] - my_df['atr'] * multiplier
    downT = my_df['high'] + my_df['atr'] * multiplier
    my_df['mfi'] = ta.MFI(my_df, period)

    first_index = period if my_df['mfi'].first_valid_index(
    ) is None else my_df['mfi'].first_valid_index()

    def calcAt(mfi, upT, downT, first_index):

        at = np.full_like(mfi, 0)
        atPrev = 0.0
        for i in range(first_index + 1, len(mfi)):
            if(mfi[i] >= 50):
                if(upT[i] < atPrev):
                    at[i] = atPrev
                else:
                    at[i] = upT[i]
            else:
                if(downT[i] > atPrev):
                    at[i] = atPrev
                else:
                    at[i] = downT[i]

            atPrev = at[i]
        return at


    my_df['at'] = calcAt(my_df['mfi'], upT, downT, first_index)
    
    #upper-lower version
    my_df['sig'] = np.where(my_df['close']<= my_df['at'], 'sell', 'buy')      

    return my_df['sig']

def LaguerreRsi(dataframe: DataFrame, alpha: float, field='close'):

    df = dataframe[[field]].copy()

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcLag(src, alpha, first_index):

        # gamma=1-alpha
        # L0 = 0.0
        # L0 := (1-gamma) * src + gamma * nz(L0[1])
        # L1 = 0.0
        # L1 := -gamma * L0 + nz(L0[1]) + gamma * nz(L1[1])

        # L2 = 0.0
        # L2 := -gamma * L1 + nz(L1[1]) + gamma * nz(L2[1])

        # L3 = 0.0
        # L3 := -gamma * L2 + nz(L2[1]) + gamma * nz(L3[1])

        # cu= (L0>L1 ? L0-L1 : 0) + (L1>L2 ? L1-L2 : 0) + (L2>L3 ? L2-L3 : 0)

        # cd= (L0<L1 ? L1-L0 : 0) + (L1<L2 ? L2-L1 : 0) + (L2<L3 ? L3-L2 : 0)

        # temp= cu+cd==0 ? -1 : cu+cd
        # LaRSI=temp==-1 ? 0 : cu/temp
        gamma = 1-alpha
        L0 = np.zeros_like(src)
        l0prev = 0
        for i in range(first_index + 1, len(src)):
            L0[i] = (1-gamma) * src[i] + gamma * (l0prev)
            l0prev = L0[i]

        L1 = np.zeros_like(src)
        l1prev = 0
        for i in range(first_index + 1, len(src)):
            L1[i] = -gamma * L0[i] + L0[i-1] + gamma * (l1prev)
            l1prev = L1[i]

        L2 = np.zeros_like(src)
        l2prev = 0
        for i in range(first_index + 1, len(src)):
            L2[i] = -gamma * L1[i] + L1[i-1] + gamma * (l2prev)
            l2prev = L2[i]

        L3 = np.zeros_like(src)
        l3prev = 0
        for i in range(first_index + 1, len(src)):
            L3[i] = -gamma * L2[i] + L2[i-1] + gamma * (l3prev)
            l3prev = L3[i]

        cu = np.where(L0 > L1, L0-L1, 0.0) + np.where(L1 > L2,
                                                      L1 - L2, 0.0) + np.where(L2 > L3, L2-L3, 0.0)
        cd = np.where(L0 < L1, L1-L0, 0.0) + np.where(L1 < L2,
                                                      L2 - L1, 0.0) + np.where(L2 < L3, L3-L2, 0.0)

        temp = np.where(L0 + L1 == 0, -1, cu + cd)
        LaRSI = np.where(temp == -1, 0, (cu / temp))

        return LaRSI

    lag = calcLag(df[field], alpha, first_index)
    return lag

def ott_trend_region(dataframe: DataFrame, window, percent, smoothing=0.0008, field='close', matype='var'):

    df = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()

    if(field == 'hl2'):
        dataframe['hl2'] = qtpylib.mid_price(dataframe)
    elif(field == 'hlc4'):
        dataframe['hlc4'] = (dataframe["high"] + dataframe["low"] +
                             dataframe["close"] + dataframe["open"]) / 4

    if matype == 'var':
        var = vidya_fast(df, window, field=field)
    elif matype == 'evwma':
        var = EVWMA(df, window, field)
    elif matype == 'vma':
        var = vma(df, window, field)
    elif matype == 'vwap':
        var = qtpylib.rolling_vwap(df, window)
    else:
        var = vidya_fast(df, window, field=field)

    first_index = window if dataframe[field].first_valid_index(
    ) is None else dataframe[field].first_valid_index()

    def calcOtt(var, percent, window, first_index):

        fark = np.multiply(np.multiply(var, percent), 0.01)
        longStop = np.subtract(var, fark)
        shortStop = np.add(var, fark)
        direction = np.ones_like(var)
        longStopPrev = 0
        shortStopPrev = 0
        direction = 1
        mt = np.full_like(var, 0)
        ott = np.full_like(var, 0)

        for i in range(first_index + 1, len(var)):

            vlongStop = max(longStopPrev, longStop[i]) if var[i] > longStopPrev else longStop[i]
            vshortStop = min(
                shortStop[i], shortStopPrev) if var[i] < shortStopPrev else shortStop[i]

            if direction == -1 and var[i] > shortStopPrev:
                direction = 1
            elif direction == 1 and var[i] < longStopPrev:
                direction = -1

            mt[i] = vlongStop if direction == 1 else vshortStop
            ott[i] = (mt[i] * (200 + percent)/200) if var[i] > mt[i] else mt[i] * (200 - percent)/200
            longStopPrev = vlongStop
            shortStopPrev = vshortStop
        return np.concatenate((np.full(2, 0), ott[:-2]))

    ott_ret = calcOtt(var, percent, window, first_index)
    ott_up = ott_ret * (1.00 + smoothing)
    ott_down = ott_ret * (1.00 - smoothing)
    signal = np.where(var > ott_up, 1, np.where(var< ott_down, 3, 2))

    return signal


def TATTrend_Region(df: DataFrame, multiplier=5.0, period: int = 14, upperc=0.01, downperc=0.01, varlength= 5):

    # coeff:=input("ATR Multiplier",0,20,1);
    # AP:=input("Common Period",1,500,14);
    # momentum:=input("Hesaplama MFI=1 RSI=2",1,2,1);
    # mom:=if(momentum=1,MFI(AP),RSI(AP));
    # upT:=L-ATR(AP)*coeff;
    # downT:=H+ATR(AP)*coeff;
    # AlphaTrend:=If(mom>=50,If(upT<PREV,PREV,upT),If(downT>PREV,PREV,downT));
    # ref(AlphaTrend,-2);
    # AlphaTrend


    my_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # my_df['tr'] = qtpylib.true_range(df)
    # my_df['atr'] = vidya_fast(my_df, period, field='tr')

    my_df['atr'] = qtpylib.atr(df, window=period, exp=False)

    upT = my_df['low'] - my_df['atr'] * multiplier
    downT = my_df['high'] + my_df['atr'] * multiplier
    my_df['mfi'] = ta.MFI(my_df, period)

    first_index = period if my_df['mfi'].first_valid_index(
    ) is None else my_df['mfi'].first_valid_index()

    def calcAt(mfi, upT, downT, first_index):

        at = np.full_like(mfi, 0)
        atPrev = 0.0
        for i in range(first_index + 1, len(mfi)):
            if(mfi[i] >= 50):
                if(upT[i] < atPrev):
                    at[i] = atPrev
                else:
                    at[i] = upT[i]
            else:
                if(downT[i] > atPrev):
                    at[i] = atPrev
                else:
                    at[i] = downT[i]

            atPrev = at[i]
        return at


    my_df['at'] = calcAt(my_df['mfi'], upT, downT, first_index)

    my_df['atup'] = my_df['at'] * (1.00 + upperc)
    my_df['atdown'] = my_df['at'] * (1.00 - downperc)

    my_df['var'] = vidya_fast(my_df, varlength, field='close')

    my_df['sig'] = np.where(my_df['var'] > my_df['atup'], 1, np.where(my_df['var']< my_df['atdown'], 3, 2))
    return my_df['sig']


def RSICheck(df: DataFrame, period: int = 14, cross = 30, lookback = 20):

    rsi = ta.RSI(df, period)
    df['check'] = (qtpylib.crossed_above(rsi, cross)).rolling(lookback).max() == True 
    return df['check']


### Son bar kontrolleri icin
### Ornek 1
### dataframe['emacrossed'] = qtpylib.crossed_above(dataframe['ema10'], dataframe['ema20'])
### dataframe['emacrossed'].rolling(10).max() == 1
### son on bard yukari kiran var mi ? 

###Ornek 2:
### example df['check'] = (df['this'] > df['that']).rolling(20).max() == True 


###Ornek 3:
### np.where(<crossed check>).rolling(20).sum() > 1
### Son 20 barda en az bir defa olma durumu 

