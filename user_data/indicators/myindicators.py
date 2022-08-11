#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import warnings
import sys
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

import talib.abstract as ta  # noqa
from talib import MA_Type
import freqtrade.vendor.qtpylib.indicators as qtpylib
import technical.indicators as indicators 
from pandas.core.base import NoNewAttributesMixin, PandasObject

# =============================================
# check min, python version
if sys.version_info < (3, 4):
    raise SystemError("QTPyLib requires Python version >= 3.4")

# =============================================
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# =============================================

def vidyanp(dataframe, period, field='close') -> np.ndarray:


    if f"vidya-ma-{period}" in dataframe.columns.values:
       return dataframe[f"vidya-ma-{period}"].to_numpy()

    dataframe['hl2'] = qtpylib.mid_price(dataframe)
    alpha = 2 / (period + 1)
    src = dataframe[field].to_numpy()

    up = np.diff(src)
    down = np.diff(src)

    up[up < 0.00] = 0.00
    down[down > 0.00] = 0.00
    down = np.abs(down)

    sm11 = np.convolve(up,np.ones(9,dtype=int),'valid')
    sm22 = np.convolve(down,np.ones(9,dtype=int),'valid')

    sm11 = np.concatenate((np.full((src.shape[0] - sm11.shape[0]), np.nan), sm11))
    sm22 = np.concatenate((np.full((src.shape[0] - sm22.shape[0]), np.nan), sm22))
    cmo = np.divide(np.subtract(sm11, sm22), np.add(sm11,sm22))

    cmo = np.abs(cmo)

    vidya = np.full_like(src, np.nan)
    for i in range(period, len(vidya)):
        vidyaprev = src[i] if np.isnan(vidya[i-1]) else vidya[i-1]
        
        vidya[i] = alpha * cmo[i] * src[i] + \
                    (1 - alpha * cmo[i]) * vidyaprev

    return vidya


def ott6( dataframe:DataFrame, window, percent, field='close') -> DataFrame:
    
    vma = vidyanp(dataframe, window, field=field)
    
    fark = vma * percent * 0.01
    longStop = vma - fark
    shortStop = vma + fark

    # @numba.jit(nopython=True)
    def calcOtt(vma, longStop,shortStop, percent, window):

        direction = 1
        
        mt = np.full_like(vma,np.nan)
        ott = np.full_like(vma,np.nan)

        for i in range(window, len(vma)): 

            if vma[i] > longStop[i-1]:
                longStop[i] = max(longStop[i], longStop[i-1])

            if vma[i] < shortStop[i-1]:
                shortStop[i] = min(shortStop[i], shortStop[i-1])

            if direction == -1 and vma[i] > shortStop[i-1]:
                direction = 1
            elif direction == 1 and vma[i] < longStop[i-1]:
                direction = -1

            mt[i] = longStop[i] if direction == 1 else shortStop[i]
            ott[i] = (mt[i] * (200 + percent)/200) if vma[i] > mt[i] else  mt[i] * (200 - percent)/200

        return ott

    result = calcOtt(vma, longStop, shortStop, percent, window)

    #ott1 = result * (1 + 0.001)
    # ott2 = result * (1 - 0.001)
    
    return  result

def StochVar3(df: DataFrame, fastk_period = 520, fastd_period=350, smoothing = 0.3):
    my_df = df[['open','high','low','close','volume']].copy()

    fastk, fastd = ta.STOCHF(my_df['high'], my_df['low'], my_df['close'],fastk_period=fastk_period, fastd_period=1, fastd_matype=0)
    my_df['fastk'] = fastk
    fastk_var = vidyanp(my_df, fastd_period, field='fastk') + 1000

    fark = fastk_var * smoothing * 0.01
    longStop = fastk_var - fark
    shortStop = fastk_var + fark

    # @numba.jit(nopython=True)
    def calcOtt(vma, longStop,shortStop, percent, window):

        direction = 1
        
        mt = np.full_like(vma,np.nan)
        ott = np.full_like(vma,np.nan)

        for i in range(window, len(vma)): 

            if vma[i] > longStop[i-1]:
                longStop[i] = max(longStop[i], longStop[i-1])

            if vma[i] < shortStop[i-1]:
                shortStop[i] = min(shortStop[i], shortStop[i-1])

            if direction == -1 and vma[i] > shortStop[i-1]:
                direction = 1
            elif direction == 1 and vma[i] < longStop[i-1]:
                direction = -1

            mt[i] = longStop[i] if direction == 1 else shortStop[i]
            ott[i] = (mt[i] * (200 + percent)/200) if vma[i] > mt[i] else  mt[i] * (200 - percent)/200

        return ott

    fastd_ott = calcOtt(fastk_var, longStop, shortStop, smoothing, 2)
    return fastk_var,fastd_ott
