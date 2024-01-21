# add common folders to path
import re
import sys
import os

import talib

from freqtrade.exchange.exchange_utils import timeframe_to_prev_date
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.strategy.strategy_helper import merge_informative_pair

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import sdnotify
from freqtrade.enums.runmode import RunMode
from typing import Dict, List, Optional
from lib.ma import MovingAveragesCalculate, MovingAveragesCalculator2
from lib.mom import MomentumANDVolatilityCalculate
from lib.cycle import CycleCalculate
from lib.trend import TrendCalculate
from lib.oscillators import OscillatorsCalculate
from lib import helpers
from lib.sagemaster import SageMasterClient
from lib.Alpha101 import get_alpha
from scipy.special import softmax

import lib.glassnode as gn

import warnings
import json
import logging
from functools import reduce
import time
import numpy as np
from technical.pivots_points import pivots_points

import pandas as pd
from pandas import DataFrame, Series
from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import BooleanParameter, DecimalParameter, IntParameter
from datetime import timedelta, datetime, timezone
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
import talib.abstract as ta
from sqlalchemy import desc
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.signal import argrelextrema
from joblib import Parallel, delayed

from technical import qtpylib
import pandas_ta as pta
from freqtrade.exchange import timeframe_to_minutes
from lib.prediction_storage import PredictionStorage

logger = logging.getLogger(__name__)

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_distance(p1, p2):
    return abs((p1) - (p2))

def candle_stats(dataframe):
    # print("candle_stats", dataframe)
    # log data
    dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
    dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
    dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] +
                          dataframe['low'] + dataframe['close']) / 4

    dataframe['hlc3_log'] = np.log(dataframe['hlc3'])
    dataframe['hl2_log'] = np.log(dataframe['hl2'])
    dataframe['ohlc4_log'] = np.log(dataframe['ohlc4'])

    dataframe['close_log'] = np.log(dataframe['close'])
    dataframe['high_log'] = np.log(dataframe['high'])
    dataframe['low_log'] = np.log(dataframe['low'])
    dataframe['open_log'] = np.log(dataframe['open'])
    return dataframe

def f(x):
    return x

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

# VWAP bands


def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def EWO(dataframe, sma_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


def get_distance(p1, p2):
    return abs((p1) - (p2))


class TM3BinaryClassV2(IStrategy):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy. Namely, the user uses:
    self.freqai.start(dataframe, metadata)

    to make predictions on their data. populate_any_indicators() automatically
    generates the variety of features indicated by the user in the
    canonical freqtrade configuration file under config['freqai'].
    """

    def heartbeat(self):
        sdnotify.SystemdNotifier().notify("WATCHDOG=1\nSTATUS=State: RUNNING.")

    def log(self, msg, *args, **kwargs):
        self.heartbeat()
        logger.info(msg, *args, **kwargs)

    minimal_roi = {
        "0": 0.04 # Target: RR 1:1
    }

    TARGET_VAR = "ohlc4_log"

    process_only_new_candles = True
    use_exit_signal = True
    can_short = True
    ignore_roi_if_entry_signal = True
    use_custom_stoploss = False

    stoploss = -0.04
    trailing_stop = False
    trailing_only_offset_is_reached  = False
    trailing_stop_positive_offset = 0

    # user should define the maximum startup candle count (the largest number of candles
    # passed to any single indicator)
    # internally freqtrade multiply it by 2, so we put here 1/2 of the max startup candle count
    startup_candle_count: int = 200

    LONG_ENTRY_SIGNAL_TRESHOLD = DecimalParameter(0.7, 0.95, decimals=2, default=0.8, space="buy", optimize=True)
    SHORT_ENTRY_SIGNAL_TRESHOLD = DecimalParameter(0.7, 0.95, decimals=2, default=0.8, space="buy", optimize=True)

    ENTRY_STRENGTH_TRESHOLD = DecimalParameter(0.4, 0.7, decimals=2, default=0.3, space="buy", optimize=True)

    LONG_TP = DecimalParameter(0.01, 0.03, decimals=3, default=0.016, space="sell", optimize=True)
    SHORT_TP = DecimalParameter(0.01, 0.03, decimals=3, default=0.016, space="sell", optimize=True)

    # user should define the maximum startup candle count (the largest number of candles
    # passed to any single indicator)
    # internally freqtrade multiply it by 2, so we put here 1/2 of the max startup candle count

    @property
    def PREDICT_TARGET(self):
        return self.config["freqai"].get("label_period_candles", 6)

    @property
    def PREDICT_STORAGE_ENABLED(self):
        return self.config["sagemaster"].get("PREDICT_STORAGE_ENABLED")

    @property
    def PREDICT_STORAGE_CONN_STRING(self):
        return self.config["sagemaster"].get("PREDICT_STORAGE_CONN_STRING")


    @property
    def TARGET_EXTREMA_KERNEL(self):
        return self.config["sagemaster"].get('TARGET_EXTREMA_KERNEL', 24)

    @property
    def TARGET_EXTREMA_WINDOW(self):
        return self.config["sagemaster"].get('TARGET_EXTREMA_WINDOW', 5)

    def bot_start(self, **kwargs) -> None:
        print("bot_start")

        if (self.PREDICT_STORAGE_ENABLED):
            self.ps = PredictionStorage(connection_string=self.config["sagemaster"].get("PREDICT_STORAGE_CONN_STRING"))

    def feature_engineering_trend(self, df: pd.DataFrame, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_trend() {metadata} {df.shape}")
        start_time = time.time()

        # Optimized Column Filtering using Regular Expression
        regex_pattern = r'^(?!.*(pmX|date|_signal|_trend|_rising|_std|_change_|_lower_band|_upper_band|_upper_envelope|_lower_envelope|%-dist_to_|%-s[123]|%-r[123]|_divergence)).*$'
        all_cols = [col for col in df.columns if re.match(regex_pattern, col)]

        # # Parallel Processing
        n_jobs = self.config["freqai"].get("data_kitchen_thread_count", 4)
        result_cols = Parallel(n_jobs=n_jobs)(
            delayed(helpers.create_col_trend)(col, self.PREDICT_TARGET, df) for col in all_cols
        )
        # # Using DataFrame apply
        # result_cols = []
        # for col in all_cols:
        #     # try:
        #     trend_col = helpers.create_col_trend(col, self.PREDICT_TARGET, df)
        #     result_cols.append(trend_col)
        #     # except Exception as e:
        #     # self.log(f"Error processing column {col}: {e}")
        #         # continue

        # Memory Efficient Concatenation
        result_df = pd.concat(result_cols, axis=1)
        df = pd.concat([df, result_df], axis=1)

        self.log(f"EXIT .feature_engineering_trend() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")

        return df

    def feature_engineering_candle_patterns(self, df, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_candle_patterns(): {metadata} {df.shape}")
        start_time = time.time()
        pref = f"%"
        df = df.copy()

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # Hammer: values [0, 100]
        df[f'{pref}-CDLHAMMER'] = ta.CDLHAMMER(df)
        # Inverted Hammer: values [0, 100]
        df[f'{pref}-CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df)
        # Dragonfly Doji: values [0, 100]
        df[f'{pref}-CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df)
        # Piercing Line: values [0, 100]
        df[f'{pref}-CDLPIERCING'] = ta.CDLPIERCING(df) # values [0, 100]
        # Morningstar: values [0, 100]
        df[f'{pref}-CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df) # values [0, 100]
        # Three White Soldiers: values [0, 100]
        df[f'{pref}-CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        df[f'{pref}-CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df)
        # Shooting Star: values [0, 100]
        df[f'{pref}-CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df)
        # Gravestone Doji: values [0, 100]
        df[f'{pref}-CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df)
        # Dark Cloud Cover: values [0, 100]
        df[f'{pref}-CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df)
        # Evening Doji Star: values [0, 100]
        df[f'{pref}-CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(df)
        # Evening Star: values [0, 100]
        df[f'{pref}-CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # Three Line Strike: values [0, -100, 100]
        df[f'{pref}-CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(df)
        # Spinning Top: values [0, -100, 100]
        df[f'{pref}-CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df) # values [0, -100, 100]
        # Engulfing: values [0, -100, 100]
        df[f'{pref}-CDLENGULFING'] = ta.CDLENGULFING(df) # values [0, -100, 100]
        # Harami: values [0, -100, 100]
        df[f'{pref}-CDLHARAMI'] = ta.CDLHARAMI(df) # values [0, -100, 100]
        # Three Outside Up/Down: values [0, -100, 100]
        df[f'{pref}-CDL3OUTSIDE'] = ta.CDL3OUTSIDE(df) # values [0, -100, 100]
        # Three Inside Up/Down: values [0, -100, 100]
        df[f'{pref}-CDL3INSIDE'] = ta.CDL3INSIDE(df) # values [0, -100, 100]

        # Pattern Recognition - The Rest
        # ------------------------------------
        # df[f'{pref}-CDL2CROWS'] = ta.CDL2CROWS(df)
        # df[f'{pref}-CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(df)
        # df[f'{pref}-CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(df)
        # df[f'{pref}-CDLABANDONEDBABY'] = ta.CDLABANDONEDBABY(df)
        # df[f'{pref}-CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(df)
        # df[f'{pref}-CDLBELTHOLD'] = ta.CDLBELTHOLD(df)
        # df[f'{pref}-CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(df)
        # df[f'{pref}-CDLCOUNTERATTACK'] = ta.CDLCOUNTERATTACK(df)
        # df[f'{pref}-CDLDOJI'] = ta.CDLDOJI(df)
        # df[f'{pref}-CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(df)
        # df[f'{pref}-CDLHARAMICROSS'] = ta.CDLHARAMICROSS(df)
        # df[f'{pref}-CDLHIKKAKE'] = ta.CDLHIKKAKE(df)
        # df[f'{pref}-CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(df)
        # df[f'{pref}-CDLINNECK'] = ta.CDLINNECK(df)
        # df[f'{pref}-CDLKICKING'] = ta.CDLKICKING(df)
        # df[f'{pref}-CDLKICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(df)
        # df[f'{pref}-CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(df)
        # df[f'{pref}-CDLLONGLINE'] = ta.CDLLONGLINE(df)
        # df[f'{pref}-CDLMARUBOZU'] = ta.CDLMARUBOZU(df)
        # df[f'{pref}-CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(df)
        # df[f'{pref}-CDLMATHOLD'] = ta.CDLMATHOLD(df)
        # df[f'{pref}-CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(df)
        # df[f'{pref}-CDLONNECK'] = ta.CDLONNECK(df)
        # df[f'{pref}-CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(df)
        # df[f'{pref}-CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(df)
        # df[f'{pref}-CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(df)
        # df[f'{pref}-CDLSHORTLINE'] = ta.CDLSHORTLINE(df)
        # df[f'{pref}-CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(df)
        # df[f'{pref}-CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(df)
        # df[f'{pref}-CDLTAKURI'] = ta.CDLTAKURI(df)
        # df[f'{pref}-CDLTASUKIGAP'] = ta.CDLTASUKIGAP(df)
        # df[f'{pref}-CDLTHRUSTING'] = ta.CDLTHRUSTING(df)
        # df[f'{pref}-CDLTRISTAR'] = ta.CDLTRISTAR(df)
        # df[f'{pref}-CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(df)
        # df[f'{pref}-CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(df)

        self.log(f"EXIT .feature_engineering_candle_patterns() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")

        return df.copy()

    def feature_engineering_alphas101(self, df:DataFrame, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_alphas101(): {metadata} {df.shape}")
        start_time = time.time()

        df = get_alpha(df)

        self.log(f"EXIT .feature_engineering_alphas101() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")
        return df


    def feature_engineering_expand_basic(self, df, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_expand_basic() {metadata} {df.shape}")
        start_time = time.time()

        # log data
        df = candle_stats(df)

        # add Alpha101
        df = self.feature_engineering_alphas101(df, metadata, **kwargs)

        # add TA features to dataframe

        # Moving Averages
        mac = MovingAveragesCalculator2(col_prefix="%-mac-", col_target='ohlc4_log', config = {
                "SMA": [24, 48, 96, 192],
                "EMA": [12, 24, 48, 96],
                "HMA": [12, 24, 48, 96],
                # "JMA": [6, 12, 24, 48],
                "KAMA": [12, 24, 48, 96],
                "ZLSMA": [12, 24, 48, 96],
            })

        df = mac.calculate_moving_averages(df)

        # Momentum Indicators
        mvc = MomentumANDVolatilityCalculate(
            df, open_col = 'open_log', close_col='close_log', high_col='high_log', low_col='low_log')
        df = mvc.calculate_all().copy()

        # Cycle Indicators
        cc = CycleCalculate(df, calc_col='close_log')
        df = cc.calculate_all().copy()

        # Trend Indicators
        tc = TrendCalculate(df, close_col='close_log', high_col='high_log', low_col='low_log',
                            open_col='open_log')
        df = tc.calculate_all().copy()

        # Oscillator Indicators
        oc = OscillatorsCalculate(df, close_col='close_log')
        df = oc.calculate_all().copy()

        # Pivot Points
        pp = pivots_points(df, timeperiod=100)
        df['r1'] = pp['r1']
        df['s1'] = pp['s1']
        df['r2'] = pp['r2']
        df['s2'] = pp['s2']
        df['r3'] = pp['r3']
        df['s3'] = pp['s3']

        df['%-dist_to_r1'] = get_distance(df['close'], df['r1'])
        df['%-dist_to_r2'] = get_distance(df['close'], df['r2'])
        df['%-dist_to_r3'] = get_distance(df['close'], df['r3'])
        df['%-dist_to_s1'] = get_distance(df['close'], df['s1'])
        df['%-dist_to_s2'] = get_distance(df['close'], df['s2'])
        df['%-dist_to_s3'] = get_distance(df['close'], df['s3'])

        cat_col = [x for x in df if x.find('pmX_10_3_12_1') != -1]

        for col in cat_col:
            df[col] = df[col].map({'down': 0, 'up': 1})

        # rename generated features so freqtrade can recognize them
        for col in df.columns:
            if col.startswith('%-') or col in ['date', 'volume', 'hl2_log', 'hl2', 'hlc3', 'hlc3_log', 'ohlc4', 'ohlc4_log', 'open', 'high', 'low', 'close', 'low_log', 'open_log', 'high_log', 'low_log', 'close_log']:
                continue
            else:
                df.rename(columns={col: "%-" + col}, inplace=True)

        # defragment df
        df = df.copy()

        # calculate trend for all features
        df = self.feature_engineering_trend(df, metadata, **kwargs).copy()

        # add chart pattern features
        df = self.feature_engineering_candle_patterns(df, metadata, **kwargs).copy()

        self.log(f"EXIT .feature_engineering_expand_basic() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")

        return df

    def feature_engineering_standard(self, df: DataFrame, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_standard() {metadata} {df.shape}")
        start_time = time.time()

        # add lag
        df = helpers.create_lag(df, 6, ignore_columns=[
            'hlc3', 'hlc3_log',
            'hl2', 'hl2_log',
            'ohlc4', 'ohlc4_log',
            'open', 'open_log',
            'high', 'high_log',
            'low', 'low_log',
            'close', 'close_log',
        ])

        # some basic features
        df["%-pct-change"] = df["close"].pct_change()
        df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
        df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25

        # volume features
        df["%-volume"] = df["volume"].copy()
        df = gn.extract_feature_metrics(df, "%-volume")
        df = df.copy()

        # fill empty values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill')
        # fill empty columns with 0
        df = df.fillna(0).copy()

        self.log(f"EXIT .feature_engineering_standard() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")
        return df


    def set_freqai_targets(self, df: DataFrame, metadata, **kwargs):
        self.log(f"ENTER .set_freqai_targets() {metadata} {df.shape}")
        start_time = time.time()

        df = candle_stats(df)

        # atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=100) / df['close']

        # target: trend slope
        kernel = 25
        df.set_index(df['date'], inplace=True)
        target = helpers.create_target(df, kernel,
                                       method='polyfit', polyfit_var=self.TARGET_VAR)
        target = target.set_index('start_windows')
        scaled_slope = RobustScaler().fit_transform(target['slope'].values.reshape(-1, 1)).reshape(-1)
        target['scaled_slope'] = scaled_slope
        # align index
        target = target.reindex(df.index)
        # set trend target
        df['trend_slope'] = np.nan
        df['trend_slope'] = target['scaled_slope']
        # reset index and get back
        df = df.reset_index(drop=True)

        ## Classify trend
        slope_filter = self.freqai_info["feature_parameters"]["target_slope_filter"]
        df['&-trend_long'] = np.where(df['trend_slope'] > slope_filter, 'trend_long', 'trend_not_long')
        df['&-trend_short'] = np.where(df['trend_slope'] < -slope_filter, 'trend_short', 'trend_not_short')


        # # target: expected range 12h
        # range_kernel = 12
        # max_range = df["high"].shift(-range_kernel).rolling(range_kernel).max() / df["close"] - 1
        # min_range = df["low"].shift(-range_kernel).rolling(range_kernel).min() / df["close"] - 1
        # tresh = atr * 2
        # df['&-trend_long-12'] = np.where(max_range > tresh, "trend_long-12", "trend_not_long-12")
        # df['&-trend_short-12'] = np.where(min_range < -tresh, "trend_short-12", "trend_not_short-12")

        # print(df['&-trend_long-12'].value_counts())
        # print(df['&-trend_short-12'].value_counts())

        # target: extrema
        df['extrema'] = 0
        min_peaks = argrelextrema(
            df["low_log"].values, np.less,
            order=self.TARGET_EXTREMA_KERNEL
        )
        max_peaks = argrelextrema(
            df["high_log"].values, np.greater,
            order=self.TARGET_EXTREMA_KERNEL
        )

        print(f"min_peaks: {len(min_peaks[0])}, max_peaks: {len(max_peaks[0])}")

        for mp in min_peaks[0]:
            df.at[mp, "extrema"] = -1
        for mp in max_peaks[0]:
            df.at[mp, "extrema"] = 1

        df['extrema'] = df['extrema'].rolling(
            window=self.TARGET_EXTREMA_WINDOW, win_type='gaussian', center=True).mean(std=0.5)

        # print(df['extrema'].value_counts())

        # Classify extrema
        df['&-extrema_maxima'] = np.where(df['extrema'] > 0, 'maxima', 'not_maxima')
        df['&-extrema_minima'] = np.where(df['extrema'] < 0, 'minima', 'not_minima')

        print(df['&-extrema_maxima'].value_counts())
        print(df['&-extrema_minima'].value_counts())

        # remove duplicated columns
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # remove last kernel rows
        # df.iloc[-kernel:, df.columns.get_loc("&-extrema_maxima")] = np.nan
        # df.iloc[-kernel:, df.columns.get_loc("&-extrema_minima")] = np.nan
        # df.iloc[-kernel:, df.columns.get_loc("&-trend_long")] = np.nan
        # df.iloc[-kernel:, df.columns.get_loc("&-trend_short")] = np.nan

        df.drop(columns=['open_log', 'low_log', 'high_log', 'close_log', 'hl2_log', 'hlc3_log', 'ohlc4_log', 'extrema'], inplace=True)
        self.log(f"EXIT .set_freqai_targets() {df.shape}, execution time: {time.time() - start_time:.2f} seconds")

        return df

    def add_slope_indicator(self, df: DataFrame, target_var = "ohlc4_log", predict_target = 6) -> DataFrame:
        df = df.set_index(df['date'])

        target = helpers.create_target(df, predict_target, method='polyfit', polyfit_var=target_var)
        target = target[['trend', 'slope', 'start_windows']].set_index('start_windows')
        target.fillna(0)

        # scale slope to 0-1
        target['slope'] = RobustScaler().fit_transform(target['slope'].values.reshape(-1, 1)).reshape(-1)

        target.rename(columns={'slope': f'{target_var}{predict_target}_exp_slope', 'trend': f'{target_var}{predict_target}_exp_trend'}, inplace=True)

        df = df.join(target[[f'{target_var}{predict_target}_exp_slope', f'{target_var}{predict_target}_exp_trend']], how='left')
        df = df.reset_index(drop=True)

        return df

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        self.log(f"ENTER .populate_indicators() {metadata} {df.shape}")
        start_time = time.time()

        df = self.freqai.start(df, metadata, self)

        df = candle_stats(df)

        # trend strength indicator
        df['trend_strength'] = df['trend_long'] - df['trend_short']
        df['trend_strength_abs'] = abs(df['trend_strength'])

        # add slope indicators
        df = self.add_slope_indicator(df, 'ohlc4_log', self.PREDICT_TARGET)

        df['L1'] = 1.0
        df['L0'] = 0
        df['L-1'] = -1.0

        # save df to file
        # df.to_csv("df_{}.csv".format(int(time.time())))

        last_candle = df.iloc[-1].squeeze()

        # self.log("Prediction values:")
        # self.log(f"{metadata['pair']}: do_predict={last_candle['do_predict']:.2f}")
        # self.log(f"{metadata['pair']}: minima={last_candle['minima']:.2f}")
        # self.log(f"{metadata['pair']}: maxima={last_candle['maxima']:.2f}")
        # self.log(f"{metadata['pair']}: trend_long={last_candle['trend_long']:.2f}")
        # self.log(f"{metadata['pair']}: trend_short={last_candle['trend_short']:.2f}")
        # self.log(f"{metadata['pair']}: trend_long_roc_auc={last_candle['&-trend_long_roc_auc']:.2f}")
        # self.log(f"{metadata['pair']}: trend_long_f1={last_candle['&-trend_long_f1']:.2f}")
        # self.log(f"{metadata['pair']}: trend_long_logloss={last_candle['&-trend_long_logloss']:.2f}")
        # self.log(f"{metadata['pair']}: trend_long_accuracy={last_candle['&-trend_long_accuracy']:.2f}")
        # self.log(f"{metadata['pair']}: trend_short_roc_auc={last_candle['&-trend_short_roc_auc']:.2f}")
        # self.log(f"{metadata['pair']}: trend_short_f1={last_candle['&-trend_short_f1']:.2f}")
        # self.log(f"{metadata['pair']}: trend_short_logloss={last_candle['&-trend_short_logloss']:.2f}")
        # self.log(f"{metadata['pair']}: trend_short_accuracy={last_candle['&-trend_short_accuracy']:.2f}")
        # self.log(f"{metadata['pair']}: extrema_maxima_roc_auc={last_candle['&-extrema_maxima_roc_auc']:.2f}")
        # self.log(f"{metadata['pair']}: extrema_maxima_f1={last_candle['&-extrema_maxima_f1']:.2f}")
        # self.log(f"{metadata['pair']}: extrema_maxima_logloss={last_candle['&-extrema_maxima_logloss']:.2f}")
        # self.log(f"{metadata['pair']}: extrema_maxima_accuracy={last_candle['&-extrema_maxima_accuracy']:.2f}")
        # self.log(f"{metadata['pair']}: extrema_minima_roc_auc={last_candle['&-extrema_minima_roc_auc']:.2f}")
        # self.log(f"{metadata['pair']}: extrema_minima_f1={last_candle['&-extrema_minima_f1']:.2f}")
        # self.log(f"{metadata['pair']}: extrema_minima_logloss={last_candle['&-extrema_minima_logloss']:.2f}")
        # self.log(f"{metadata['pair']}: extrema_minima_accuracy={last_candle['&-extrema_minima_accuracy']:.2f}")

        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=100)


        self.dp.send_msg(f"{metadata['pair']} predictions: \n  minima={last_candle['minima']:.2f}, \n  maxima={last_candle['maxima']:.2f}, \n  trend long={last_candle['trend_long']:.2f}, \n  trend short={last_candle['trend_short']:.2f}, \n  trend strength={last_candle['trend_strength']:.2f}")

        # ## save prediction to db
        # if (self.PREDICT_STORAGE_ENABLED and self.ps
        #         and (last_candle['do_predict'] == 1)):
        #     self.ps.save_prediction(
        #         model_version=self.freqai.identifier,
        #         pair=metadata['pair'],
        #         timeframe=self.timeframe,
        #         trend_long = float(last_candle['trend_long']),
        #         trend_short = float(last_candle['trend_short']),
        #         maxima = float(last_candle['maxima']),
        #         minima = float(last_candle['minima']),
        #         trend_long_roc_auc = float(last_candle['&-trend_long_roc_auc']),
        #         trend_long_f1 = float(last_candle['&-trend_long_f1']),
        #         trend_long_logloss = float(last_candle['&-trend_long_logloss']),
        #         trend_long_accuracy = float(last_candle['&-trend_long_accuracy']),
        #         trend_short_roc_auc = float(last_candle['&-trend_short_roc_auc']),
        #         trend_short_f1 = float(last_candle['&-trend_short_f1']),
        #         trend_short_logloss = float(last_candle['&-trend_short_logloss']),
        #         trend_short_accuracy = float(last_candle['&-trend_short_accuracy']),
        #         extrema_maxima_roc_auc = float(last_candle['&-extrema_maxima_roc_auc']),
        #         extrema_maxima_f1 = float(last_candle['&-extrema_maxima_f1']),
        #         extrema_maxima_logloss = float(last_candle['&-extrema_maxima_logloss']),
        #         extrema_maxima_accuracy = float(last_candle['&-extrema_maxima_accuracy']),
        #         extrema_minima_roc_auc = float(last_candle['&-extrema_minima_roc_auc']),
        #         extrema_minima_f1 = float(last_candle['&-extrema_minima_f1']),
        #         extrema_minima_logloss = float(last_candle['&-extrema_minima_logloss']),
        #         extrema_minima_accuracy = float(last_candle['&-extrema_minima_accuracy']),
        #         candle_time=last_candle['date']
        #     )

        self.log(f"EXIT populate_indicators {df.shape}, execution time: {time.time() - start_time:.2f} seconds")
        return df

    def protection_di(self, df: DataFrame):
        return (df["DI_values"] < df["DI_cutoff"])


    def signal_good_long(self, df: DataFrame):
        return (
            (df['minima'] >= 0.5) &
            (df['trend_long'] >= 0.8) &
            (df['maxima'] <= 0.3) &
            (df['trend_short'] <= 0.2)
        )

    def signal_super_long(self, df: DataFrame):
        return (
            (df['minima'] >= 0.8) &
            (df['trend_long'] >= 0.8) &
            (df['maxima'] <= 0.5) &
            (df['trend_short'] <= 0.3)
        )

    def signal_minima_pullback(self, df: DataFrame):
        return (
            (df['minima'] >= 0.9) &
            (df['trend_long'] >= 0.2) &
            (df['maxima'] <= 0.4) &
            (df['trend_short'] <= 0.4)
        )

    def signal_combo_long(self, df: DataFrame):
        return (
            (df['minima'] >= 0.7) &
            (df['trend_long'] >= 0.7) &
            (df['maxima'] <= 0.3) &
            (df['trend_short'] <= 0.3)
        )

    def scalp_long(self, df: DataFrame):
        return (
            (df['minima'] >= 0.2) &
            (df['trend_long'] >= 0.7) &
            (df['maxima'] <= 0.3) &
            (df['trend_short'] <= 0.3)
        )

    def signal_maxima_pullback(self, df: DataFrame):
        return (
            (df['maxima'] >= 0.7) &
            (df['trend_short'] >= 0.2) &
            (df['minima'] <= 0.3) &
            (df['trend_short'] <= 0.2)
        )

    def signal_strong_short(self, df: DataFrame):
        return (
            (df['maxima'] >= 0.4) &
            (df['trend_short'] >= 0.8) &
            (df['minima'] <= 0.5) &
            (df['trend_long'] <= 0.15)
        )


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        ## LONG signals

        # super long
        df.loc[(
            self.signal_super_long(df)
        ), ['enter_long', 'enter_tag']] = (1, 'super_long')

        # good long
        df.loc[(
            self.signal_good_long(df)
        ), ['enter_long', 'enter_tag']] = (1, 'good_long')

        # minima pullback
        df.loc[(
            self.signal_minima_pullback(df)
        ), ['enter_long', 'enter_tag']] = (1, 'minima_pullback')

        # combo long
        df.loc[(
            self.signal_combo_long(df)
        ), ['enter_long', 'enter_tag']] = (1, 'combo_long')

        # scalp long
        df.loc[(
            self.scalp_long(df)
        ), ['enter_long', 'enter_tag']] = (1, 'scalp_long')

        ## SHORT signals

        # maxima pullback
        df.loc[(
            self.signal_maxima_pullback(df)
        ), ['enter_short', 'enter_tag']] = (1, 'maxima_pullback')

        # strong short
        df.loc[(
            self.signal_strong_short(df)
        ), ['enter_short', 'enter_tag']] = (1, 'strong_short')


        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # LONGS
        df.loc[(
            self.signal_strong_short(df)
        ), ['exit_long', 'exit_tag']] = (1, 'exit_strong_short')

        df.loc[(
            self.signal_maxima_pullback(df)
        ), ['exit_long', 'exit_tag']] = (1, 'exit_maxima_pullback')

        # SHORTS
        df.loc[(
            self.signal_good_long(df)
        ), ['exit_short', 'exit_tag']] = (1, 'exit_good_long')

        df.loc[(
            self.signal_super_long(df)
        ), ['exit_short', 'exit_tag']] = (1, 'exit_super_long')

        df.loc[(
            self.signal_minima_pullback(df)
        ), ['exit_short', 'exit_tag']] = (1, 'exit_minima_pullback')

        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:

        # once the profit has risen above 2%, keep the stoploss at breakeven
        if current_profit >= 0.02:
            return stoploss_from_open(0.001, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        return 1


    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if df.empty:
            # Handle the empty DataFrame case
            return None  # Or other appropriate handling

        last_candle = df.iloc[-1].squeeze()

        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        is_short = trade.is_short == True
        is_long = trade.is_short == False
        is_profitable = current_profit > 0

        # 1. Check ROI
        roi_result = self.check_roi(pair, current_time, trade.open_date_utc, current_profit)
        if roi_result:
            return roi_result

        # 2. Check trend & extrema
        if is_long and last_candle['maxima'] >= 0.7 and is_profitable:
            return "almost_maxima"

        if is_long and last_candle['trend_short'] >= 0.7 and is_profitable:
            return "trend_reverse_to_short"

        if is_short and last_candle['minima'] >= 0.7 and is_profitable:
            return "almost_minima"

        if is_short and last_candle['trend_long'] >= 0.7 and is_profitable:
            return "trend_reserse_to_long"

        # 3. custom exit per enter_tag
        if trade.enter_tag == "scalp_long" and current_profit >= 0.015:
            return "scalp_long_target_reached"

        if is_short and current_profit >= 0.03:
            return "short_target_reached"


    ####
    # Dynamic ROI
    cached_roi_tables = {}

    def get_or_create_roi_table(self, pair, kernel=6):

        # Reset cache if new day
        if datetime.now().hour == 0:
            self.cached_roi_tables = {}

        # Check cache first
        if pair in self.cached_roi_tables:
            return self.cached_roi_tables[pair]

        # Get analyzed dataframe
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return None

        # Analyze candles to create ROI table
        min_peaks = argrelextrema(df["low"].values, np.less, order=kernel)[0]
        max_peaks = argrelextrema(df["high"].values, np.greater, order=kernel)[0]

        # Prepare lists for data
        distances = []
        candles_between_peaks = []

        # Iterate over low peaks to find the next high peak
        for low_peak in min_peaks:
            next_high_peaks = max_peaks[max_peaks > low_peak]
            if next_high_peaks.size > 0:
                high_peak = next_high_peaks[0]
                low_price = df.at[low_peak, 'close']
                high_price = df.at[high_peak, 'close']
                distance_percentage = ((high_price - low_price) / low_price)
                distances.append(distance_percentage)
                num_candles = high_peak - low_peak
                candles_between_peaks.append(num_candles)

        # Iterate over high peaks to find the next low peak
        for high_peak in max_peaks:
            next_low_peaks = min_peaks[min_peaks > high_peak]
            if next_low_peaks.size > 0:
                low_peak = next_low_peaks[0]
                high_price = df.at[high_peak, 'close']
                low_price = df.at[low_peak, 'close']
                distance_percentage = -((low_price - high_price) / high_price)
                distances.append(distance_percentage)
                num_candles = low_peak - high_peak
                candles_between_peaks.append(num_candles)

        if not distances or not candles_between_peaks:
            return None

        distances_description = pd.Series(distances).describe()
        candles_between_peaks_description = pd.Series(candles_between_peaks).describe()

        # Creating dynamic ROI table using calculated statistics
        minutes = timeframe_to_minutes(self.timeframe)
        dynamic_roi = {
            "0": distances_description['75%'], # trying to catch top 75% profit
            # str(int(candles_between_peaks_description['25%'] * minutes)): distances_description['50%'],
            # str(int(candles_between_peaks_description['50%'] * minutes)): distances_description['25%'],
            str(int(candles_between_peaks_description['75%'] * minutes)): 0.00  # closing the rest 25% which has lower profit
        }

        # Cache the ROI table
        self.cached_roi_tables[pair] = dynamic_roi
        return dynamic_roi


    def check_roi(self, pair, current_time, trade_open_date_utc, current_profit):
        dynamic_roi = self.get_or_create_roi_table(pair, kernel=self.TARGET_EXTREMA_KERNEL)
        if not dynamic_roi:
            return None

        # print("dymanic roi for pair:", pair)
        # print(dynamic_roi)

        trade_duration = (current_time - trade_open_date_utc).seconds / 60
        for roi_time, roi_value in dynamic_roi.items():
            if trade_duration >= int(roi_time) and current_profit >= roi_value:
                # print(f"ROI reached: {roi_value} at {roi_time} minutes")
                return "dynamic_roi"

        return None


    # def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
    #                         time_in_force: str, current_time: datetime, entry_tag: Optional[str],
    #                         side: str, **kwargs) -> bool:

    #     if self.config.get('runmode') in (RunMode.DRY_RUN, RunMode.LIVE):
    #         self.log(f"ENTER confirm_trade_entry() {pair}, {current_time}, {rate}, {entry_tag}, {side}")

    #     # if not enabled, exit with True
    #     if (not self.config['sagemaster'].get('enabled', False)):
    #         return True

    #     # get client and load params
    #     sgm = SageMasterClient(self.config['sagemaster']['webhook_api_key'], self.config['sagemaster']['webhook_url'], self.config['sagemaster']['trader_nickname'])

    #     [market, symbol_base, symbol_quote] = helpers.extract_currencies(pair)
    #     deal_type = 'buy' if side == 'long' else 'sell'
    #     tp_tip = round(self.LONG_TP.value * 100, 4) if side == 'long' else round(self.SHORT_TP.value * 100, 4)
    #     sl_tip = round(self.stoploss * 100, 4)

    #     # generate trade_id, which is +1 to last trade in db
    #     trade_id = "1"
    #     trade = Trade.get_trades(None).order_by(desc(Trade.open_date)).first()
    #     if (trade):
    #         trade_id = str(trade.id + 1)

    #     # convert trade_id to uuid
    #     trade_id = helpers.get_uuid_from_key(str(trade_id))

    #     sgm.open_deal(
    #         market=market,
    #         symbol_base=symbol_base,
    #         symbol_quote=symbol_quote,
    #         deal_type=deal_type,
    #         buy_price=rate,
    #         tp_tip=tp_tip,
    #         sl_tip=sl_tip,
    #         trade_id=trade_id
    #     )

    #     return True

    # def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
    #                        rate: float, time_in_force: str, exit_reason: str,
    #                        current_time: datetime, **kwargs) -> bool:

    #     if self.config.get('runmode') in (RunMode.DRY_RUN, RunMode.LIVE):
    #         self.log(f"ENTER confirm_trade_entry() {pair}, {current_time}, {rate}")

    #     # if not enabled, exit with True
    #     if (not self.config['sagemaster'].get('enabled', False)):
    #         return True

    #     sgm = SageMasterClient(self.config['sagemaster']['webhook_api_key'], self.config['sagemaster']['webhook_url'], self.config['sagemaster']['trader_nickname'])

    #     [market, symbol_base, symbol_quote] = helpers.extract_currencies(pair)
    #     tp_tip = round(self.LONG_TP.value * 100, 4) if trade.is_short == False else round(self.SHORT_TP.value * 100, 4)
    #     sl_tip = round(self.stoploss * 100, 4)
    #     profit_ratio = trade.calc_profit_ratio(rate)
    #     deal_type = 'buy' if trade.is_short == False else 'sell'
    #     trade_id = helpers.get_uuid_from_key(str(trade.id))
    #     allow_stoploss = self.config['sagemaster'].get('allow_stoploss', False)

    #     sgm.close_deal(
    #         market=market,
    #         symbol_base=symbol_base,
    #         symbol_quote=symbol_quote,
    #         deal_type=deal_type,
    #         buy_price=rate,
    #         tp_tip=tp_tip,
    #         sl_tip=sl_tip,
    #         trade_id=trade_id,
    #         profit_ratio=profit_ratio,
    #         allow_stoploss=allow_stoploss
    #     )

    #     return True
