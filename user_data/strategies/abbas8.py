from freqtrade.strategy import (IStrategy, informative)
from typing import Dict, List
from pandas import DataFrame, Series
import pandas_ta as pta
import numpy as np
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
import logging
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter, stoploss_from_open
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real
from datetime import datetime, timedelta, timezone
from technical.indicators import RMI,vwmacd

logger = logging.getLogger(__name__)

# Buy hyperspace params:
buy_params = {
    "buy_bb20_close_bblowerband": 0.98,
    "buy_bb20_volume": 25,
    "buy_bb40_bbdelta_close": 0.038,
    "buy_bb40_closedelta_close": 0.015,
    "buy_bb40_tail_bbdelta": 0.3,
    "base_nb_candles_buy": 48,  # value loaded from strategy
    "buy_bb_delta": 0.034,  # value loaded from strategy
    "buy_bb_factor": 0.994,  # value loaded from strategy
    "buy_bb_width": 0.077,  # value loaded from strategy
    "buy_c10_1": -110,  # value loaded from strategy
    "buy_c10_2": -0.98,  # value loaded from strategy
    "buy_c2_1": 0.024,  # value loaded from strategy
    "buy_c2_2": 0.986,  # value loaded from strategy
    "buy_c2_3": -0.7,  # value loaded from strategy
    "buy_c6_1": 0.17,  # value loaded from strategy
    "buy_c6_2": 0.18,  # value loaded from strategy
    "buy_c6_3": 0.038,  # value loaded from strategy
    "buy_c6_4": 0.014,  # value loaded from strategy
    "buy_c6_5": 0.34,  # value loaded from strategy
    "buy_c7_1": 1.01,  # value loaded from strategy
    "buy_c7_2": 1.04,  # value loaded from strategy
    "buy_c7_3": -89,  # value loaded from strategy
    "buy_c7_4": -61,  # value loaded from strategy
    "buy_c7_5": 81,  # value loaded from strategy
    "buy_c9_1": 37,  # value loaded from strategy
    "buy_c9_2": -71,  # value loaded from strategy
    "buy_c9_3": -68,  # value loaded from strategy
    "buy_c9_4": 54,  # value loaded from strategy
    "buy_c9_5": 31,  # value loaded from strategy
    "buy_c9_6": 65,  # value loaded from strategy
    "buy_c9_7": -92,  # value loaded from strategy
    "buy_cci": -134,  # value loaded from strategy
    "buy_cci_length": 40,  # value loaded from strategy
    "buy_closedelta": 16.3,  # value loaded from strategy
    "buy_clucha_bbdelta_close": 0.037,
    "buy_clucha_bbdelta_tail": 0.7,
    "buy_clucha_closedelta_close": 0.006,
    "buy_clucha_rocr_1h": 0.93,
    "buy_con3_1": 0.023,  # value loaded from strategy
    "buy_con3_2": 0.995,  # value loaded from strategy
    "buy_con3_3": 0.958,  # value loaded from strategy
    "buy_con3_4": -0.87,  # value loaded from strategy
    "buy_dip_threshold_1": 0.28,  # value loaded from strategy
    "buy_dip_threshold_2": 0.3,  # value loaded from strategy
    "buy_dip_threshold_3": 0.4,  # value loaded from strategy
    "buy_dip_threshold_5": 0.024,  # value loaded from strategy
    "buy_dip_threshold_6": 0.061,  # value loaded from strategy
    "buy_dip_threshold_7": 0.07,  # value loaded from strategy
    "buy_dip_threshold_8": 0.214,  # value loaded from strategy
    "buy_ema_open_mult_1": 0.02,  # value loaded from strategy
    "buy_macd_41": 0.02,  # value loaded from strategy
    "buy_mfi": 36.0,  # value loaded from strategy
    "buy_mfi_1": 39,  # value loaded from strategy
    "buy_min_inc": 0.01,  # value loaded from strategy
    "buy_min_inc_1": 0.015,  # value loaded from strategy
    "buy_pump_pull_threshold_1": 2.16,  # value loaded from strategy
    "buy_pump_threshold_1": 0.751,  # value loaded from strategy
    "buy_rmi": 32,  # value loaded from strategy
    "buy_rmi_length": 19,  # value loaded from strategy
    "buy_rsi": 38.5,  # value loaded from strategy
    "buy_rsi_1": 38,  # value loaded from strategy
    "buy_rsi_1h": 67.0,  # value loaded from strategy
    "buy_rsi_1h_42": 45,  # value loaded from strategy
    "buy_rsi_1h_max_1": 88,  # value loaded from strategy
    "buy_rsi_1h_min_1": 26,  # value loaded from strategy
    "buy_rsi_diff": 50.48,  # value loaded from strategy
    "buy_srsi_fk": 39,  # value loaded from strategy
    "buy_volume_1": 2.0,  # value loaded from strategy
    "buy_volume_drop_41": 1.8,  # value loaded from strategy
    "buy_volume_pump_41": 0.3,  # value loaded from strategy
    "buy_vwap_closedelta": 10.19,  # value loaded from strategy
    "buy_vwap_cti": -0.06,  # value loaded from strategy
    "buy_vwap_width": 2.72,  # value loaded from strategy
    "bzv7_buy_macd_1": 0.02,  # value loaded from strategy
    "bzv7_buy_macd_2": 0.03,  # value loaded from strategy
    "bzv7_buy_rsi_1": 28,  # value loaded from strategy
    "bzv7_buy_rsi_1h_1": 16,  # value loaded from strategy
    "bzv7_buy_rsi_1h_2": 15,  # value loaded from strategy
    "bzv7_buy_rsi_1h_3": 20,  # value loaded from strategy
    "bzv7_buy_rsi_1h_4": 35,  # value loaded from strategy
    "bzv7_buy_rsi_1h_5": 39,  # value loaded from strategy
    "bzv7_buy_rsi_2": 10,  # value loaded from strategy
    "bzv7_buy_rsi_3": 14,  # value loaded from strategy
    "bzv7_buy_volume_drop_1": 9.3,  # value loaded from strategy
    "bzv7_buy_volume_drop_3": 1.0,  # value loaded from strategy
    "bzv7_buy_volume_pump_1": 0.1,  # value loaded from strategy
    "ewo_high": 4.03,  # value loaded from strategy
    "ewo_high_2": 0.6,  # value loaded from strategy
    "ewo_low": -13.15,  # value loaded from strategy
    "fast_ewo": 28,  # value loaded from strategy
    "low_offset": 1.03,  # value loaded from strategy
    "low_offset_2": 0.95,  # value loaded from strategy
    "rsi_buy": 70,  # value loaded from strategy
    "rsi_ewo2": 34,  # value loaded from strategy
    "rsi_fast_ewo1": 29,  # value loaded from strategy
    "slow_ewo": 212  # value loaded from strategy
}
# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 23,
    "high_offset": 1.01
}

def EWO(dataframe, ema_length=5, ema2_length=35):
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe["low"] * 100

# VWAP bands
def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df["vwap"] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df["vwap"].rolling(window=window_size).std()
    df["vwap_low"] = df["vwap"] - (rolling_std * num_of_std)
    df["vwap_high"] = df["vwap"] + (rolling_std * num_of_std)
    return df["vwap_low"], df["vwap"], df["vwap_high"]

def ha_typical_price(bars):
    res = (bars["ha_high"] + bars["ha_low"] + bars["ha_close"]) / 3.
    return Series(index=bars.index, data=res)

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc

# SSL Channels
def SSLChannels(dataframe, length=7):
    df = dataframe.copy()
    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN))
    df["hlv"] = df["hlv"].ffill()
    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["sslDown"], df["sslUp"]

# Chaikin Money Flow Volume
def MFV(dataframe):
    df = dataframe.copy()
    N = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    M = N * df["volume"]
    return M


class abbas8(IStrategy):
    def version(self) -> str:
        return "v9.8.6"
    INTERFACE_VERSION = 3
    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.12, -0.03, decimals=3, name="stoploss")]
        # Define custom trailing space
        def trailing_space() -> List[Dimension]:
            return[
                Categorical([True], name="trailing_stop"),
                SKDecimal(0.0002, 0.0020, decimals=4, name="trailing_stop_positive"),
                SKDecimal(0.010,  0.030, decimals=3, name="trailing_stop_positive_offset_p1"),
                Categorical([True], name="trailing_only_offset_is_reached"),
            ]
        # Define custom ROI space
        def roi_space() -> List[Dimension]:
            return [
                Integer(  6, 60, name="roi_t1"),
                Integer( 60, 120, name="roi_t2"),
                Integer(120, 200, name="roi_t3")
            ]
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            roi_table = {}
            roi_table[params["roi_t1"]] = 0
            roi_table[params["roi_t2"]] = -0.015
            roi_table[params["roi_t3"]] = -0.030
            return roi_table
    timeframe = "5m"
    info_timeframes = ["15m", "30m", "1h"]
    minimal_roi = {
        "99": 0,
        "140": -0.015,
        "232": -0.03
    }
    order_time_in_force = {
        "entry": "gtc",
        "exit": "ioc"
    }
    slippage_protection = {
        "retries": 3,
        "max_slippage": -0.002
    }
    stoploss = -0.067
    trailing_stop = True
    trailing_stop_positive = 0.0004
    trailing_stop_positive_offset = 0.0174
    trailing_only_offset_is_reached = True
    use_exit_signal = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True
    startup_candle_count = 449

    smaoffset_optimize = False
    base_nb_candles_buy = IntParameter(20, 55, default=buy_params["base_nb_candles_buy"], space="buy", optimize=smaoffset_optimize)
    base_nb_candles_sell = IntParameter(8, 30, default=sell_params["base_nb_candles_sell"], space="sell", optimize=smaoffset_optimize)
    low_offset = DecimalParameter(0.90, 1.25, default=buy_params["low_offset"], space="buy", decimals=2, optimize=smaoffset_optimize)
    low_offset_2 = DecimalParameter(0.90, 1.25, default=buy_params["low_offset_2"], space="buy", decimals=2, optimize=smaoffset_optimize)
    high_offset = DecimalParameter(0.90, 1.30, default=sell_params["high_offset"], space="sell", decimals=2, optimize=smaoffset_optimize)

    ewo_optimize = False
    fast_ewo = IntParameter(5,30, default=buy_params["fast_ewo"], space="buy", optimize=ewo_optimize)
    slow_ewo = IntParameter(120,250, default=buy_params["slow_ewo"], space="buy", optimize=ewo_optimize)
    rsi_fast_ewo1 = IntParameter(40, 70, default=buy_params["rsi_fast_ewo1"], space="buy", optimize=ewo_optimize)
    rsi_ewo2 = IntParameter(10, 40, default=buy_params["rsi_ewo2"], space="buy", optimize=ewo_optimize)

    protection_optimize = False
    ewo_low = DecimalParameter(-20.0, -4.0, default=buy_params["ewo_low"], space="buy", decimals=2, optimize=protection_optimize)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params["ewo_high"], space="buy", decimals=2, optimize=protection_optimize)
    ewo_high_2 = DecimalParameter(-7.0, 13.0, default=buy_params["ewo_high_2"], space="buy", decimals=2, optimize=protection_optimize)
    rsi_buy = IntParameter(45, 75, default=buy_params["rsi_buy"], space="buy", optimize=protection_optimize)

    is_optimize_clucha = False
    buy_clucha_bbdelta_close    = DecimalParameter(0.010, 0.060,  default=buy_params["buy_clucha_bbdelta_close"],    space="buy", decimals=3, optimize = is_optimize_clucha)
    buy_clucha_bbdelta_tail     = DecimalParameter(0.40,   1.00,  default=buy_params["buy_clucha_bbdelta_tail"],     space="buy", decimals=2, optimize = is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.001,  0.030, default=buy_params["buy_clucha_closedelta_close"], space="buy", decimals=3, optimize = is_optimize_clucha)
    buy_clucha_rocr_1h          = DecimalParameter(0.050,   1.00, default=buy_params["buy_clucha_rocr_1h"],          space="buy", decimals=2, optimize = is_optimize_clucha)

    is_optimize_vwap = False
    buy_vwap_width      = DecimalParameter(0.5, 10.0,    default=0.80,  space="buy", decimals=1, optimize = is_optimize_vwap)
    buy_vwap_closedelta = DecimalParameter(10.0, 30.0,   default=15.0,  space="buy", decimals=1, optimize = is_optimize_vwap)
    buy_vwap_cti        = DecimalParameter(-0.90, -0.00, default=-0.60, space="buy", decimals=2, optimize = is_optimize_vwap)

    # BeastBotXBLR
    ###########################################################################
    # Buy
    optc1 = False
    buy_rmi_length = IntParameter(8, 20,     default=buy_params["buy_rmi_length"],    space='buy', optimize= optc1) # 20-8 = 12
    buy_cci_length = IntParameter(25, 45,    default=buy_params["buy_cci_length"],    space='buy', optimize= optc1) # 45-25 = 20
    buy_cci        = IntParameter(-135, -90, default=buy_params["buy_cci"],           space='buy', optimize= optc1) # -90-(-135) = 45
    buy_srsi_fk    = IntParameter(30, 50,    default=buy_params["buy_srsi_fk"],       space='buy', optimize= optc1) # 50-30 = 20
    buy_bb_width   = DecimalParameter(0.065, 0.135, default=buy_params["buy_bb_width"], space='buy', decimals=3, optimize = optc1) # 0.135-0.065 = 0.070 | 70
    buy_bb_delta   = DecimalParameter(0.018, 0.035, default=buy_params["buy_bb_delta"], space='buy', decimals=3, optimize = optc1) # 0.035-0.018 = 0.017 | 17
    buy_bb_factor  = DecimalParameter(0.990, 0.999, default=buy_params["buy_bb_factor"], space='buy', decimals=3, optimize = optc1) # 0.999-0.990 = 0.009 | 9
    buy_closedelta = DecimalParameter( 12.0, 18.0,  default=buy_params["buy_closedelta"], space='buy', decimals=1, optimize = optc1) # 18.0-12.0 = 6.0 | 60

    optc2 = False
    buy_c2_1 = DecimalParameter(0.010, 0.025, default=buy_params["buy_c2_1"], space='buy', decimals=3, optimize=optc2) # 0.025-0.010 = 0.015 | 15
    buy_c2_2 = DecimalParameter(0.980, 0.995, default=buy_params["buy_c2_2"], space='buy', decimals=3, optimize=optc2) # 0.995-0.980 = 0.015 | 15
    buy_c2_3 = DecimalParameter(-0.8, -0.3,   default=buy_params["buy_c2_3"], space='buy', decimals=1, optimize=optc2) # -0.3-(-0.8) = 0.5 | 5
    
    optc3 = False
    buy_con3_1 = DecimalParameter(0.010, 0.025, default=buy_params["buy_con3_1"], space='buy', decimals=3, optimize=optc3) # 0.025-0.010 = 0.015 | 15
    buy_con3_2 = DecimalParameter(0.980, 0.995, default=buy_params["buy_con3_2"], space='buy', decimals=3, optimize=optc3) # 0.995-0.980 = 0.015 | 15
    buy_con3_3 = DecimalParameter(0.955, 0.975, default=buy_params["buy_con3_3"], space='buy', decimals=3, optimize=optc3) # 0.975-0.955 = 0.020 | 20
    buy_con3_4 = DecimalParameter(-0.95, -0.70, default=buy_params["buy_con3_4"], space='buy', decimals=2, optimize=optc3) # -0.70-(-0.95) = 0.25 | 25

    optc4 = False
    buy_rsi_1h_42      = IntParameter(10, 50,         default=buy_params["buy_rsi_1h_42"], space='buy', optimize=optc4) # 50-10 = 40
    buy_macd_41        = DecimalParameter(0.01, 0.09, default=buy_params["buy_macd_41"], space='buy', decimals=2, optimize=optc4) # 0.09-0.01 = 0.08 | 8
    buy_volume_pump_41 = DecimalParameter(0.1, 0.9,   default=buy_params["buy_volume_pump_41"], space='buy', decimals=1, optimize=optc4) # 0.9-0.1 = 0.8 | 8
    buy_volume_drop_41 = DecimalParameter(1, 10,      default=buy_params["buy_volume_drop_41"], space='buy', decimals=1, optimize=optc4) # 10-1 = 9 | 90

    optc7 = False
    buy_c7_1 = DecimalParameter(0.95, 1.10, default=buy_params["buy_c7_1"], space='buy', decimals=2, optimize=optc7) # 1.10-0.95 = 0.15 | 15
    buy_c7_2 = DecimalParameter(0.95, 1.10, default=buy_params["buy_c7_2"], space='buy', decimals=2, optimize=optc7) # 1.10-0.95 = 0.15 | 15
    buy_c7_3 = IntParameter(-100, -80, default=buy_params["buy_c7_3"], space='buy', optimize=optc7) # -80-(-100) = 20
    buy_c7_4 = IntParameter(-90, -60,  default=buy_params["buy_c7_4"], space='buy', optimize=optc7) # -60-(-90) = 30
    buy_c7_5 = IntParameter(75, 90,    default=buy_params["buy_c7_5"], space='buy', optimize=optc7) # 90-75 = 15

    optc8 = False
    buy_min_inc_1 = DecimalParameter(0.010, 0.050, default=buy_params["buy_min_inc_1"], space='buy', decimals=3, optimize=optc8) # 0.050-0.010 = 0.040 | 40
    buy_rsi_1h_min_1 = IntParameter(25, 40, default=buy_params["buy_rsi_1h_min_1"], space='buy', optimize=optc8) # 40-25 = 15
    buy_rsi_1h_max_1 = IntParameter(70, 90, default=buy_params["buy_rsi_1h_max_1"], space='buy', optimize=optc8) # 90-70 = 20
    buy_rsi_1        = IntParameter(20, 40, default=buy_params["buy_rsi_1"], space='buy', optimize=optc8) # 40-20 = 20
    buy_mfi_1        = IntParameter(20, 40, default=buy_params["buy_mfi_1"], space='buy', optimize=optc8) # 40-20 = 20

    optc9 = False
    buy_c9_1 = IntParameter(25, 44,   default=buy_params["buy_c9_1"], space='buy', optimize=optc9) # 44-25 = 19
    buy_c9_2 = IntParameter(-80, -67, default=buy_params["buy_c9_2"], space='buy', optimize=optc9) # -67-(-80) = 13
    buy_c9_3 = IntParameter(-80, -67, default=buy_params["buy_c9_3"], space='buy', optimize=optc9) # -67-(-80) = 13
    buy_c9_4 = IntParameter(35, 54,   default=buy_params["buy_c9_4"], space='buy', optimize=optc9) # 54-35 = 19
    buy_c9_5 = IntParameter(20, 44,   default=buy_params["buy_c9_5"], space='buy', optimize=optc9) # 44-20 = 24
    buy_c9_6 = IntParameter(65, 94,   default=buy_params["buy_c9_6"], space='buy', optimize=optc9) # 94-65 = 29
    buy_c9_7 = IntParameter(-110, -80, default=buy_params["buy_c9_7"], space='buy', optimize=optc9) # -80-(-110) = 30

    optc10 = False
    buy_c10_1 = IntParameter(-110, -80,        default=buy_params["buy_c10_1"], space='buy', optimize=optc10) # -80-(-110) = 30
    buy_c10_2 = DecimalParameter(-1.00, -0.50, default=buy_params["buy_c10_2"], space='buy', decimals=2, optimize=optc10) # -0.50-(-1.00) = 0.50 | 50

    dip_optimize = False
    buy_dip_threshold_5 = DecimalParameter(0.020, 0.070, default=buy_params["buy_dip_threshold_5"], space='buy', decimals=3, optimize=dip_optimize) ## 0.070-0.020 = 0.050 | 50
    buy_dip_threshold_6 = DecimalParameter(0.050, 0.100, default=buy_params["buy_dip_threshold_6"], space='buy', decimals=3, optimize=dip_optimize) # 0.100-0.050 = 0.050 | 50
    buy_dip_threshold_7 = DecimalParameter(0.050, 0.100, default=buy_params["buy_dip_threshold_7"], space='buy', decimals=3, optimize=dip_optimize) # 0.100-0.050 = 0.050 | 50
    buy_dip_threshold_8 = DecimalParameter(0.150, 0.250, default=buy_params["buy_dip_threshold_8"], space='buy', decimals=3, optimize=dip_optimize) # 0.250-0.150 = 0.100 | 100
    # 24 hours
    buy_pump_optimize = False
    buy_pump_pull_threshold_1 = DecimalParameter(1.50, 3.00, default=buy_params["buy_pump_pull_threshold_1"], space='buy', decimals=2, optimize=buy_pump_optimize) # 3.00-1.50 = 1.50 | 150
    buy_pump_threshold_1    = DecimalParameter(0.600, 1.000, default=buy_params["buy_pump_threshold_1"], space='buy', decimals=3, optimize=buy_pump_optimize) # 1.000-0.600 = 0.400 | 400

    #  Strategy: BigZ07
    # Buy HyperParam
    bzv7_buy_volume_pump_1 = DecimalParameter(0.1, 0.9, default=0.4, space="buy", decimals=1, optimize=True)
    bzv7_buy_volume_drop_1 = DecimalParameter(1, 10,    default=3.8, space="buy", decimals=1, optimize=True)
    bzv7_buy_volume_drop_3 = DecimalParameter(1, 10,    default=2.7, space="buy", decimals=1, optimize=True)

    bzv7_rsi_optimize = True
    bzv7_buy_rsi_1h_1 = IntParameter(8, 30, default=16, space="buy", optimize=bzv7_rsi_optimize)
    bzv7_buy_rsi_1h_2 = IntParameter(20, 45, default=15, space="buy", optimize=bzv7_rsi_optimize)
    bzv7_buy_rsi_1h_3 = IntParameter(20, 45, default=20, space="buy", optimize=bzv7_rsi_optimize)
    bzv7_buy_rsi_1h_4 = IntParameter(10, 30, default=35, space="buy", optimize=bzv7_rsi_optimize)
    bzv7_buy_rsi_1h_5 = IntParameter(30, 65, default=39, space="buy", optimize=bzv7_rsi_optimize)

    bzv7_buy_rsi_1    = IntParameter(4, 20,  default=28, space="buy", optimize=True) # 40-7 = 33
    bzv7_buy_rsi_2    = IntParameter(4, 20,  default=10, space="buy", optimize=True) # 40-7 = 33
    bzv7_buy_rsi_3    = IntParameter(4, 20,  default=14, space="buy", optimize=True) # 40-7 = 33

    bzv7_buy_macd_1 = DecimalParameter(0.01, 0.09, default=buy_params["buy_macd_41"], space="buy", decimals=2, optimize=True) # 0.09-0.01 = 0.08 | 8
    bzv7_buy_macd_2 = DecimalParameter(0.001, 0.030, default=buy_params["buy_macd_41"], space="buy", decimals=3, optimize=True) # 0.09-0.01 = 0.08 | 8

    buy_dip_threshold_optimize = True
    buy_dip_threshold_1 = DecimalParameter(0.20, 0.40, default=buy_params["buy_dip_threshold_1"], space="buy", decimals=2, optimize=buy_dip_threshold_optimize)
    buy_dip_threshold_2 = DecimalParameter(0.20, 0.50, default=buy_params["buy_dip_threshold_2"], space="buy", decimals=2, optimize=buy_dip_threshold_optimize)
    buy_dip_threshold_3 = DecimalParameter(0.30, 0.60, default=buy_params["buy_dip_threshold_3"], space="buy", decimals=2, optimize=buy_dip_threshold_optimize)

    bb40_optimize = True
    buy_bb40_bbdelta_close     = DecimalParameter(0.025, 0.045, default=buy_params["buy_bb40_bbdelta_close"],       space="buy", decimals=3, optimize=bb40_optimize)
    buy_bb40_closedelta_close  = DecimalParameter(0.010, 0.030, default=buy_params["buy_bb40_closedelta_close"],    space="buy", decimals=3, optimize=bb40_optimize)
    buy_bb40_tail_bbdelta      = DecimalParameter(0.250, 0.350, default=buy_params["buy_bb40_tail_bbdelta"],        space="buy", decimals=3, optimize=bb40_optimize)
    buy_bb20_close_bblowerband = DecimalParameter(0.950, 1.050, default=buy_params["buy_bb20_close_bblowerband"],   space="buy", decimals=3, optimize=bb40_optimize)
    buy_bb20_volume = IntParameter(18, 36, default=buy_params["buy_bb20_volume"], space="buy", optimize=bb40_optimize)
    
    bzv7_rsimfi_optimize = True
    buy_rsi_diff        =  IntParameter(34, 60, default=buy_params["buy_rsi_diff"], space="buy",  optimize=bzv7_rsimfi_optimize)
    buy_rsi_1h          =  IntParameter(40, 70, default=buy_params["buy_rsi_1h"], space="buy",  optimize=bzv7_rsimfi_optimize)
    buy_rsi             =  IntParameter(30, 40, default=buy_params["buy_rsi"], space="buy",  optimize=bzv7_rsimfi_optimize)
    buy_mfi             =  IntParameter(36, 65, default=buy_params["buy_mfi"], space="buy",  optimize=bzv7_rsimfi_optimize)
    buy_volume_1        =  IntParameter(1, 10,  default=buy_params["buy_volume_1"], space="buy",  optimize=bzv7_rsimfi_optimize)
    buy_min_inc         =  DecimalParameter(0.005, 0.050, default=buy_params["buy_min_inc"], space="buy", decimals=3, optimize=bzv7_rsimfi_optimize)
    buy_ema_open_mult_1 =  DecimalParameter(0.010, 0.050, default=buy_params["buy_ema_open_mult_1"], space="buy", decimals=3, optimize=bzv7_rsimfi_optimize)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for info_timeframe in self.info_timeframes:
            informative_pairs.extend([(pair, info_timeframe) for pair in pairs])
        return informative_pairs
    
    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")

        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)

        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)
        #informative_1h['not_downtrend'] = ((informative_1h['close'] > informative_1h['close'].shift(2)) | (informative_1h['rsi'] > 50))
        informative_1h['r_480'] = williams_r(dataframe, period=480)
        informative_1h['safe_pump_24'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) /
            informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_1.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) /
            self.buy_pump_pull_threshold_1.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))

        informative_1h['cti'] = pta.cti(informative_1h["close"], length=20) 

        ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
        informative_1h["ssl_down"] = ssl_down_1h
        informative_1h["ssl_up"] = ssl_up_1h
        informative_1h["ssl-dir"] = np.where(ssl_up_1h > ssl_down_1h, "up", "down")

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        informative_1h["bb_lowerband"] = bollinger["lower"]
        informative_1h["bb_middleband"] = bollinger["mid"]
        informative_1h["bb_upperband"] = bollinger["upper"]

        # Heikin Ashi
        inf_heikinashi = qtpylib.heikinashi(informative_1h)
        informative_1h["ha_close"] = inf_heikinashi["close"]
        informative_1h["rocr"] = ta.ROCR(informative_1h["ha_close"], timeperiod=168)

        return informative_1h
    
    def informative_30m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_30m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="30m")    
        return informative_30m
    
    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="15m")
        return informative_15m

    def base_tf_5m_indicators(self, metadata: dict, dataframe: DataFrame) -> DataFrame:
        dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_buy.value))
        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_sell.value))
        dataframe["ewo"] = EWO(dataframe, int(self.fast_ewo.value), int(self.slow_ewo.value))
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)
        dataframe["volume_mean_slow_30"] = dataframe["volume"].rolling(window=30).mean()

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        # nuevo #
        
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe['bb_delta'] = ((dataframe['bb_lowerband'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband'])
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        bb_40 = qtpylib.bollinger_bands(dataframe["close"], window=40, stds=2)
        dataframe["lower"] = bb_40["lower"]
        dataframe["mid"] = bb_40["mid"]
        dataframe["bbdelta"] = (bb_40["mid"] - dataframe["lower"]).abs()

        # MACD
        dataframe["macd"], dataframe["signal"], dataframe["hist"] = ta.MACD(dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9)

        # Chaikin A/D Oscillator
        dataframe["mfv"] = MFV(dataframe)
        dataframe["cmf"] = (dataframe["mfv"].rolling(20).sum() / dataframe["volume"].rolling(20).sum())

        # CCI hyperopt
        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cci'] = ta.CCI(dataframe, 26)

        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)

        # SRSI hyperopt ?
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']

        # Volume
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=48).mean()

        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_32'] = williams_r(dataframe, period=32)
        dataframe['r_64'] = williams_r(dataframe, period=64)
        dataframe["sma_5"] = ta.SMA(dataframe, timeperiod=5)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        
        dataframe['safe_dips_strict'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_5.value) &
                                  (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_6.value) &
                                  (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_7.value) &
                                  (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_8.value))

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["ha_high"] = heikinashi["high"]
        dataframe["ha_low"] = heikinashi["low"]
        ## BB 40
        bollinger2_40 = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=40, stds=2)
        dataframe["bb_lowerband2_40"] = bollinger2_40["lower"]
        dataframe["bb_middleband2_40"] = bollinger2_40["mid"]
        # RSI
        dataframe["rsi_84"] = ta.RSI(dataframe, timeperiod=84)
        dataframe["rsi_112"] = ta.RSI(dataframe, timeperiod=112)
        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe["vwap_upperband"] = vwap_high
        dataframe["vwap_middleband"] = vwap
        dataframe["vwap_lowerband"] = vwap_low
        dataframe["vwap_width"] = ( (dataframe["vwap_upperband"] - dataframe["vwap_lowerband"]) / dataframe["vwap_middleband"] ) * 100
        # ClucHA
        dataframe["bb_delta_cluc"] = (dataframe["bb_middleband2_40"] - dataframe["bb_lowerband2_40"]).abs()
        dataframe["ha_closedelta"] = (dataframe["ha_close"] - dataframe["ha_close"].shift()).abs()
        dataframe["ha_tail"] = (dataframe["ha_close"] - dataframe["ha_low"]).abs()
        dataframe["rocr"] = ta.ROCR(dataframe["ha_close"], timeperiod=28)
        # CTI
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)
        # BinH
        dataframe["closedelta"] = (dataframe["close"] - dataframe["close"].shift()).abs()
        return dataframe
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, sell_reason: str, current_time: datetime, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        try:
            state = self.slippage_protection["__pair_retries"]
        except KeyError:
            state = self.slippage_protection["__pair_retries"] = {}
        candle = dataframe.iloc[-1].squeeze()
        slippage = (rate / candle["close"]) - 1
        if slippage < self.slippage_protection["max_slippage"]:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection["retries"]:
                state[pair] = pair_retries + 1
                return False
        state[pair] = 0
        return True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe       = self.base_tf_5m_indicators(metadata, dataframe)
        informative_1h  = self.informative_1h_indicators(dataframe, metadata)
        dataframe       = merge_informative_pair(dataframe, informative_1h, self.timeframe, "1h", ffill=True)
        informative_30m = self.informative_30m_indicators(dataframe, metadata)
        dataframe       = merge_informative_pair(dataframe, informative_30m, self.timeframe, "30m", ffill=True)
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe       = merge_informative_pair(dataframe, informative_15m, self.timeframe, "15m", ffill=True)

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_tag"] = ""
        dataframe.loc[
            (
                (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
                (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
                (dataframe["ewo"] > self.ewo_high.value) &
                (dataframe["rsi"] < self.rsi_buy.value) &
                (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
            ),
            ["enter_long", "enter_tag"]] = (1, "ewo1")
        dataframe.loc[
            (
                (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
                (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
                (dataframe["ewo"] < self.ewo_low.value) &
                (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
            ),
            ["enter_long", "enter_tag"]] = (1, "ewolow")
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["vwap_lowerband"]) &
                (dataframe["vwap_width"] > self.buy_vwap_width.value) &
                (dataframe["closedelta"] > dataframe["close"] * self.buy_vwap_closedelta.value / 1000 ) &
                (dataframe["cti"] < self.buy_vwap_cti.value) &
                (dataframe["rsi_84"] < 60) &
                (dataframe["rsi_112"] < 60)
            ),
            ["enter_long", "enter_tag"]] = (1, "vwap")
        dataframe.loc[
            (
                (dataframe["rocr_1h"] > self.buy_clucha_rocr_1h.value ) &
                (dataframe["bb_lowerband2_40"].shift() > 0) &
                (dataframe["bb_delta_cluc"] > dataframe["ha_close"] * self.buy_clucha_bbdelta_close.value) &
                (dataframe["ha_closedelta"] > dataframe["ha_close"] * self.buy_clucha_closedelta_close.value) &
                (dataframe["ha_tail"] < dataframe["bb_delta_cluc"] * self.buy_clucha_bbdelta_tail.value) &
                (dataframe["ha_close"] < dataframe["bb_lowerband2_40"].shift()) &
                (dataframe["ha_close"] < dataframe["ha_close"].shift()) &
                (dataframe["rsi_84"] < 60) &
                (dataframe["rsi_112"] < 60)
            ),
            ["enter_long", "enter_tag"]] = (1, "clucha")
        
        dataframe.loc[
            (
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_c2_1.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_c2_2.value)) &
                (dataframe['cti_1h'] > self.buy_c2_3.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con2")
        dataframe.loc[
            (
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_con3_1.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_con3_2.value)) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_con3_3.value) &
                (dataframe['cti'] < self.buy_con3_4.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con3")
        dataframe.loc[
            (
                (dataframe['rsi_1h'] < self.buy_rsi_1h_42.value) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_41.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_41.value)) &
                (dataframe['volume_mean_slow_30'] > dataframe['volume_mean_slow_30'].shift(48) * self.buy_volume_pump_41.value) &
                (dataframe['volume_mean_slow_30'] * self.buy_volume_pump_41.value < dataframe['volume_mean_slow_30'].shift(48))
            ),
            ["enter_long", "enter_tag"]] = (1, "con4")
        dataframe.loc[
            (
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * self.buy_c7_1.value)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_c7_2.value)) &
                (dataframe['r_14'] < self.buy_c7_3.value) &
                (dataframe['r_64'] < self.buy_c7_4.value) &
                (dataframe['rsi_1h'] < self.buy_c7_5.value) 
            ),
            ["enter_long", "enter_tag"]] = (1, "con7")
        dataframe.loc[
            (
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (dataframe['sma_200'] > dataframe['sma_200'].shift(50)) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_24_1h']) &

                (((dataframe['close'] - dataframe['open'].rolling(36).min()) / dataframe['open'].rolling(36).min()) > self.buy_min_inc_1.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_1.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_1.value) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con8")
        dataframe.loc[
            (
                (((dataframe['close'] - dataframe['open'].rolling(12).min()) / dataframe['open'].rolling(12).min()) > 0.032) &
                (dataframe['rsi'] < self.buy_c9_1.value) &
                (dataframe['r_14'] < self.buy_c9_2.value) &
                (dataframe['r_32'] < self.buy_c9_3.value) &
                (dataframe['mfi'] < self.buy_c9_4.value) &
                (dataframe['rsi_1h'] > self.buy_c9_5.value) &
                (dataframe['rsi_1h'] < self.buy_c9_6.value) &
                (dataframe['r_480_1h'] > self.buy_c9_7.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con9")
        dataframe.loc[
            (
                (dataframe['close'].shift(4) < (dataframe['close'].shift(3))) &
                (dataframe['close'].shift(3) < (dataframe['close'].shift(2))) &
                (dataframe['close'].shift(2) < (dataframe['close'].shift())) &
                (dataframe['close'].shift(1) < (dataframe['close'])) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['close'] > (dataframe['open'])) &
                (dataframe['cci'].shift() < dataframe['cci']) &
                (dataframe['ssl-dir_1h'] == 'up') &
                (dataframe['cci'] < self.buy_c10_1.value) &
                (dataframe['cti'] < self.buy_c10_2.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con10")
        # #BCMBİGZ
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200_1h"]) & 
                (dataframe["ema_50"] > dataframe["ema_200"]) &
                (dataframe["ema_50_1h"] > dataframe["ema_200_1h"]) &
                (((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]) < self.buy_dip_threshold_1.value) &
                (((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"])< self.buy_dip_threshold_2.value) &
                dataframe["lower"].shift().gt(0) &
                dataframe["bbdelta"].gt(dataframe["close"] * self.buy_bb40_bbdelta_close.value) &
                dataframe["closedelta"].gt(dataframe["close"] * self.buy_bb40_closedelta_close.value) &
                dataframe["tail"].lt(dataframe["bbdelta"] * self.buy_bb40_tail_bbdelta.value) &
                dataframe["close"].lt(dataframe["lower"].shift()) &
                dataframe["close"].le(dataframe["close"].shift())
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 0")
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_100_1h"])
                & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"])
                & (((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]) < self.buy_dip_threshold_1.value)
                & (((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"]) < self.buy_dip_threshold_2.value)
                & (((dataframe["open"].rolling(144).max() - dataframe["close"]) / dataframe["close"]) < self.buy_dip_threshold_3.value)
                & (dataframe["volume"].rolling(4).mean() * self.buy_volume_1.value > dataframe["volume"])
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.buy_ema_open_mult_1.value))
                & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 4")
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema_50"])
                & (dataframe["close"] < 0.975 * dataframe["bb_lowerband"])
                & ((dataframe["volume"]< (dataframe["volume_mean_slow"].shift(1) * 20)) | (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4))
                & (dataframe["rsi_1h"] < 15)  # Don't buy if someone drop the market.
                & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 7")
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.02))
                & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
                & ((dataframe["volume"] < (dataframe["volume"].shift() * 4)) | (dataframe["volume_mean_slow"] > dataframe["volume_mean_slow"].shift(30) * 0.4))
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 8")
        dataframe.loc[
            (
                (dataframe["ema_26"] > dataframe["ema_12"])
                & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.03))
                & ((dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100))
                & (dataframe["volume"] < (dataframe["volume"].shift() * 4))
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 9")
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['rsi'] < 30) &
                (dataframe['close'] * 1.024 < dataframe['open'].shift(3)) &
                (dataframe['rsi_1h'] < 71) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 10")
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] < dataframe['bb_lowerband'] * 0.985) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 12")
        dataframe.loc[
            (
                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_1.value) &
                (dataframe['close'] < dataframe['bb_lowerband']) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 14")
        dataframe.loc[
            (
                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_5.value) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.bzv7_buy_macd_2.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'])) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 16")
        dataframe.loc[
            (
                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_2.value) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.bzv7_buy_macd_1.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 17")
        dataframe.loc[
            (
                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_3.value) &
                (dataframe['rsi'] < self.bzv7_buy_rsi_1.value) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 18")
        dataframe.loc[
            (
                (dataframe['rsi_1h'] < self.bzv7_buy_rsi_1h_4.value) &
                (dataframe['rsi'] < self.bzv7_buy_rsi_2.value) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 19")
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['close'] < dataframe['bb_lowerband'] * 0.993) &
                (dataframe['low'] < dataframe['bb_lowerband'] * 0.985) &
                (dataframe['close'].shift() > dataframe['bb_lowerband']) &
                (dataframe['rsi_1h'] < 72.8) &
                (dataframe['open'] > dataframe['close']) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.bzv7_buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.bzv7_buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.bzv7_buy_volume_drop_1.value)) &
                ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2))
            ),
            ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 22")

        dont_buy_conditions = []
        
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, "enter_long"] = 0
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    # ## Trailing params
    # use_custom_stoploss = True
    # # hard stoploss profit
    # pHSL = DecimalParameter(-0.10, -0.040, default=-0.08, decimals=3, space='sell', load=True, optimize=True)
    # # profit threshold 1, trigger point, SL_1 is used
    # pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True, optimize=True)
    # pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True, optimize=True)
    # # profit threshold 2, SL_2 is used
    # pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True, optimize=True)
    # pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True, optimize=True)
    # # Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
    #     # hard stoploss profit
    #     HSL = self.pHSL.value
    #     PF_1 = self.pPF_1.value
    #     SL_1 = self.pSL_1.value
    #     PF_2 = self.pPF_2.value
    #     SL_2 = self.pSL_2.value
    #     # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
    #     # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
    #     # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.
    #     if (current_profit > PF_2):
    #         sl_profit = SL_2 + (current_profit - PF_2)
    #     elif (current_profit > PF_1):
    #         sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
    #     else:
    #         sl_profit = HSL
    #     # Only for hyperopt invalid return
    #     if (sl_profit >= current_profit):
    #         return -0.99
    #     return stoploss_from_open(sl_profit, current_profit)