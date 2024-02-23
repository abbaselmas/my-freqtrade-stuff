from freqtrade.strategy import (IStrategy, informative)
from typing import Dict, List
from pandas import DataFrame, Series
import pandas_ta as pta
import numpy as np
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real
from datetime import datetime
from technical.indicators import RMI

# Buy hyperspace params:
buy_params = {
    "buy_bb40_bbdelta_close": 0.031,
    "buy_bb40_closedelta_close": 0.015,
    "buy_bb40_tail_bbdelta": 0.344,
    "buy_c10_1": -105,
    "buy_c10_2": -1.0,
    "buy_c2_1": 0.023,
    "buy_c2_2": 0.982,
    "buy_c2_3": -0.6,
    "buy_c7_1": 1.03,
    "buy_c7_2": 0.96,
    "buy_c7_3": -82,
    "buy_c7_4": -70,
    "buy_c7_5": 77,
    "buy_c9_1": 35,
    "buy_c9_2": -73,
    "buy_c9_3": -74,
    "buy_c9_4": 35,
    "buy_c9_5": 34,
    "buy_c9_6": 87,
    "buy_c9_7": -91,
    "buy_cci": -122,
    "buy_cci_length": 28,
    "buy_clucha_bbdelta_close": 0.017,
    "buy_clucha_bbdelta_tail": 0.5,
    "buy_clucha_closedelta_close": 0.009,
    "buy_clucha_rocr_1h": 0.82,
    "buy_con3_1": 0.012,
    "buy_con3_2": 0.991,
    "buy_con3_3": 0.975,
    "buy_con3_4": -0.76,
    "buy_dip_threshold_1": 0.28,
    "buy_dip_threshold_2": 0.22,
    "buy_dip_threshold_3": 0.51,
    "buy_dip_threshold_5": 0.022,
    "buy_dip_threshold_6": 0.05,
    "buy_dip_threshold_7": 0.058,
    "buy_dip_threshold_8": 0.19,
    "buy_ema_open_mult_1": 0.012,
    "buy_macd_41": 0.02,
    "buy_mfi_1": 30,
    "buy_min_inc_1": 0.043,
    "buy_pump_pull_threshold_1": 2.42,
    "buy_pump_threshold_1": 0.941,
    "buy_rmi_length": 12,
    "buy_rsi_1": 36,
    "buy_rsi_1h_42": 33,
    "buy_rsi_1h_max_1": 85,
    "buy_rsi_1h_min_1": 25,
    "buy_volume_1": 4,
    "buy_volume_drop_41": 3.2,
    "buy_volume_pump_41": 0.3,
    "buy_vwap_closedelta": 14.8,
    "buy_vwap_cti": -0.01,
    "buy_vwap_width": 3.9,
    "bzv7_buy_macd_1": 0.02,
    "bzv7_buy_macd_2": 0.012,
    "bzv7_buy_rsi_1": 12,
    "bzv7_buy_rsi_1h_1": 30,
    "bzv7_buy_rsi_1h_2": 42,
    "bzv7_buy_rsi_1h_3": 41,
    "bzv7_buy_rsi_1h_4": 16,
    "bzv7_buy_rsi_1h_5": 41,
    "bzv7_buy_rsi_2": 4,
    "bzv7_buy_volume_drop_1": 9.2,
    "bzv7_buy_volume_pump_1": 0.5,
    "ewo_candles_buy": 23,
    "ewo_candles_sell": 4,
    "ewo_high_offset": 1.18302,
    "ewo_low": -9.842,
    "ewo_low_offset": 1.10711,
    "ewo_low_rsi_4": 49,
    "fast_ewo": 6,
    "slow_ewo": 188
}

def EWO(dataframe, ema_length=5, ema2_length=35):
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe["low"] * 100
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
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )
    return WR * -100
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

class abbas8(IStrategy):
    def version(self) -> str:
        return "v9.10"
    INTERFACE_VERSION = 3
    class HyperOpt:
        def stoploss_space():
            return [SKDecimal(-0.12, -0.03, decimals=3, name="stoploss")]
        def trailing_space() -> List[Dimension]:
            return[
                Categorical([True], name="trailing_stop"),
                SKDecimal(0.0003, 0.0020, decimals=4, name="trailing_stop_positive"),
                SKDecimal(0.010,  0.030, decimals=3, name="trailing_stop_positive_offset_p1"),
                Categorical([True], name="trailing_only_offset_is_reached"),
            ]
        def roi_space() -> List[Dimension]:
            return [
                Integer(  6,  34, name="roi_t1"),
                Integer( 20,  64, name="roi_t2"),
                Integer( 50, 104, name="roi_t3"),
                Integer( 80, 194, name="roi_t4"),
                Integer(180, 264, name="roi_t5"),
                Integer(250, 400, name="roi_t6"),
            ]
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            roi_table = {}
            roi_table[params["roi_t1"]] = 0.015
            roi_table[params["roi_t2"]] = 0.010
            roi_table[params["roi_t3"]] = 0.005
            roi_table[params["roi_t4"]] = 0
            roi_table[params["roi_t5"]] = -0.02
            roi_table[params["roi_t6"]] = -0.04
            return roi_table
    timeframe = "5m"
    info_timeframes = ["15m", "30m", "1h"]
    minimal_roi = {
        "23": 0.015,
        "63": 0.01,
        "96": 0.005,
        "176": 0,
        "254": -0.02,
        "387": -0.04
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
    trailing_stop_positive = 0.0003
    trailing_stop_positive_offset = 0.020
    trailing_only_offset_is_reached = True
    use_exit_signal = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True
    startup_candle_count = 449

    # ewo_1 and ewo_low
    ewo1_low_optimize = False
    ewo_candles_buy = IntParameter(2, 30, default=buy_params['ewo_candles_buy'], space='buy', optimize=ewo1_low_optimize)
    ewo_candles_sell = IntParameter(2, 35, default=buy_params['ewo_candles_sell'], space='buy', optimize=ewo1_low_optimize)
    ewo_low_offset = DecimalParameter(0.7, 1.2, default=buy_params['ewo_low_offset'], decimals=5, space='buy', optimize=ewo1_low_optimize)
    ewo_high_offset = DecimalParameter(0.75, 1.5, default=buy_params['ewo_high_offset'], decimals=5, space='buy', optimize=ewo1_low_optimize)
    ewo_low_rsi_4 = IntParameter(1, 50, default=buy_params['ewo_low_rsi_4'], space='buy', optimize=ewo1_low_optimize)
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=ewo1_low_optimize)
    fast_ewo = IntParameter(5,30, default=buy_params["fast_ewo"], space="buy", optimize=ewo1_low_optimize)
    slow_ewo = IntParameter(120,250, default=buy_params["slow_ewo"], space="buy", optimize=ewo1_low_optimize)

    is_optimize_clucha = False
    buy_clucha_bbdelta_close    = DecimalParameter(0.010, 0.060,  default=buy_params["buy_clucha_bbdelta_close"],    space="buy", decimals=3, optimize = is_optimize_clucha) # 0.060-0.010 = 0.050 | 50
    buy_clucha_bbdelta_tail     = DecimalParameter(0.40,   1.00,  default=buy_params["buy_clucha_bbdelta_tail"],     space="buy", decimals=2, optimize = is_optimize_clucha) # 1.00-0.40 = 0.60 | 60
    buy_clucha_closedelta_close = DecimalParameter(0.001,  0.030, default=buy_params["buy_clucha_closedelta_close"], space="buy", decimals=3, optimize = is_optimize_clucha) # 0.030-0.001 = 0.029 | 29
    buy_clucha_rocr_1h          = DecimalParameter(0.050,   1.00, default=buy_params["buy_clucha_rocr_1h"],          space="buy", decimals=2, optimize = is_optimize_clucha) # 1.00-0.050 = 0.950 | 95

    is_optimize_vwap = False
    buy_vwap_width      = DecimalParameter(0.5, 10.0,    default=buy_params["buy_vwap_width"],      space="buy", decimals=1, optimize = is_optimize_vwap) # 10.0-0.5 = 9.5 | 95
    buy_vwap_closedelta = DecimalParameter(10.0, 30.0,   default=buy_params["buy_vwap_closedelta"], space="buy", decimals=1, optimize = is_optimize_vwap) # 30.0-10.0 = 20.0 | 200
    buy_vwap_cti        = DecimalParameter(-0.90, -0.00, default=buy_params["buy_vwap_cti"],        space="buy", decimals=2, optimize = is_optimize_vwap) # -0.00-(-0.90) = 0.90 | 90

    # BeastBotXBLR
    ###########################################################################
    # Buy
    optc1 = True
    buy_rmi_length = IntParameter(8, 20,     default=buy_params["buy_rmi_length"],    space='buy', optimize= optc1) # 20-8 = 12
    buy_cci_length = IntParameter(25, 45,    default=buy_params["buy_cci_length"],    space='buy', optimize= optc1) # 45-25 = 20
    buy_cci        = IntParameter(-135, -90, default=buy_params["buy_cci"],           space='buy', optimize= optc1) # -90-(-135) = 45

    optc2 = True
    buy_c2_1 = DecimalParameter(0.010, 0.025, default=buy_params["buy_c2_1"], space='buy', decimals=3, optimize=optc2) # 0.025-0.010 = 0.015 | 15
    buy_c2_2 = DecimalParameter(0.980, 0.995, default=buy_params["buy_c2_2"], space='buy', decimals=3, optimize=optc2) # 0.995-0.980 = 0.015 | 15
    buy_c2_3 = DecimalParameter(-0.8, -0.3,   default=buy_params["buy_c2_3"], space='buy', decimals=1, optimize=optc2) # -0.3-(-0.8) = 0.5 | 5
    
    optc3 = True
    buy_con3_1 = DecimalParameter(0.010, 0.025, default=buy_params["buy_con3_1"], space='buy', decimals=3, optimize=optc3) # 0.025-0.010 = 0.015 | 15
    buy_con3_2 = DecimalParameter(0.980, 0.995, default=buy_params["buy_con3_2"], space='buy', decimals=3, optimize=optc3) # 0.995-0.980 = 0.015 | 15
    buy_con3_3 = DecimalParameter(0.955, 0.975, default=buy_params["buy_con3_3"], space='buy', decimals=3, optimize=optc3) # 0.975-0.955 = 0.020 | 20
    buy_con3_4 = DecimalParameter(-0.95, -0.70, default=buy_params["buy_con3_4"], space='buy', decimals=2, optimize=optc3) # -0.70-(-0.95) = 0.25 | 25

    optc4 = True
    buy_rsi_1h_42      = IntParameter(10, 50,         default=buy_params["buy_rsi_1h_42"], space='buy', optimize=optc4) # 50-10 = 40
    buy_macd_41        = DecimalParameter(0.01, 0.09, default=buy_params["buy_macd_41"], space='buy', decimals=2, optimize=optc4) # 0.09-0.01 = 0.08 | 8
    buy_volume_pump_41 = DecimalParameter(0.1, 0.9,   default=buy_params["buy_volume_pump_41"], space='buy', decimals=1, optimize=optc4) # 0.9-0.1 = 0.8 | 8
    buy_volume_drop_41 = DecimalParameter(1, 10,      default=buy_params["buy_volume_drop_41"], space='buy', decimals=1, optimize=optc4) # 10-1 = 9 | 90

    optc8 = True
    buy_min_inc_1 = DecimalParameter(0.010, 0.050, default=buy_params["buy_min_inc_1"], space='buy', decimals=3, optimize=optc8) # 0.050-0.010 = 0.040 | 40
    buy_rsi_1h_min_1 = IntParameter(25, 40, default=buy_params["buy_rsi_1h_min_1"], space='buy', optimize=optc8) # 40-25 = 15
    buy_rsi_1h_max_1 = IntParameter(70, 90, default=buy_params["buy_rsi_1h_max_1"], space='buy', optimize=optc8) # 90-70 = 20
    buy_rsi_1        = IntParameter(20, 40, default=buy_params["buy_rsi_1"], space='buy', optimize=optc8) # 40-20 = 20
    buy_mfi_1        = IntParameter(20, 40, default=buy_params["buy_mfi_1"], space='buy', optimize=optc8) # 40-20 = 20

    optc9 = True
    buy_c9_1 = IntParameter(25, 44,   default=buy_params["buy_c9_1"], space='buy', optimize=optc9) # 44-25 = 19
    buy_c9_2 = IntParameter(-80, -67, default=buy_params["buy_c9_2"], space='buy', optimize=optc9) # -67-(-80) = 13
    buy_c9_3 = IntParameter(-80, -67, default=buy_params["buy_c9_3"], space='buy', optimize=optc9) # -67-(-80) = 13
    buy_c9_4 = IntParameter(35, 54,   default=buy_params["buy_c9_4"], space='buy', optimize=optc9) # 54-35 = 19
    buy_c9_5 = IntParameter(20, 44,   default=buy_params["buy_c9_5"], space='buy', optimize=optc9) # 44-20 = 24
    buy_c9_6 = IntParameter(65, 94,   default=buy_params["buy_c9_6"], space='buy', optimize=optc9) # 94-65 = 29
    buy_c9_7 = IntParameter(-110, -80, default=buy_params["buy_c9_7"], space='buy', optimize=optc9) # -80-(-110) = 30

    optc10 = True
    buy_c10_1 = IntParameter(-110, -80,        default=buy_params["buy_c10_1"], space='buy', optimize=optc10) # -80-(-110) = 30
    buy_c10_2 = DecimalParameter(-1.00, -0.50, default=buy_params["buy_c10_2"], space='buy', decimals=2, optimize=optc10) # -0.50-(-1.00) = 0.50 | 50

    dip_optimize = True
    buy_dip_threshold_5 = DecimalParameter(0.020, 0.070, default=buy_params["buy_dip_threshold_5"], space='buy', decimals=3, optimize=dip_optimize) ## 0.070-0.020 = 0.050 | 50
    buy_dip_threshold_6 = DecimalParameter(0.050, 0.100, default=buy_params["buy_dip_threshold_6"], space='buy', decimals=3, optimize=dip_optimize) # 0.100-0.050 = 0.050 | 50
    buy_dip_threshold_7 = DecimalParameter(0.050, 0.100, default=buy_params["buy_dip_threshold_7"], space='buy', decimals=3, optimize=dip_optimize) # 0.100-0.050 = 0.050 | 50
    buy_dip_threshold_8 = DecimalParameter(0.150, 0.250, default=buy_params["buy_dip_threshold_8"], space='buy', decimals=3, optimize=dip_optimize) # 0.250-0.150 = 0.100 | 100
    # 24 hours
    buy_pump_optimize = True
    buy_pump_pull_threshold_1 = DecimalParameter(1.50, 3.00, default=buy_params["buy_pump_pull_threshold_1"], space='buy', decimals=2, optimize=buy_pump_optimize) # 3.00-1.50 = 1.50 | 150
    buy_pump_threshold_1    = DecimalParameter(0.600, 1.000, default=buy_params["buy_pump_threshold_1"], space='buy', decimals=3, optimize=buy_pump_optimize) # 1.000-0.600 = 0.400 | 400

    #  Strategy: BigZ07
    # Buy HyperParam
    pump_dump_optimize = True
    bzv7_buy_volume_pump_1 = DecimalParameter(0.1, 0.9, default=buy_params["bzv7_buy_volume_pump_1"], space="buy", decimals=1, optimize=pump_dump_optimize)
    bzv7_buy_volume_drop_1 = DecimalParameter(1, 10,    default=buy_params["bzv7_buy_volume_drop_1"], space="buy", decimals=1, optimize=pump_dump_optimize)

    bzv7_rsi_optimize = True
    bzv7_buy_rsi_1h_1 = IntParameter(8, 30, default=buy_params["bzv7_buy_rsi_1h_1"], space="buy", optimize=bzv7_rsi_optimize)
    bzv7_buy_rsi_1h_2 = IntParameter(20, 45, default=buy_params["bzv7_buy_rsi_1h_2"], space="buy", optimize=bzv7_rsi_optimize)
    bzv7_buy_rsi_1h_4 = IntParameter(10, 30, default=buy_params["bzv7_buy_rsi_1h_4"], space="buy", optimize=bzv7_rsi_optimize)
    bzv7_buy_rsi_2    = IntParameter(4, 20,  default=buy_params["bzv7_buy_rsi_2"], space="buy", optimize=bzv7_rsi_optimize) # 40-7 = 33

    bzv7_macd_optimize = True
    bzv7_buy_macd_1 = DecimalParameter(0.01, 0.09, default=buy_params["bzv7_buy_macd_1"], space="buy", decimals=2, optimize=bzv7_macd_optimize) # 0.09-0.01 = 0.08 | 8

    buy_dip_threshold_optimize = True
    buy_dip_threshold_1 = DecimalParameter(0.20, 0.40, default=buy_params["buy_dip_threshold_1"], space="buy", decimals=2, optimize=buy_dip_threshold_optimize)
    buy_dip_threshold_2 = DecimalParameter(0.20, 0.50, default=buy_params["buy_dip_threshold_2"], space="buy", decimals=2, optimize=buy_dip_threshold_optimize)
    buy_dip_threshold_3 = DecimalParameter(0.30, 0.60, default=buy_params["buy_dip_threshold_3"], space="buy", decimals=2, optimize=buy_dip_threshold_optimize)

    bcmbigz0 = True
    buy_bb40_bbdelta_close     = DecimalParameter(0.025, 0.045, default=buy_params["buy_bb40_bbdelta_close"],       space="buy", decimals=3, optimize=bcmbigz0)
    buy_bb40_closedelta_close  = DecimalParameter(0.010, 0.030, default=buy_params["buy_bb40_closedelta_close"],    space="buy", decimals=3, optimize=bcmbigz0)
    buy_bb40_tail_bbdelta      = DecimalParameter(0.250, 0.350, default=buy_params["buy_bb40_tail_bbdelta"],        space="buy", decimals=3, optimize=bcmbigz0)
    
    bcmbigz4 = True
    buy_volume_1        =  IntParameter(1, 10,  default=buy_params["buy_volume_1"], space="buy",  optimize=bcmbigz4)
    buy_ema_open_mult_1 =  DecimalParameter(0.010, 0.050, default=buy_params["buy_ema_open_mult_1"], space="buy", decimals=3, optimize=bcmbigz4)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for info_timeframe in self.info_timeframes:
            informative_pairs.extend([(pair, info_timeframe) for pair in pairs])
        return informative_pairs
    
    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")

        # EMA
        informative_1h['ema_12'] = ta.EMA(informative_1h, timeperiod=12)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)

        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)
        informative_1h['r_480'] = williams_r(dataframe, period=480)
        informative_1h['safe_pump_24'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) /
            informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_1.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) /
            self.buy_pump_pull_threshold_1.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))

        informative_1h['cti'] = pta.cti(informative_1h["close"], length=20) 

        ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
        informative_1h["ssl-dir"] = np.where(ssl_up_1h > ssl_down_1h, "up", "down")

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
        dataframe[f'ma_buy_{self.ewo_candles_buy.value}'] = ta.EMA(dataframe, timeperiod=int(self.ewo_candles_buy.value))
        dataframe[f'ma_sell_{self.ewo_candles_sell.value}'] = ta.EMA(dataframe, timeperiod=int(self.ewo_candles_sell.value))
        dataframe["ewo"] = EWO(dataframe, int(self.fast_ewo.value), int(self.slow_ewo.value))
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe["volume_mean_slow_30"] = dataframe["volume"].rolling(window=30).mean()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        bb_21268 = qtpylib.bollinger_bands(dataframe["close"], window=21, stds=2.68)
        dataframe["bb_lowerband"] = bb_21268["lower"]
        dataframe["bb_upperband"] = bb_21268["upper"]
        dataframe["bbdelta"] = (bb_21268["mid"] - bb_21268["lower"]).abs()
        dataframe["macd"], dataframe["signal"], dataframe["hist"] = ta.MACD(dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)
        dataframe['cci'] = ta.CCI(dataframe, 26)
        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=48).mean()
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_32'] = williams_r(dataframe, period=32)
        dataframe['r_64'] = williams_r(dataframe, period=64)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['mfi'] = ta.MFI(dataframe)
        
        # Dip protection
        # dataframe['safe_dips_normal'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
        #                           (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_2.value) &
        #                           (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_3.value) &
        #                           (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_4.value))

        dataframe['safe_dips_strict'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_5.value) &
                                  (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_6.value) &
                                  (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_7.value) &
                                  (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_8.value))

        # dataframe['safe_dips_loose'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_9.value) &
        #                           (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_10.value) &
        #                           (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_11.value) &
        #                           (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_12.value))

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["ha_high"] = heikinashi["high"]
        dataframe["ha_low"] = heikinashi["low"]
        dataframe["rsi_84"] = ta.RSI(dataframe, timeperiod=84)
        dataframe["rsi_112"] = ta.RSI(dataframe, timeperiod=112)
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe["vwap_upperband"] = vwap_high
        dataframe["vwap_middleband"] = vwap
        dataframe["vwap_lowerband"] = vwap_low
        dataframe["vwap_width"] = ( (dataframe["vwap_upperband"] - dataframe["vwap_lowerband"]) / dataframe["vwap_middleband"] ) * 100
        bollinger2_40 = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=40, stds=2)
        dataframe["bb_lowerband2_40"] = bollinger2_40["lower"]
        dataframe["bb_middleband2_40"] = bollinger2_40["mid"]
        dataframe["bb_delta_cluc"] = (dataframe["bb_middleband2_40"] - dataframe["bb_lowerband2_40"]).abs()
        dataframe["ha_closedelta"] = (dataframe["ha_close"] - dataframe["ha_close"].shift()).abs()
        dataframe["ha_tail"] = (dataframe["ha_close"] - dataframe["ha_low"]).abs()
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)
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
        # informative_30m = self.informative_30m_indicators(dataframe, metadata)
        # dataframe       = merge_informative_pair(dataframe, informative_30m, self.timeframe, "30m", ffill=True)
        # informative_15m = self.informative_15m_indicators(dataframe, metadata)
        # dataframe       = merge_informative_pair(dataframe, informative_15m, self.timeframe, "15m", ffill=True)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['rsi_4'] <  self.ewo_low_rsi_4.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.ewo_candles_buy.value}'] * self.ewo_low_offset.value)) &
                (dataframe['ewo'] < self.ewo_low.value) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.ewo_candles_sell.value}'] * self.ewo_high_offset.value))
            ),
            ["enter_long", "enter_tag"]] = (1, "ewolow__")
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
                dataframe["bb_lowerband"].shift().gt(0) &
                dataframe["bbdelta"].gt(dataframe["close"] * self.buy_bb40_bbdelta_close.value) &
                dataframe["closedelta"].gt(dataframe["close"] * self.buy_bb40_closedelta_close.value) &
                dataframe["tail"].lt(dataframe["bbdelta"] * self.buy_bb40_tail_bbdelta.value) &
                dataframe["close"].lt(dataframe["bb_lowerband"].shift()) &
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

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe