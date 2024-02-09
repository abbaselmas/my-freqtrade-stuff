from freqtrade.strategy import (IStrategy, informative)
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
import logging
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real
from market_profile import MarketProfile
logger = logging.getLogger(__name__)
# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 15,
    "ewo_high": 8.32,
    "ewo_high_2": -4.44,
    "ewo_low": -6.42,
    "fast_ewo": 9,
    "low_offset": 1.18,
    "low_offset_2": 0.93,
    "rsi_buy": 67,
    "rsi_ewo2": 18,
    "rsi_fast_ewo1": 56,
    "slow_ewo": 198,
    "buy_condition_0_enable": True,
    "buy_condition_10_enable": True,
    "buy_condition_11_enable": True,
    "buy_condition_12_enable": True,
    "buy_condition_13_enable": True,
    "buy_condition_14_enable": True,
    "buy_condition_15_enable": True,
    "buy_condition_16_enable": True,
    "buy_condition_17_enable": True,
    "buy_condition_18_enable": True,
    "buy_condition_19_enable": True,
    "buy_condition_1_enable": True,
    "buy_condition_2_enable": True,
    "buy_condition_3_enable": True,
    "buy_condition_4_enable": True,
    "buy_condition_5_enable": True,
    "buy_condition_6_enable": True,
    "buy_condition_7_enable": True,
    "buy_condition_8_enable": True,
    "buy_condition_9_enable": True,
    "buy_bb20_close_bblowerband_safe_1": 0.726,
    "buy_bb20_close_bblowerband_safe_2": 0.962,
    "buy_macd_1": 0.09,
    "buy_macd_2": 0.01,
    "buy_rsi_1": 35.6,
    "buy_rsi_1h_1": 14.1,
    "buy_rsi_1h_2": 38.4,
    "buy_rsi_1h_3": 26.6,
    "buy_rsi_1h_4": 11.1,
    "buy_rsi_1h_5": 58.5,
    "buy_rsi_2": 24.6,
    "buy_rsi_3": 18.4,
    "buy_volume_drop_1": 3.2,
    "buy_volume_drop_2": 1.9,
    "buy_volume_drop_3": 10.0,
    "buy_volume_pump_1": 0.8
}
# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 15,
    "high_offset": 1.11,
    "volume_warn": 5.0,
    "percent_change_length": 60,
    "percent_change_low": -0.05,
    "percent_change_high": 0.05,
    "volume_mean_long": 48,
    "volume_mean_short": 4
}
class abbas8(IStrategy):
    def version(self) -> str:
        return "v9.8.3"
    INTERFACE_VERSION = 3
    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.08, -0.05, decimals=3, name="stoploss")]
        # Define custom trailing space
        def trailing_space() -> List[Dimension]:
            return[
                Categorical([True], name="trailing_stop"),
                SKDecimal(0.0002, 0.0006, decimals=4, name="trailing_stop_positive"),
                SKDecimal(0.010,  0.020, decimals=3, name="trailing_stop_positive_offset_p1"),
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
    stoploss = -0.067
    trailing_stop = True
    trailing_stop_positive = 0.0003
    trailing_stop_positive_offset = 0.0146
    trailing_only_offset_is_reached = True
    use_exit_signal = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True
    startup_candle_count = 449

    smaoffset_optimize = False
    base_nb_candles_buy = IntParameter(10, 50, default=buy_params["base_nb_candles_buy"], space="buy", optimize=smaoffset_optimize)
    base_nb_candles_sell = IntParameter(10, 50, default=sell_params["base_nb_candles_sell"], space="sell", optimize=smaoffset_optimize)
    low_offset = DecimalParameter(0.8, 1.2, default=buy_params["low_offset"], space="buy", decimals=2, optimize=smaoffset_optimize)
    low_offset_2 = DecimalParameter(0.8, 1.2, default=buy_params["low_offset_2"], space="buy", decimals=2, optimize=smaoffset_optimize)
    high_offset = DecimalParameter(0.8, 1.2, default=sell_params["high_offset"], space="sell", decimals=2, optimize=smaoffset_optimize)

    ewo_optimize = False
    fast_ewo = IntParameter(5,40, default=buy_params["fast_ewo"], space="buy", optimize=ewo_optimize)
    slow_ewo = IntParameter(80,250, default=buy_params["slow_ewo"], space="buy", optimize=ewo_optimize)
    rsi_fast_ewo1 = IntParameter(20, 60, default=buy_params["rsi_fast_ewo1"], space="buy", optimize=ewo_optimize)
    rsi_ewo2 = IntParameter(10, 40, default=buy_params["rsi_ewo2"], space="buy", optimize=ewo_optimize)

    protection_optimize = False
    ewo_low = DecimalParameter(-20.0, -4.0, default=buy_params["ewo_low"], space="buy", decimals=2, optimize=protection_optimize)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params["ewo_high"], space="buy", decimals=2, optimize=protection_optimize)
    ewo_high_2 = DecimalParameter(-6.0, 12.0, default=buy_params["ewo_high_2"], space="buy", decimals=2, optimize=protection_optimize)
    rsi_buy = IntParameter(50, 85, default=buy_params["rsi_buy"], space="buy", optimize=protection_optimize)

    volume_optimize = True
    volume_mean_long = IntParameter(20, 200, default=sell_params["volume_mean_long"], space="sell", optimize=volume_optimize)
    volume_mean_short = IntParameter(2, 50, default=sell_params["volume_mean_short"], space="sell", optimize=volume_optimize)
    volume_warn = DecimalParameter(0.0, 10.0, default=sell_params["volume_warn"], space="sell", decimals=2, optimize=volume_optimize)

    percent_change_optimize = True
    percent_change_length = IntParameter(5, 288, default=sell_params["percent_change_length"], space="sell", optimize=percent_change_optimize)
    percent_change_low = DecimalParameter(-0.50, 0.00, default=sell_params["percent_change_low"], decimals=2, space="sell", optimize=percent_change_optimize)
    percent_change_high = DecimalParameter(0.00, 0.70, default=sell_params["percent_change_high"], decimals=2, space="sell", optimize=percent_change_optimize)

    order_time_in_force = {
        "entry": "gtc",
        "exit": "ioc"
    }
    slippage_protection = {
        "retries": 3,
        "max_slippage": -0.002
    }
    buy_condition_optimize = False
    buy_condition_0_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_5_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_6_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_7_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_8_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_9_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_10_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_11_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_12_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_13_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_14_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_15_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_16_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_17_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_18_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)
    buy_condition_19_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_condition_optimize)

    buy_bb20_close_bblowerband_safe_1 = DecimalParameter(0.9, 1.1, default=0.99, space='buy', decimals=2, optimize=True)
    buy_bb20_close_bblowerband_safe_2 = DecimalParameter(0.9, 1.1, default=0.982, space='buy', decimals=2, optimize=True)
    
    buy_volume_optimize = True
    buy_volume_pump_1 = DecimalParameter(0.1, 0.9, default=0.4, space='buy', decimals=1, optimize=buy_volume_optimize)
    buy_volume_drop_1 = DecimalParameter(1, 10, default=4, space='buy', decimals=1, optimize=buy_volume_optimize)
    buy_volume_drop_2 = DecimalParameter(1, 10, default=3, space='buy', decimals=1, optimize=buy_volume_optimize)
    buy_volume_drop_3 = DecimalParameter(1, 10, default=2.7, space='buy', decimals=1, optimize=buy_volume_optimize)
    
    buy_rsi_optimize = True
    buy_rsi_1h_1 = DecimalParameter(10.0, 40.0, default=16.5, space='buy', decimals=1, optimize=buy_rsi_optimize)
    buy_rsi_1h_2 = DecimalParameter(10.0, 40.0, default=15.0, space='buy', decimals=1, optimize=buy_rsi_optimize)
    buy_rsi_1h_3 = DecimalParameter(10.0, 40.0, default=20.0, space='buy', decimals=1, optimize=buy_rsi_optimize)
    buy_rsi_1h_4 = DecimalParameter(10.0, 40.0, default=35.0, space='buy', decimals=1, optimize=buy_rsi_optimize)
    buy_rsi_1h_5 = DecimalParameter(10.0, 60.0, default=39.0, space='buy', decimals=1, optimize=buy_rsi_optimize)
    buy_rsi_1 = DecimalParameter(10.0, 40.0, default=28.0, space='buy', decimals=1, optimize=buy_rsi_optimize)
    buy_rsi_2 = DecimalParameter(7.0, 40.0, default=10.0, space='buy', decimals=1, optimize=buy_rsi_optimize)
    buy_rsi_3 = DecimalParameter(7.0, 40.0, default=14.2, space='buy', decimals=1, optimize=buy_rsi_optimize)
    
    buy_macd_1 = DecimalParameter(0.01, 0.09, default=0.02, space='buy', decimals=2, optimize=True)
    buy_macd_2 = DecimalParameter(0.01, 0.09, default=0.03, space='buy', decimals=2, optimize=True)
    
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for info_timeframe in self.info_timeframes:
            informative_pairs.extend([(pair, info_timeframe) for pair in pairs])
        return informative_pairs
    
    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2.68)
        # informative_1h["bb20_2_low"] = bollinger["lower"]
        # informative_1h["bb20_2_mid"] = bollinger["mid"]
        # informative_1h["bb20_2_upp"] = bollinger["upper"]
        # EMA
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)
        
        return informative_1h
    
    # def informative_30m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     assert self.dp, "DataProvider is required for multiple timeframes."
    #     informative_30m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="30m")
    #     bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_30m), window=20, stds=2.8)
    #     informative_30m["bb20_2_low"] = bollinger["lower"]
    #     informative_30m["bb20_2_mid"] = bollinger["mid"]
    #     informative_30m["bb20_2_upp"] = bollinger["upper"]    
    #     return informative_30m
    
    # def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     assert self.dp, "DataProvider is required for multiple timeframes."
    #     informative_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="15m")
    #     bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_15m), window=20, stds=2.68)
    #     informative_15m["bb20_2_low"] = bollinger["lower"]
    #     informative_15m["bb20_2_mid"] = bollinger["mid"]
    #     informative_15m["bb20_2_upp"] = bollinger["upper"]
    #     return informative_15m

    def base_tf_5m_indicators(self, metadata: dict, dataframe: DataFrame) -> DataFrame:
        dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_buy.value))
        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_sell.value))
        dataframe["ewo"] = EWO(dataframe, int(self.fast_ewo.value), int(self.slow_ewo.value))
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        # EMA
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe
    
    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['volume_mean_short'] = dataframe['volume'].rolling(self.volume_mean_short.value).mean()
        dataframe['volume_mean_long'] = dataframe['volume'].rolling(self.volume_mean_long.value).mean()
        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > self.volume_warn.value), -1, 0)
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
        dataframe = self.pump_dump_protection(dataframe, metadata)
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_tag"] = ""
        # dataframe.loc[
        #     (
        #         (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
        #         (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
        #         (dataframe["ewo"] > self.ewo_high.value) &
        #         (dataframe["rsi"] < self.rsi_buy.value) &
        #         (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "ewo1")
        # dataframe.loc[
        #     (
        #         (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
        #         (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset_2.value)) &
        #         (dataframe["ewo"] > self.ewo_high_2.value) &
        #         (dataframe["rsi"] < self.rsi_buy.value) &
        #         (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value)) &
        #         (dataframe["rsi"] < self.rsi_ewo2.value)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "ewo2")
        # dataframe.loc[
        #     (
        #         (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
        #         (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
        #         (dataframe["ewo"] < self.ewo_low.value) &
        #         (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "ewolow")
        
        # CombinedBinHClucAndMADV9
        dataframe.loc[
            (
                self.buy_condition_0_enable.value &
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['rsi'] < 30) &
                (dataframe['close'] * 1.024 < dataframe['open'].shift(3)) &
                (dataframe['rsi_1h'] < 71) &
                (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            ),
            ["enter_long", "enter_tag"]] = (1, "cond 0")
        # dataframe.loc[
        #     (    
        #         self.buy_condition_1_enable.value &
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] > dataframe['ema_200_1h']) &
        #         (dataframe['close'] <  dataframe['bb_lowerband'] * self.buy_bb20_close_bblowerband_safe_1.value) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &
        #         (dataframe['volume'] > 0)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 1")
        # dataframe.loc[
        #     (   
        #         self.buy_condition_2_enable.value &
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] < dataframe['bb_lowerband'] *  self.buy_bb20_close_bblowerband_safe_2.value) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &
        #         (dataframe['volume'] > 0)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 2")
        # dataframe.loc[
        #     (   
        #         self.buy_condition_3_enable.value &
        #         (dataframe['close'] > dataframe['ema_200_1h']) &
        #         (dataframe['close'] < dataframe['bb_lowerband']) &
        #         (dataframe['rsi'] < self.buy_rsi_3.value) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume'] > 0)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 3")
        # dataframe.loc[
        #     (   
        #         self.buy_condition_4_enable.value &
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_1.value) &
        #         (dataframe['close'] < dataframe['bb_lowerband']) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume'] > 0)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 4")
        # dataframe.loc[
        #     (   
        #         self.buy_condition_5_enable.value &
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] > dataframe['ema_200_1h']) &
        #         (dataframe['ema_26'] > dataframe['ema_12']) &
        #         ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
        #         ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
        #         (dataframe['close'] < (dataframe['bb_lowerband'])) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume'] > 0) # Make sure Volume is not 0
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 5")
        # dataframe.loc[
        #     (   
        #         self.buy_condition_6_enable.value &
        #         (dataframe['ema_26'] > dataframe['ema_12']) &
        #         ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_2.value)) &
        #         ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
        #         (dataframe['close'] < (dataframe['bb_lowerband'])) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume'] > 0)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 6")
        # dataframe.loc[
        #     (
        #         self.buy_condition_7_enable.value &
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_2.value) &
        #         (dataframe['ema_26'] > dataframe['ema_12']) &
        #         ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
        #         ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume'] > 0)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 7")
        # dataframe.loc[
        #     (   
        #         self.buy_condition_8_enable.value &
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_3.value) &
        #         (dataframe['rsi'] < self.buy_rsi_1.value) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * self.buy_volume_pump_1.value)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 8")
        # dataframe.loc[
        #     (   
        #         self.buy_condition_9_enable.value &
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_4.value) &
        #         (dataframe['rsi'] < self.buy_rsi_2.value) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * self.buy_volume_pump_1.value)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 9")
        # dataframe.loc[
        #     (
        #         self.buy_condition_10_enable.value &
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] < dataframe['bb_lowerband'] *  self.buy_bb20_close_bblowerband_safe_2.value) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 10")
        # dataframe.loc[
        #     (
        #         self.buy_condition_11_enable.value &
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] > dataframe['ema_200_1h']) &
        #         (dataframe['close'] <  dataframe['bb_lowerband'] * self.buy_bb20_close_bblowerband_safe_1.value) &
        #         (dataframe['rsi_1h'] < 69) &
        #         (dataframe['open'] > dataframe['close']) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 11")
        # dataframe.loc[
        #     (
        #         self.buy_condition_12_enable.value &
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] > dataframe['ema_200_1h']) &
        #         (dataframe['close'] < dataframe['bb_lowerband'] * 0.993) &
        #         (dataframe['low'] < dataframe['bb_lowerband'] * 0.985) &
        #         (dataframe['close'].shift() > dataframe['bb_lowerband']) &
        #         (dataframe['rsi_1h'] < 72.8) &
        #         (dataframe['open'] > dataframe['close']) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 12")
        # dataframe.loc[
        #     (
        #         self.buy_condition_13_enable.value &
        #         (dataframe['close'] > dataframe['ema_200_1h']) &
        #         (dataframe['close'] < dataframe['bb_lowerband']) &
        #         (dataframe['rsi'] < self.buy_rsi_3.value) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_3.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 13")
        # dataframe.loc[
        #     (
        #         self.buy_condition_14_enable.value &
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_1.value) &
        #         (dataframe['close'] < dataframe['bb_lowerband']) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 14")
        # dataframe.loc[
        #     (
        #         self.buy_condition_15_enable.value &
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] > dataframe['ema_200_1h']) &
        #         (dataframe['ema_26'] > dataframe['ema_12']) &
        #         ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
        #         ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
        #         (dataframe['close'] < (dataframe['bb_lowerband'])) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 15")
        # dataframe.loc[
        #     (
        #         self.buy_condition_16_enable.value &
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_5.value) &
        #         (dataframe['ema_26'] > dataframe['ema_12']) &
        #         ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_2.value)) &
        #         ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
        #         (dataframe['close'] < (dataframe['bb_lowerband'])) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 16")
        # dataframe.loc[
        #     (
        #         self.buy_condition_17_enable.value &
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_2.value) &
        #         (dataframe['ema_26'] > dataframe['ema_12']) &
        #         ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
        #         ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 17")
        # dataframe.loc[
        #     (
        #         self.buy_condition_18_enable.value &
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_3.value) &
        #         (dataframe['rsi'] < self.buy_rsi_1.value) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 18")
        # dataframe.loc[
        #     (
        #         self.buy_condition_19_enable.value &
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_4.value) &
        #         (dataframe['rsi'] < self.buy_rsi_2.value) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 19")
        
        # dont_buy_conditions = []
        # dont_buy_conditions.append(
        #     (
        #         (dataframe['pnd_volume_warn'] < 0.0)
        #     )
        # )
        # dont_buy_conditions.append(
        #     (
        #         ((dataframe['open'].rolling(self.percent_change_length.value).max() - dataframe['close']) / dataframe['close'] < self.percent_change_low.value) &
        #         ((dataframe['open'].rolling(self.percent_change_length.value).max() - dataframe['close']) / dataframe['close'] > self.percent_change_high.value)
        #     )
        # )
        
        # if dont_buy_conditions:
        #     for condition in dont_buy_conditions:
        #         dataframe.loc[condition, "enter_long"] = 0
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
def EWO(dataframe, ema_length=5, ema2_length=35):
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe["low"] * 100