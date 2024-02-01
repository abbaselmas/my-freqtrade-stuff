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

# Protection hyperspace params:
protection_params = {
    "cooldown_stop_duration_candles": 0,
    "maxdrawdown_lookback_period_candles": 3,
    "maxdrawdown_max_allowed_drawdown": 0.06,
    "maxdrawdown_stop_duration_candles": 162,
    "maxdrawdown_trade_limit": 10,
    "stoplossguard_lookback_period_candles": 24,
    "stoplossguard_stop_duration_candles": 29,
    "stoplossguard_trade_limit": 3
}
# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 12,
    "ewo_high": 7.54,
    "ewo_high_2": 9.88,
    "ewo_low": -5.53,
    "fast_ewo": 9,
    "low_offset": 1.18,
    "low_offset_2": 0.97,
    "rsi_buy": 55,
    "rsi_ewo2": 38,
    "rsi_fast_ewo1": 52,
    "slow_ewo": 168
}
# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 21,
    "high_offset": 1.01,
    "volume_warn": 5.0,
    "btc_rsi_8_1h": 35,
    "percent_change_length": 60,
    "hl_pct_change_06_1h": 0.80,
    "hl_pct_change_12_1h": 0.90,
    "hl_pct_change_24_1h": 0.95,
    "hl_pct_change_48_1h": 1.00
}

class abbas8(IStrategy):
    def version(self) -> str:
        return "v9.6"
    INTERFACE_VERSION = 3

    cooldown_stop_duration_candles = IntParameter(0, 5, default = protection_params["cooldown_stop_duration_candles"], space="protection", optimize=True)

    maxdrawdown_optimize = True
    maxdrawdown_lookback_period_candles = IntParameter(1, 200, default=protection_params["maxdrawdown_lookback_period_candles"], space="protection", optimize=maxdrawdown_optimize)
    maxdrawdown_trade_limit = IntParameter(1, 10, default=protection_params["maxdrawdown_trade_limit"], space="protection", optimize=maxdrawdown_optimize)
    maxdrawdown_stop_duration_candles = IntParameter(20, 200, default=protection_params["maxdrawdown_stop_duration_candles"], space="protection", optimize=maxdrawdown_optimize)
    maxdrawdown_max_allowed_drawdown = DecimalParameter(0.01, 0.10, default=protection_params["maxdrawdown_max_allowed_drawdown"], space="protection", decimals=2, optimize=maxdrawdown_optimize)

    stoplossguard_optimize = True
    stoplossguard_lookback_period_candles = IntParameter(5, 200, default=protection_params["stoplossguard_lookback_period_candles"], space="protection", optimize=stoplossguard_optimize)
    stoplossguard_trade_limit = IntParameter(1, 5, default=protection_params["stoplossguard_trade_limit"], space="protection", optimize=stoplossguard_optimize)
    stoplossguard_stop_duration_candles = IntParameter(1, 50, default=protection_params["stoplossguard_stop_duration_candles"], space="protection", optimize=stoplossguard_optimize)

    @property
    def protections(self):
        prot = []
        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_stop_duration_candles.value
        })
        prot.append({
            "method": "MaxDrawdown",
            "lookback_period_candles": self.maxdrawdown_lookback_period_candles.value,
            "trade_limit": self.maxdrawdown_trade_limit.value,
            "stop_duration_candles": self.maxdrawdown_stop_duration_candles.value,
            "max_allowed_drawdown": self.maxdrawdown_max_allowed_drawdown.value
        })
        prot.append({
            "method": "StoplossGuard",
            "lookback_period_candles": self.stoplossguard_lookback_period_candles.value,
            "trade_limit": self.stoplossguard_trade_limit.value,
            "stop_duration_candles": self.stoplossguard_stop_duration_candles.value,
            "only_per_pair": False
        })
        return prot

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
                Integer(  6, 120, name="roi_t1"),
                Integer( 60, 200, name="roi_t2"),
                Integer(120, 300, name="roi_t3")
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
        "106": 0,
        "189": -0.02,
        "224": -0.04
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

    rsi_fast_ewo1 = IntParameter(20, 60, default=buy_params["rsi_fast_ewo1"], space="buy", optimize=True)
    rsi_ewo2 = IntParameter(10, 40, default=buy_params["rsi_ewo2"], space="buy", optimize=True)

    smaoffset_optimize = False
    base_nb_candles_buy = IntParameter(10, 50, default=buy_params["base_nb_candles_buy"], space="buy", optimize=smaoffset_optimize)
    base_nb_candles_sell = IntParameter(10, 50, default=sell_params["base_nb_candles_sell"], space="sell", optimize=smaoffset_optimize)
    low_offset = DecimalParameter(0.8, 1.2, default=buy_params["low_offset"], space="buy", decimals=2, optimize=smaoffset_optimize)
    low_offset_2 = DecimalParameter(0.8, 1.2, default=buy_params["low_offset_2"], space="buy", decimals=2, optimize=smaoffset_optimize)
    high_offset = DecimalParameter(0.8, 1.2, default=sell_params["high_offset"], space="sell", decimals=2, optimize=smaoffset_optimize)

    ewo_optimize = True
    fast_ewo = IntParameter(5,40, default=buy_params["fast_ewo"], space="buy", optimize=ewo_optimize)
    slow_ewo = IntParameter(80,250, default=buy_params["slow_ewo"], space="buy", optimize=ewo_optimize)

    protection_optimize = True
    ewo_low = DecimalParameter(-20.0, -4.0, default=buy_params["ewo_low"], space="buy", decimals=2, optimize=protection_optimize)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params["ewo_high"], space="buy", decimals=2, optimize=protection_optimize)
    ewo_high_2 = DecimalParameter(-6.0, 12.0, default=buy_params["ewo_high_2"], space="buy", decimals=2, optimize=protection_optimize)
    rsi_buy = IntParameter(50, 85, default=buy_params["rsi_buy"], space="buy", optimize=protection_optimize)

    dontbuy_optimize = False
    volume_warn = DecimalParameter(0.0, 10.0, default=sell_params["volume_warn"], space="sell", decimals=2, optimize=dontbuy_optimize)
    btc_rsi_8_1h = IntParameter(0, 50, default=sell_params["btc_rsi_8_1h"], space="sell", optimize=dontbuy_optimize)
    percent_change_length = IntParameter(5, 288, default=sell_params["percent_change_length"], space="sell", optimize=dontbuy_optimize)

    pct_chage_optimize = True
    hl_pct_change_06_1h = DecimalParameter(00.30, 0.90, default=sell_params["hl_pct_change_06_1h"], decimals=2, space="sell", optimize=pct_chage_optimize)
    hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=sell_params["hl_pct_change_12_1h"], decimals=2, space="sell", optimize=pct_chage_optimize)
    hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=sell_params["hl_pct_change_24_1h"], decimals=2, space="sell", optimize=pct_chage_optimize)
    hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=sell_params["hl_pct_change_48_1h"], decimals=2, space="sell", optimize=pct_chage_optimize)

    # Optional order time in force.
    order_time_in_force = {
        "entry": "gtc",
        "exit": "ioc"
    }
    slippage_protection = {
        "retries": 3,
        "max_slippage": -0.002
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()

        informative_pairs = []
        for info_timeframe in self.info_timeframes:
            informative_pairs.extend([(pair, info_timeframe) for pair in pairs])

        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_1h)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2.68)
        informative_1h["bb20_2_low"] = bollinger["lower"]
        informative_1h["bb20_2_mid"] = bollinger["mid"]
        informative_1h["bb20_2_upp"] = bollinger["upper"]

        # Pump protections
        informative_1h["hl_pct_change_48"] = range_percent_change(self, informative_1h, "HL", 48)
        informative_1h["hl_pct_change_36"] = range_percent_change(self, informative_1h, "HL", 36)
        informative_1h["hl_pct_change_24"] = range_percent_change(self, informative_1h, "HL", 24)
        informative_1h["hl_pct_change_12"] = range_percent_change(self, informative_1h, "HL", 12)
        informative_1h["hl_pct_change_6"]  = range_percent_change(self, informative_1h, "HL", 6)
        
        return informative_1h

    def informative_30m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_30m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_30m)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_30m), window=20, stds=2.68)
        informative_30m["bb20_2_low"] = bollinger["lower"]
        informative_30m["bb20_2_mid"] = bollinger["mid"]
        informative_30m["bb20_2_upp"] = bollinger["upper"]
        
        return informative_30m
    
    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_15m)

        # BB - 20 STD2
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_15m), window=20, stds=2.68)
        informative_15m["bb20_2_low"] = bollinger["lower"]
        informative_15m["bb20_2_mid"] = bollinger["mid"]
        informative_15m["bb20_2_upp"] = bollinger["upper"]
        
        return informative_15m

    def base_tf_5m_indicators(self, metadata: dict, dataframe: DataFrame) -> DataFrame:
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2.68)
        dataframe["bb20_2_low"] = bollinger["lower"]
        dataframe["bb20_2_mid"] = bollinger["mid"]
        dataframe["bb20_2_upp"] = bollinger["upper"]

        dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_buy.value))
        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_sell.value))
        dataframe["ewo"] = EWO(dataframe, int(self.fast_ewo.value), int(self.slow_ewo.value))
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)

        return dataframe

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df36h = dataframe.copy().shift( 432 )
        df24h = dataframe.copy().shift( 288 )
        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
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
        dataframe       = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        informative_30m = self.informative_30m_indicators(dataframe, metadata)
        dataframe       = merge_informative_pair(dataframe, informative_30m, self.timeframe, self.inf_30m, ffill=True)
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe       = merge_informative_pair(dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)

        dataframe = self.pump_dump_protection(dataframe, metadata)
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
                (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset_2.value)) &
                (dataframe["ewo"] > self.ewo_high_2.value) &
                (dataframe["rsi"] < self.rsi_buy.value) &
                (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value)) &
                (dataframe["rsi"] < self.rsi_ewo2.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "ewo2")
        dataframe.loc[
            (
                (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
                (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
                (dataframe["ewo"] < self.ewo_low.value) &
                (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
            ),
            ["enter_long", "enter_tag"]] = (1, "ewolow")

        dont_buy_conditions = []
        # # don't buy if there seems to be a Pump and Dump event.
        # dont_buy_conditions.append(
        #     (
        #         (dataframe['pnd_volume_warn'] < 0.0)
        #     )
        # )
        # # BTC price protection
        # dont_buy_conditions.append(
        #     (
        #         (dataframe['btc_usdt_rsi_8_1h'] < self.btc_rsi_8_1h.value)
        #     )
        # )
        # # don't buy if the price has changed too much in the last *5 hours
        # dont_buy_conditions.append(
        #     (
        #         ((dataframe['open'].rolling(self.percent_change_length.value).max() - dataframe['close']) / dataframe['close'])
        #     )
        # )
        # pump protections
        dont_buy_conditions.append(
            (
                (dataframe['hl_pct_change_48'] > self.hl_pct_change_48_1h.value) &
                (dataframe['hl_pct_change_36'] > self.hl_pct_change_36_1h.value) &
                (dataframe['hl_pct_change_24'] > self.hl_pct_change_24_1h.value) &
                (dataframe['hl_pct_change_12'] > self.hl_pct_change_12_1h.value) &
                (dataframe['hl_pct_change_06'] > self.hl_pct_change_06_1h.value)
            )
        )

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, "enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

def EWO(dataframe, ema_length=5, ema2_length=35):
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe["low"] * 100

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe["close"] * dataframe["volume"]
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe["volume"], timeperiod=length))
    vwma = vwma.fillna(0, inplace=True)
    return vwma

def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
    if method == "HL":
        return (dataframe["high"].rolling(length).max() - dataframe["low"].rolling(length).min()) / dataframe["low"].rolling(length).min()
    elif method == "OC":
        return (dataframe["open"].rolling(length).max() - dataframe["close"].rolling(length).min()) / dataframe["close"].rolling(length).min()
    else:
        raise ValueError(f"Method {method} not defined!")

def top_percent_change(self, dataframe: DataFrame, length: int) -> float:
    if length == 0:
        return (dataframe["open"] - dataframe["close"]) / dataframe["close"]
    else:
        return (dataframe["open"].rolling(length).max() - dataframe["close"]) / dataframe["close"]