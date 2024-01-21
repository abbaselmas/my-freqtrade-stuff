from freqtrade.strategy.interface import IStrategy
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
    "base_nb_candles_buy": 36,
    "ewo_high": 4.46,
    "ewo_high_2": -0.56,
    "ewo_low": -6.0,
    "fast_ewo": 6,
    "low_offset": 1.2,
    "low_offset_2": 1.12,
    "pump_factor": 1.67,
    "pump_rolling": 69,
    "rsi_buy": 80,
    "rsi_ewo2": 21,
    "rsi_fast_ewo1": 57,
    "slow_ewo": 229
}
# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 23,
    "high_offset": 1.1,
    "min_profit": 0.73
}

class abbas8(IStrategy):
    def version(self) -> str:
        return "v9.4"
    INTERFACE_VERSION = 3

    pump_factor = DecimalParameter(1.00, 1.70, default = buy_params["pump_factor"] , space = "buy", decimals = 2, optimize = True)
    pump_rolling = IntParameter(2, 100, default = buy_params["pump_rolling"], space="buy", optimize=True)

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
                Integer(  5, 120, name="roi_t1"),
                Integer( 60, 200, name="roi_t2"),
                Integer(120, 300, name="roi_t3")
            ]

        def generate_roi_table(params: Dict) -> Dict[int, float]:
            roi_table = {}
            roi_table[params["roi_t1"]] = 0
            roi_table[params["roi_t2"]] = -0.02
            roi_table[params["roi_t3"]] = -0.04
            return roi_table

    timeframe = "5m"
    inf_1h = "1h"
    minimal_roi = {
        "191": 0,
        "232": -0.02,
        "578": -0.04
    }

    stoploss = -0.067
    trailing_stop = True
    trailing_stop_positive = 0.0003
    trailing_stop_positive_offset = 0.0146
    trailing_only_offset_is_reached = True

    use_exit_signal = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True
    startup_candle_count = 120

    min_profit = DecimalParameter(0.01, 2.00, default=sell_params["min_profit"], space="sell", decimals=2, optimize=True)

    rsi_fast_ewo1 = IntParameter(20, 60, default=buy_params["rsi_fast_ewo1"], space="buy", optimize=True)
    rsi_ewo2 = IntParameter(10, 40, default=buy_params["rsi_ewo2"], space="buy", optimize=True)

    smaoffset_optimize = True
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

    # Optional order time in force.
    order_time_in_force = {
        "entry": "gtc",
        "exit": "ioc"
    }
    slippage_protection = {
        "retries": 3,
        "max_slippage": -0.002
    }

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

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1h") for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_1h)
        informative_1h["ewo"] = EWO(informative_1h, int(self.fast_ewo.value), int(self.slow_ewo.value))
        informative_1h["rsi"] = ta.RSI(informative_1h, timeperiod=14)
        informative_1h["rsi_fast"] = ta.RSI(informative_1h, timeperiod=4)
        informative_1h["rsi_slow"] = ta.RSI(informative_1h, timeperiod=20)
        logger.debug(f"informative 1h dataframe values: {informative_1h.iloc[-1]}")
        return informative_1h

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_buy.value))
        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_sell.value))
        dataframe["ewo"] = EWO(dataframe, int(self.fast_ewo.value), int(self.slow_ewo.value))
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)

        logger.debug(f"populate indicators before combine dataframe values: {dataframe.iloc[-1]}")

        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        logger.debug(f"populate indicators after combine dataframe values: {dataframe.iloc[-1]}")
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
        dont_buy_conditions.append(
            (
                (dataframe["close_1h"].rolling(24).max() < (dataframe["close"] * self.min_profit.value ))
            )
        )
        dont_buy_conditions.append(
            (
                (dataframe["high"].rolling(self.pump_rolling.value).max() >= (dataframe["high"] * self.pump_factor.value ))
            )
        )
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, "enter_long"] = 0
        logger.debug(f"populate entry dataframe values: {dataframe.iloc[-1]}")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.debug(f"populate exit dataframe values: {dataframe.iloc[-1]}")
        return dataframe

def EWO(dataframe, ema_length=5, ema2_length=35):
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe["low"] * 100