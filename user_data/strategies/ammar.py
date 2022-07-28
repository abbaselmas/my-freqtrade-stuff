from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 18,
    "ewo_high": 2.102,
    "ewo_high_2": -2.92,
    "ewo_low": -8.27,
    "low_offset": 1.079,
    "low_offset_2": 0.95,
    "rsi_buy": 55,
    "rsi_fast_buy": 35,
    "profit_threshold": 1.0408,
    "lookback_candles": 22,
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 17,
    "high_offset": 1.07,
}

# Protection hyperspace params:
protection_params = {
    "cooldown_stop_duration_candles": 0,
}

class ammar(IStrategy):

    INTERFACE_VERSION = 2

    cooldown_stop_duration_candles = IntParameter(0, 20, default=protection_params["cooldown_stop_duration_candles"], space="protection", optimize=True)

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_stop_duration_candles.value
        })

        return prot

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.150, -0.030, decimals=3, name="stoploss")]

        # Define custom trailing space
        def trailing_space() -> List[Dimension]:
            return[
                Categorical([True], name="trailing_stop"),
                SKDecimal(0.0001, 0.0010, decimals=4, name="trailing_stop_positive"),
                SKDecimal(0.0080, 0.0200, decimals=4, name="trailing_stop_positive_offset_p1"),
                Categorical([True], name="trailing_only_offset_is_reached"),
            ]

    # ROI table:
    minimal_roi = {
        "200": 0
    }

    # Stoploss:
    stoploss = -0.061

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0002
    trailing_stop_positive_offset = 0.0136
    trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = False
    ignore_roi_if_entry_signal = False

    # SMAOffset
    smaoffset_optimize = True
    base_nb_candles_buy = IntParameter(15, 30, default=buy_params["base_nb_candles_buy"], space="buy", optimize=smaoffset_optimize)
    base_nb_candles_sell = IntParameter(5, 30, default=sell_params["base_nb_candles_sell"], space="sell", optimize=smaoffset_optimize)
    low_offset = DecimalParameter(1.0, 1.1, default=buy_params["low_offset"], space="buy", decimals=3, optimize=smaoffset_optimize)
    low_offset_2 = DecimalParameter(0.94, 0.98, default=buy_params["low_offset_2"], space="buy", decimals=3, optimize=smaoffset_optimize)
    high_offset = DecimalParameter(1.0, 1.1, default=sell_params["high_offset"], space="sell", decimals=3, optimize=smaoffset_optimize)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    protection_optimize = True
    ewo_low = DecimalParameter(-12.0, -8.0,default=buy_params["ewo_low"], space="buy", decimals=2, optimize=protection_optimize)
    ewo_high = DecimalParameter(1.0, 2.2, default=buy_params["ewo_high"], space="buy", decimals=3, optimize=protection_optimize)
    ewo_high_2 = DecimalParameter(-4.0, -2.0, default=buy_params["ewo_high_2"], space="buy", decimals=2, optimize=protection_optimize)
    rsi_buy = IntParameter(55, 85, default=buy_params["rsi_buy"], space="buy", optimize=protection_optimize)

    rsi_fast_buy = IntParameter(25, 45, default=buy_params["rsi_fast_buy"], space="buy", optimize=protection_optimize)
    profit_threshold = DecimalParameter(0.99, 1.05, default=buy_params["profit_threshold"], space="buy", optimize=protection_optimize)
    lookback_candles = IntParameter(1, 36, default=buy_params["lookback_candles"], space="buy", optimize=protection_optimize)

    # Optional order time in force.
    order_time_in_force = {
        "entry": "gtc",
        "exit": "ioc"
    }

    # Optimal timeframe for the strategy
    timeframe = "5m"
    inf_15m = "15m"

    process_only_new_candles = True
    startup_candle_count = 200

    plot_config = {
        "main_plot": {
            "bb_upperband28": {"color": "#bc281d","type": "line"},
            "bb_lowerband28": {"color": "#792bbb","type": "line"}
        },
        "subplots": {
            "rsi": {
                "rsi": {"color": "orange"},
                "rsi_fast": {"color": "red"},
                "rsi_slow": {"color": "green"},
            },
            "ewo": {
                "EWO": {"color": "orange"}
            },
        }
    }

    slippage_protection = {
        "retries": 3,
        "max_slippage": -0.002
    }

    buy_signals = {}

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, sell_reason: str, current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # slippage
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
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, "15m") for pair in pairs]
        return informative_pairs

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_15m)

        return informative_15m

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f"ma_buy_{val}"] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f"ma_sell_{val}"] = ta.EMA(dataframe, timeperiod=val)

        # Elliot
        dataframe["EWO"] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2.8)
        dataframe["bb_lowerband28"] = bollinger2["lower"]
        dataframe["bb_middleband28"] = bollinger2["mid"]
        dataframe["bb_upperband28"] = bollinger2["upper"]

        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["rsi_fast"] < self.rsi_fast_buy.value) & #35
                (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
                (dataframe["EWO"] > self.ewo_high.value) &
                (dataframe["rsi"] < self.rsi_buy.value) &
                (dataframe["volume"] > 0) &
                (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
            ),
            ["buy", "buy_tag"]] = (1, "ewo1")

        dataframe.loc[
            (
                (dataframe["rsi_fast"] < self.rsi_fast_buy.value) & #35
                (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset_2.value)) &
                (dataframe["EWO"] > self.ewo_high_2.value) &
                (dataframe["rsi"] < self.rsi_buy.value) &
                (dataframe["volume"] > 0) &
                (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value)) &
                (dataframe["rsi"] < 25)
            ),
            ["buy", "buy_tag"]] = (1, "ewo2")

        dataframe.loc[
            (
                (dataframe["rsi_fast"] < self.rsi_fast_buy.value) & #35
                (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
                (dataframe["EWO"] < self.ewo_low.value) &
                (dataframe["volume"] > 0) &
                (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
            ),
            ["buy", "buy_tag"]] = (1, "ewolow")

        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                (dataframe["close_15m"].rolling(self.lookback_candles.value).max() < (dataframe["close"] * self.profit_threshold.value))
            )
        )

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, "buy"] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["low"] * 100
    return emadif
