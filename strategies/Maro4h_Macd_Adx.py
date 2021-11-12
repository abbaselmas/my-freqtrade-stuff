# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


class Maro4hMacdAdx(IStrategy):

    max_open_trades = 100
    stake_amount = 1000
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"

    minimal_roi = {
        "0": 100
    }
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -100

    # Optimal timeframe for the strategy
    timeframe = '4h'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.1
    trailing_stop_positive_offset = 0.2

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def informative_pairs(self):
        return [("BTC/USD", "4h"), ("ETH/USD", "4h")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # RSI
        #dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe, timeperiod=14)

        #dataframe['ema'] = ta.EMA(dataframe, timeperiod=200)

        #dataframe['ema2'] = ta.EMA(dataframe, timeperiod=1000)

        #stoch = ta.STOCH(dataframe, fastk_period=200, slowd_period=3, slowk_period=100)
        #dataframe['slowk'] = stoch['slowk']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
            (qtpylib.crossed_above(dataframe['macdhist'],0)) &
            ((dataframe['adx'] >= 25) & (dataframe['di_plus'] > dataframe['di_minus']) &
             (dataframe['adx'] > dataframe['adx'].shift(1)))
            ),'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
            (qtpylib.crossed_below(dataframe['macdhist'],0)) &
            ((dataframe['adx'] >= 25) & (dataframe['di_minus'] > dataframe['di_plus']) &
             (dataframe['adx'] > dataframe['adx'].shift(1)) )
            ),'sell'] = 1

        return dataframe