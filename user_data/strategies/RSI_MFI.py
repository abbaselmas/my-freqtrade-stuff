# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.hyper import IntParameter
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade

import logging
logger = logging.getLogger(__name__)

class RSI_MFI(IStrategy):
    INTERFACE_VERSION = 2
    # 129/1000:     22 trades. 21/0/1 Wins/Draws/Losses. Avg profit   7.62%. Median profit   4.26%.
    #Total profit  662.16442958 USD (  66.22%). Avg duration 2 days, 15:06:00 min. Objective: -27486.19805

     
    buy_rsi = IntParameter(1, 25, default=18, space="buy") 
    buy_mfi = IntParameter(1, 15, default=2, space="buy") 
    sell_rsi = IntParameter(75, 99, default=91, space="sell") 
    sell_mfi = IntParameter(85, 99, default=91, space="sell") 
    
    # ROI table:
    minimal_roi = {
        #"0": 0.10,
        #"40": 0.05,
        #"92": 0.03,
        #"210": 0.005
        "0": 10
    }

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.011
    trailing_stop_positive_offset = 0.085
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy 
    timeframe = '5m'
    
    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = False
    
    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # run "populate_indicators" only for new candle
    process_only_new_candles = False
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=8)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=4)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=8)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
                
        dataframe.loc[
            (
                (dataframe["rsi"] <= self.buy_rsi.value) &
                (dataframe["mfi"] <= self.buy_mfi.value) &
                (dataframe['roc'] <= -1) & #guard
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ## functionally not used until fixed

        dataframe.loc[
            (    
                (dataframe["rsi"] >= self.sell_rsi.value) &
                (dataframe["mfi"] >= self.sell_mfi.value) &
                (dataframe['roc'] >= 1) & #guard
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'sell'] = 1
        return dataframe