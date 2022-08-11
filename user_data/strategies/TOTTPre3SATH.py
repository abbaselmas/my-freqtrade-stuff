# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import decimal
from operator import index
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame  # noqa
from datetime import datetime  # noqa
from typing import Optional, Union, List  # noqa

from freqtrade.optimize.space.decimalspace import SKDecimal


from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import user_data.indicators.finalindicators as myind
import user_data.indicators.helpers as helpers

class TOTTPre3SATH(IStrategy):

    class HyperOpt:
    # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.080, -0.01, decimals=3, name='stoploss')]

        def generate_estimator(dimensions: List['Dimension'], **kwargs):
            from skopt.learning import ExtraTreesRegressor, RandomForestRegressor, GaussianProcessRegressor, GradientBoostingQuantileRegressor
            # Corresponds to "ET" - but allows additional parameters.
            # return "GBRT"
            # return RandomForestRegressor()
            # return "ET"
            return ExtraTreesRegressor(n_estimators=20, n_jobs=-1)







#  # Buy hyperspace params:
# freqtrade    |     buy_params = {
# freqtrade    |         "s_third_ott_length": 0.4,
# freqtrade    |         "s_third_ott_smoothing": 2,
# freqtrade    |         "s_third_perc_length": 0.4,
# freqtrade    |         "s_third_stsd_length": 3,
# freqtrade    |         "s_third_stsk_length": 6,
# freqtrade    |         "s_third_vidya_length": 4,
# freqtrade    |         "s_hhv1_length": 6,  # value loaded from strategy
# freqtrade    |         "s_hhv2_length": 6,  # value loaded from strategy
# freqtrade    |         "s_hhv3_length": 6,  # value loaded from strategy
# freqtrade    |         "s_hott1_length": 0.6,  # value loaded from strategy
# freqtrade    |         "s_hott2_length": 0.4,  # value loaded from strategy
# freqtrade    |         "s_hott3_length": 0.4,  # value loaded from strategy
# freqtrade    |         "s_minor_trend_mult_length": 1.7,  # value loaded from strategy
# freqtrade    |         "s_second_ott_length": 0.4,  # value loaded from strategy
# freqtrade    |         "s_second_ott_smoothing": 1,  # value loaded from strategy
# freqtrade    |         "s_second_perc_length": 0.4,  # value loaded from strategy
# freqtrade    |         "s_second_stsd_length": 2,  # value loaded from strategy
# freqtrade    |         "s_second_stsk_length": 5,  # value loaded from strategy
# freqtrade    |         "s_second_vidya_length": 3,  # value loaded from strategy
# freqtrade    |         "s_trend_atr_length": 5,  # value loaded from strategy
# freqtrade    |         "s_trend_mult_length": 15,  # value loaded from strategy
# freqtrade    |     }
# freqtrade    |
# freqtrade    |     # Sell hyperspace params:
# freqtrade    |     sell_params = {
# freqtrade    |         "s_fifth_ott_length": 0.3,  # value loaded from strategy
# freqtrade    |         "s_fifth_ott_smoothing": 2,  # value loaded from strategy
# freqtrade    |         "s_fifth_perc_length": 0.4,  # value loaded from strategy
# freqtrade    |         "s_fifth_stsd_length": 2,  # value loaded from strategy
# freqtrade    |         "s_fifth_stsk_length": 2,  # value loaded from strategy
# freqtrade    |         "s_fifth_vidya_length": 2,  # value loaded from strategy
# freqtrade    |         "s_fourth_ott_length": 0.4,  # value loaded from strategy
# freqtrade    |         "s_fourth_ott_smoothing": 1,  # value loaded from strategy
# freqtrade    |         "s_fourth_perc_length": 0.4,  # value loaded from strategy
# freqtrade    |         "s_fourth_stsd_length": 2,  # value loaded from strategy
# freqtrade    |         "s_fourth_stsk_length": 4,  # value loaded from strategy
# freqtrade    |         "s_fourth_vidya_length": 2,  # value loaded from strategy
# freqtrade    |         "s_llv2_length": 4,  # value loaded from strategy
# freqtrade    |         "s_llv3_length": 5,  # value loaded from strategy
# freqtrade    |         "s_lott2_length": 0.6,  # value loaded from strategy
# freqtrade    |         "s_lott3_length": 0.5,  # value loaded from strategy
# freqtrade    |         "s_sixth_ott_length": 0.4,  # value loaded from strategy
# freqtrade    |         "s_sixth_ott_smoothing": 1,  # value loaded from strategy
# freqtrade    |         "s_sixth_perc_length": 0.3,  # value loaded from strategy
# freqtrade    |         "s_sixth_stsd_length": 3,  # value loaded from strategy
# freqtrade    |         "s_sixth_stsk_length": 5,  # value loaded from strategy
# freqtrade    |         "s_sixth_vidya_length": 4,  # value loaded from strategy
# freqtrade    |     }







    INTERFACE_VERSION = 3

    timeframe = '3m'
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.99
    }

    stoploss = -0.06

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 500

    optimize_trend = False

    s_trend_atr_length = IntParameter(2, 6, default=4, space="buy",optimize=optimize_trend, load=True)
    s_trend_mult_length = IntParameter(12, 24, default=7, space="buy", optimize=optimize_trend, load=True)
    s_minor_trend_mult_length = DecimalParameter( 1.2, 5.0, default=3.0,  decimals=1, space="buy", optimize=optimize_trend, load=True)
    #2470


    # optimize_first = True

    # first_vidya_length = IntParameter(2, 4, default=3, space="buy",optimize=optimize_first, load=True)
    # first_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="buy",optimize=optimize_first, load=True)
    # first_ott_smoothing = IntParameter(1, 3, default=2, space='buy', optimize=optimize_first, load=True)
    # first_stsk_length = IntParameter(2,6, default=4, space="buy", optimize=optimize_first, load=True)
    # first_stsd_length = IntParameter(2,3, default=2, space="buy", optimize=optimize_first, load=True)
    # first_perc_length = DecimalParameter(0.2, 0.4, default=0.3, decimals=1, space="buy", optimize=optimize_first, load=True)

    s_optimize_second = False 
    s_second_vidya_length = IntParameter(2, 4, default=3, space="buy", optimize=s_optimize_second, load=True)
    s_second_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="buy", optimize=s_optimize_second, load=True)
    s_second_ott_smoothing = IntParameter(1, 3, default=2, space='buy', optimize=s_optimize_second, load=True)
    s_second_stsk_length = IntParameter(2,6, default=4, space="buy", optimize=s_optimize_second, load=True)
    s_second_stsd_length = IntParameter(2,3, default=2, space="buy", optimize=s_optimize_second, load=True)
    s_second_perc_length = DecimalParameter(0.2, 0.4, default=0.3, decimals=1, space="buy", optimize=s_optimize_second, load=True)

    s_optimize_third = True
    s_third_vidya_length = IntParameter(2, 4, default=3, space="buy",optimize=s_optimize_third, load=True)
    s_third_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="buy", optimize=s_optimize_third, load=True)
    s_third_ott_smoothing = IntParameter( 1, 3, default=2, space='buy', optimize=s_optimize_third, load=True)
    s_third_stsk_length = IntParameter(2,6, default=4, space="buy", optimize=s_optimize_third, load=True)
    s_third_stsd_length = IntParameter(2,3, default=2, space="buy", optimize=s_optimize_third, load=True)
    s_third_perc_length = DecimalParameter(0.2, 0.4, default=0.2, decimals=1, space="buy", optimize=s_optimize_third, load=True)

    s_optimize_fourth = False
    s_fourth_vidya_length = IntParameter(2, 4, default=3, space="sell", optimize=s_optimize_fourth, load=True)
    s_fourth_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="sell", optimize=s_optimize_fourth, load=True)
    s_fourth_ott_smoothing = IntParameter(1, 3, default=2, space='sell', optimize=s_optimize_fourth, load=True)
    s_fourth_stsk_length = IntParameter(2,6, default=4, space="sell", optimize=s_optimize_fourth, load=True)
    s_fourth_stsd_length = IntParameter(2,3, default=2, space="sell", optimize=s_optimize_fourth, load=True)
    s_fourth_perc_length = DecimalParameter(0.2, 0.4, default=0.2, decimals=1, space="sell", optimize=s_optimize_fourth, load=True)

    s_optimize_fifth = False
    s_fifth_vidya_length = IntParameter(2, 4, default=3, space="sell", optimize=s_optimize_fifth, load=True)
    s_fifth_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="sell",optimize=s_optimize_fifth, load=True)
    s_fifth_ott_smoothing = IntParameter(1, 3, default=2, space='sell', optimize=s_optimize_fifth, load=True)
    s_fifth_stsk_length = IntParameter(2,6, default=4, space="sell", optimize=s_optimize_fifth, load=True)
    s_fifth_stsd_length = IntParameter(2,3, default=2, space="sell", optimize=s_optimize_fifth, load=True)
    s_fifth_perc_length = DecimalParameter(0.2, 0.4, default=0.2, decimals=1, space="sell", optimize=s_optimize_fifth, load=True)

    s_optimize_sixth = False  
    s_sixth_vidya_length = IntParameter(2, 4, default=3, space="sell",optimize=s_optimize_sixth, load=True)
    s_sixth_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="sell", optimize=s_optimize_sixth, load=True)
    s_sixth_ott_smoothing = IntParameter(1, 3, default=2, space='sell', optimize=s_optimize_sixth, load=True)
    s_sixth_stsk_length = IntParameter(2,6, default=4, space="sell", optimize=s_optimize_sixth, load=True)
    s_sixth_stsd_length = IntParameter(2,3, default=2, space="sell", optimize=s_optimize_sixth, load=True)
    s_sixth_perc_length = DecimalParameter(0.2, 0.4, default=0.2, decimals=1, space="sell", optimize=s_optimize_sixth, load=True)

    s_optimize_patches = False
    s_optimize_patches2 = False
    s_hhv1_length = IntParameter(2, 6, default=4, space='buy', optimize=s_optimize_patches, load=True)
    s_hhv2_length = IntParameter(2, 6, default=4, space='buy', optimize=s_optimize_patches, load=True)
    s_hhv3_length = IntParameter(2, 6, default=4, space='buy', optimize=s_optimize_patches, load=True)

    s_hott1_length = DecimalParameter(0.4, 0.6, default=0.4, decimals=1,space="buy", optimize=s_optimize_patches, load=True)
    s_hott2_length = DecimalParameter(0.4, 0.6, default=0.4, decimals=1,space="buy", optimize=s_optimize_patches, load=True)
    s_hott3_length = DecimalParameter(0.4, 0.6, default=0.4, decimals=1,space="buy", optimize=s_optimize_patches, load=True)

    # s_llv1_length = IntParameter(2, 6, default=4, space='sell', optimize=optimize_patches2, load=True)
    s_llv2_length = IntParameter(2, 6, default=4, space='sell', optimize=s_optimize_patches2, load=True)
    s_llv3_length = IntParameter(2, 6, default=4, space='sell',optimize=s_optimize_patches2, load=True)
    # s_lott1_length = DecimalParameter(0.4, 0.6, default=0.4, decimals=1,  space="sell", optimize=s_optimize_patches2, load=True)
    s_lott2_length = DecimalParameter(0.4, 0.6, default=0.4, decimals=1, space="sell", optimize=s_optimize_patches2, load=True)
    s_lott3_length = DecimalParameter(0.4, 0.6, default=0.4, decimals=1,space="sell", optimize=s_optimize_patches2, load=True)



    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                }
            }
        }

    def informative_pairs(self):
        return []

    def stosk_check(self, stsk1:int, stsk2:int, var: int, ott: decimal) -> bool:

        check = False
        if var == 2:
            if ott == 0.3:
                if stsk1==2 or stsk1 == 3:
                    if stsk2 == 2:
                        check = True
            if ott == 0.4:
                if stsk1 == 4 and stsk2==2:
                    check = True
        if var == 3:
            if ott == 0.3:
                if stsk1==4 and stsk2 == 2:
                    check = True
            if ott == 0.4:
                if stsk1 == 5 and stsk2==2:
                    check = True
                if stsk1 == 4 and stsk2==3:
                    check = True
        if var == 4:
            if ott == 0.3:
                if stsk1==5 and stsk2 == 2:
                    check = True
            if ott == 0.4:
                if stsk1 == 5 and stsk2==3:
                    check = True
                if stsk1 == 6 and stsk2==3:
                    check = True
        return check

    def validty_check(self) -> bool:

        
        # first_check = self.stosk_check(self.s_first_stsk_length.value, self.s_first_stsd_length.value, self.s_first_vidya_length.value, self.s_first_ott_length.value)
        second_check = self.stosk_check(self.s_second_stsk_length.value, self.s_second_stsd_length.value, self.s_second_vidya_length.value, self.s_second_ott_length.value)
        third_check = self.stosk_check(self.s_third_stsk_length.value, self.s_third_stsd_length.value, self.s_third_vidya_length.value, self.s_third_ott_length.value)
        fourth_check = self.stosk_check(self.s_fourth_stsk_length.value, self.s_fourth_stsd_length.value, self.s_fourth_vidya_length.value, self.s_fourth_ott_length.value)
        fifth_check = self.stosk_check(self.s_fifth_stsk_length.value, self.s_fifth_stsd_length.value, self.s_fifth_vidya_length.value, self.s_fifth_ott_length.value)
        sixth_check = self.stosk_check(self.s_sixth_stsk_length.value, self.s_sixth_stsd_length.value, self.s_sixth_vidya_length.value, self.s_sixth_ott_length.value)

        return second_check  & third_check & fourth_check & fifth_check & sixth_check


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        heikinashi["volume"] = dataframe["volume"]

        dataframe['TREND_REGION'] = myind.ott_bolge_signal(dataframe, window=int(self.s_trend_atr_length.value)*10, percent=self.s_trend_mult_length.value/2, percent2=self.s_minor_trend_mult_length.value) 

        pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)

        # indicators = {}

        
        frames = [dataframe]

        for var in  helpers.int_range(self.s_second_vidya_length.low, self.s_second_vidya_length.high):
            for ott in  helpers.decimal_range(self.s_second_ott_length.low, self.s_second_ott_length.high, decimals=1):
                for smt in helpers.int_range(self.s_second_ott_smoothing.low, self.s_second_ott_smoothing.high):
                        up , down = myind.ott_smt_signal(heikinashi, window=var * 10, percent=ott*2.0, smoothing=(smt * 0.0002), field='close', matype='var')
                        frames.append(pd.DataFrame({f'ott_up_{var}_{ott}_{smt}' : up}))
                        frames.append(pd.DataFrame({f'ott_down_{var}_{ott}_{smt}' : down}))
                        #  indicators[f'ott_up_{var}_{ott}_{smt}', f'ott_down_{var}_{ott}_{smt}'] = myind.ott_smt_signal(heikinashi, window=var * 10, percent=ott*2.0, smoothing=(smt * 0.0002), field='close', matype='var')
                        # frames.append(pd.DataFrame({
                        #     f'ott_up_{var}_{ott}_{smt}' : up,
                        #     f'ott_down_{var}_{ott}_{smt}': down
                        # }))
                        # dataframe[f'ott_up_{var}_{ott}_{smt}'], dataframe[f'ott_down_{var}_{ott}_{smt}'] = myind.ott_smt_signal(heikinashi, window=var * 10, percent=ott*2.0, smoothing=(smt * 0.0002), field='close', matype='var')
        print("ott bitti")
        ##OTT 18 columns

        for stsk in helpers.int_range(self.s_second_stsk_length.low,self.s_second_stsk_length.high):
            for stsd in helpers.int_range(self.s_second_stsd_length.low,self.s_second_stsd_length.high):
                for smt in helpers.decimal_range(self.s_second_perc_length.low, self.s_second_perc_length.high, decimals=1):
                    # indicators[f'stsk_{stsk}_{stsd}_{smt}'] = myind.StochVar_Signal(heikinashi, fastk_period=stsk*50, fastd_period=stsd*50, smoothing=smt)
                    frames.append(pd.DataFrame({f'stsk_{stsk}_{stsd}_{smt}' :  myind.StochVar_Signal(heikinashi, fastk_period=stsk*50, fastd_period=stsd*50, smoothing=smt)}
                        ))
                    #   dataframe[f'stsk_{stsk}_{stsd}_{smt}'] = myind.StochVar_Signal(heikinashi, fastk_period=stsk*50, fastd_period=stsd*50, smoothing=smt)
        print("stosk bitti")
        #STOTT 30 column

        for hhv in helpers.int_range(self.s_hhv1_length.low, self.s_hhv1_length.high):
            frames.append(pd.DataFrame({f'hhv_{hhv}' : myind.hhv_signal(heikinashi, hhv * 10)}))
            for hott in helpers.decimal_range(self.s_hott1_length.low,  self.s_hott1_length.high, decimals=1):
                # indicators[f'hhv_{hhv}'] =  myind.hhv_signal(heikinashi, hhv * 10)
                # indicators[f'hott_{hhv}_{hott}'] = myind.hott_signal(heikinashi, hhv * 5, perc=hott)
                frames.append(pd.DataFrame({f'hott_{hhv}_{hott}' : myind.hott_signal(heikinashi, hhv * 5, perc=hott)}))

                # dataframe[f'hhv_{hhv}'] = myind.hhv_signal(heikinashi, hhv * 10)
                # dataframe[f'hott_{hhv}_{hott}'] = myind.hott_signal(heikinashi, hhv * 5, perc=hott)
        print("hhv bitti")

        #HHV 5 column
        #HOTT 15 column

        for llv in  helpers.int_range(self.s_llv2_length.low, self.s_llv2_length.high):
            frames.append(pd.DataFrame({f'llv_{llv}' : myind.llv_signal(heikinashi, llv * 10)}))
            for lott in helpers.decimal_range(self.s_lott2_length.low,  self.s_lott2_length.high, decimals=1):
                frames.append(pd.DataFrame({f'lott_{llv}_{lott}' : myind.lott_signal(heikinashi, llv*5 , perc=lott)}))

                # indicators[f'llv_{llv}'] = myind.llv_signal(heikinashi, llv * 10)
                # indicators[f'lott_{llv}_{lott}'] = myind.lott_signal(heikinashi, llv*5 , perc=lott)

                # dataframe[f'llv_{llv}'] = myind.llv_signal(heikinashi, llv * 10)
                # dataframe[f'lott_{llv}_{lott}'] = myind.lott_signal(heikinashi, llv*5 , perc=lott)
        print("llv bitti")
        #LLV 5 column
        #LOTT 15 column



        # indicators = pd.DataFrame(indicators, index=dataframe.index)
        # dataframe = pd.concat([dataframe, indicators], axis=1)
        dataframe2 =  pd.concat(frames, axis=1,verify_integrity=True).convert_dtypes()
        
        # dataframe2 = pd.concat([dataframe, indicators], axis=1).convert_dtypes()

        print(dataframe2.dtypes)
        print(dataframe2.columns.to_list())

        # print(dataframe.tail(3))
        # print(dataframe2.head(20))
        print(dataframe2.__len__)
        print(dataframe.__len__)
        # print(dataframe.__len__ == dataframe2.__len__)

        # dataframe2 = dataframe.copy()
        # dataframe = dataframe2
        ##toplam 1150 column
        return dataframe2

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.validty_check() == False:
            return dataframe
        
        # dataframe.loc[
        #     (
        #         (
        #             (
        #                 (dataframe['TREND_OTT'] == "buy")
        #                 &
        #                 (dataframe['MIN_TREND_OTT'] == "buy")
        #                 &
        #                 (dataframe['FIRST_DOWN'] == "sell")
        #                 &
        #                 (dataframe['FIRST_STSK'] == "sell")
        #                 &
        #                 (dataframe['LLV1'] == "sell")
        #                 &
        #                 (dataframe['LOTT1'] == "sell")
        #             )
        #         )
        #         &
        #         (dataframe['volume'] > 0)
        #     ),
        #     ['enter_short', 'enter_tag']] = (1, 'short_1')

        # # Heiken Ashi
        # heikinashi = qtpylib.heikinashi(dataframe)
        # heikinashi["volume"] = dataframe["volume"]


        # MAV > OTT:
        #     MAV > ROTT : Bolge1 (Trend Long)  (Trend LongExit) - (NoTrendShortExit)
        #     MAV < ROTT : Bolge2 (Trend Long)  (Trend LongExit) (Firsatci ShortEntry) (NoTrendShortExit)
        # MAV < OTT:
        #     MAV > ROTT : Bolge3 (Firsatci Long) (NoTrend LongExit) -- (Short Entry) - (ShortExit)
        #     MAV < ROTT : Bolge4  (NoTrendLongExit)      -- (ShortEntry)  - (ShortExit)  

        dataframe.loc[
            (
                (dataframe['TREND_REGION'].eq(2))
                &
                (dataframe[f'ott_down_{self.s_second_vidya_length.value}_{self.s_second_ott_length.value}_{self.s_second_ott_smoothing.value}'].eq("sell"))
                &
                (dataframe[f'stsk_{self.s_second_stsk_length.value}_{self.s_second_stsd_length.value}_{self.s_second_perc_length.value}'].eq("sell"))
                &
                (dataframe[f'llv_{self.s_llv2_length.value}'].eq("sell"))
                &
                (dataframe[f'lott_{self.s_llv2_length.value}_{self.s_lott2_length.value}'].eq("sell"))
                &
                (dataframe['volume'] > 0.0)
            ),
            ['enter_short', 'enter_tag']] = (1, 'short_2')
        
        dataframe.loc[
            (
                (
                    (
                        ((dataframe['TREND_REGION'] == 3) | (dataframe['TREND_REGION'] == 4))
                        &
                        (dataframe[f'ott_down_{self.s_third_vidya_length.value}_{self.s_third_ott_length.value}_{self.s_third_ott_smoothing.value}'] == "sell")
                        &
                        (dataframe[f'stsk_{self.s_third_stsk_length.value}_{self.s_third_stsd_length.value}_{self.s_third_perc_length.value}'] == "sell")
                        &
                        (dataframe[f'llv_{self.s_llv3_length.value}'] == "sell")
                        &
                        (dataframe[f'lott_{self.s_llv3_length.value}_{self.s_lott3_length.value}'] == "sell")
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            ['enter_short', 'enter_tag']] = (1, 'short_3')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.validty_check() == False:
            return dataframe
        dataframe.loc[
            (
                (
                    (
                        ((dataframe['TREND_REGION']== 1))
                        &
                        (dataframe[f'ott_up_{self.s_fourth_vidya_length.value}_{self.s_fourth_ott_length.value}_{self.s_fourth_ott_smoothing.value}'] == "buy")
                        &
                        (dataframe[f'stsk_{self.s_fourth_stsk_length.value}_{self.s_fourth_stsd_length.value}_{self.s_fourth_perc_length.value}'] == "buy")
                        &
                        (dataframe[f'hhv_{self.s_hhv1_length.value}'] == "buy")
                        &
                        (dataframe[f'hott_{self.s_hhv1_length.value}_{self.s_hott1_length.value}'] == "buy")
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            ['exit_short', 'exit_tag']] = (1, 'short_ex_4')

        dataframe.loc[
            (
                (
                    (
                        ((dataframe['TREND_REGION']== 2))
                        &
                        (dataframe[f'ott_up_{self.s_fifth_vidya_length.value}_{self.s_fifth_ott_length.value}_{self.s_fifth_ott_smoothing.value}'] == "buy")
                        &
                        (dataframe[f'stsk_{self.s_fifth_stsk_length.value}_{self.s_fifth_stsd_length.value}_{self.s_fifth_perc_length.value}'] == "buy")
                        &
                        (dataframe[f'hhv_{self.s_hhv2_length.value}'] == "buy")
                        &
                        (dataframe[f'hott_{self.s_hhv2_length.value}_{self.s_hott2_length.value}'] == "buy")
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            ['exit_short', 'exit_tag']] = (1, 'short_ex_5')
        
        dataframe.loc[
            (
                (
                    (
                        ((dataframe['TREND_REGION']== 3) | (dataframe['TREND_REGION']== 4))
                        &
                        (dataframe[f'ott_up_{self.s_sixth_vidya_length.value}_{self.s_sixth_ott_length.value}_{self.s_sixth_ott_smoothing.value}'] == "buy")
                        &
                        (dataframe[f'stsk_{self.s_sixth_stsk_length.value}_{self.s_sixth_stsd_length.value}_{self.s_sixth_perc_length.value}'] == "buy")
                        &
                        (dataframe[f'hhv_{self.s_hhv3_length.value}'] == "buy")
                        &
                        (dataframe[f'hott_{self.s_hhv3_length.value}_{self.s_hott3_length.value}'] == "buy")
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            ['exit_short', 'exit_tag']] = (1, 'short_ex_6')
        return dataframe
    
    def bot_loop_start(self, **kwargs) -> None:
        pass

    def custom_entry_price(self, pair: str, current_time: 'datetime', proposed_rate: float,
                           entry_tag: 'Optional[str]', side: str, **kwargs) -> float:
        return proposed_rate

    def adjust_entry_price(self, trade: 'Trade', order: 'Optional[Order]', pair: str,
                            current_time: datetime, proposed_rate: float, current_order_rate: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:


        return current_order_rate

    def custom_exit_price(self, pair: str, trade: 'Trade',
                          current_time: 'datetime', proposed_rate: float,
                          current_profit: float, exit_tag: Optional[str], **kwargs) -> float:
        return proposed_rate

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        return proposed_stake

    use_custom_stoploss = False

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:

        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> 'Optional[Union[str, bool]]':
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: 'datetime', **kwargs) -> bool:
       
        return True

    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                            current_time: datetime, **kwargs) -> bool:
      
        return False

    def check_exit_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                           current_time: datetime, **kwargs) -> bool:
       
        return False

    def adjust_trade_position(self, trade: 'Trade', current_time: 'datetime',
                              current_rate: float, current_profit: float, min_stake: Optional[float],
                              max_stake: float, **kwargs) -> 'Optional[float]':
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 1.0