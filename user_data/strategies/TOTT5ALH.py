# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import decimal
from typing import List
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from datetime import datetime


from freqtrade.optimize.space.decimalspace import SKDecimal


from freqtrade.strategy import (DecimalParameter, IStrategy, IntParameter)
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import user_data.indicators.finalindicators as myind


class TOTT5ALH(IStrategy):
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

    # cooldown_lookback = IntParameter(
    #     2, 360, default=5, space="protection", optimize=False, load=True)

    # @property
    # def protections(self):
    #     return [
    #         {
    #             "method": "CooldownPeriod",
    #             "stop_duration_candles": self.cooldown_lookback.value
    #         }
    #     ]
    #     return prot
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.99
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.066

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '3m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 600

    optimize_trend = True

    trend_atr_length = IntParameter(2, 6, default=4, space="buy",optimize=optimize_trend, load=True)
    trend_mult_length = IntParameter(12, 24, default=7, space="buy", optimize=optimize_trend, load=True)
    minor_trend_mult_length = DecimalParameter( 1.2, 5.0, default=3.0,  decimals=1, space="buy", optimize=optimize_trend, load=True)
    #2470


    optimize_first = False

    first_vidya_length = IntParameter(2, 4, default=3, space="buy",optimize=optimize_first, load=True)
    first_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="buy",optimize=optimize_first, load=True)
    first_ott_smoothing = IntParameter(2, 4, default=2, space='buy', optimize=optimize_first, load=True)
    first_stsk_length = IntParameter(2,6, default=4, space="buy", optimize=optimize_first, load=True)
    first_stsd_length = IntParameter(2,3, default=2, space="buy", optimize=optimize_first, load=True)
    first_perc_length = DecimalParameter(0.2, 0.4, default=0.3, decimals=1, space="buy", optimize=optimize_first, load=True)
    #540

    optimize_second = False 
    second_vidya_length = IntParameter(2, 4, default=3, space="buy", optimize=optimize_second, load=True)
    second_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="buy", optimize=optimize_second, load=True)
    second_ott_smoothing = IntParameter(2, 4, default=2, space='buy', optimize=optimize_second, load=True)
    second_stsk_length = IntParameter(2,6, default=4, space="buy", optimize=optimize_second, load=True)
    second_stsd_length = IntParameter(2,3, default=2, space="buy", optimize=optimize_second, load=True)
    second_perc_length = DecimalParameter(0.2, 0.4, default=0.3, decimals=1, space="buy", optimize=optimize_second, load=True)

    optimize_third = False
    third_vidya_length = IntParameter(2, 4, default=3, space="sell",optimize=optimize_third, load=True)
    third_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="sell", optimize=optimize_third, load=True)
    third_ott_smoothing = IntParameter( 2, 4, default=2, space='sell', optimize=optimize_third, load=True)
    third_stsk_length = IntParameter(2,6, default=4, space="sell", optimize=optimize_third, load=True)
    third_stsd_length = IntParameter(2,3, default=2, space="sell", optimize=optimize_third, load=True)
    third_perc_length = DecimalParameter(0.2, 0.4, default=0.2, decimals=1, space="sell", optimize=optimize_third, load=True)

    optimize_fourth = False
    fourth_vidya_length = IntParameter(2, 4, default=3, space="sell", optimize=optimize_fourth, load=True)
    fourth_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="sell", optimize=optimize_fourth, load=True)
    fourth_ott_smoothing = IntParameter(2, 4, default=2, space='sell', optimize=optimize_fourth, load=True)
    fourth_stsk_length = IntParameter(2,6, default=4, space="sell", optimize=optimize_fourth, load=True)
    fourth_stsd_length = IntParameter(2,3, default=2, space="sell", optimize=optimize_fourth, load=True)
    fourth_perc_length = DecimalParameter(0.2, 0.4, default=0.2, decimals=1, space="sell", optimize=optimize_fourth, load=True)

    optimize_fifth = False
    fifth_vidya_length = IntParameter(2, 4, default=3, space="sell", optimize=optimize_fifth, load=True)
    fifth_ott_length = DecimalParameter(0.3, 0.4, default=0.3, decimals=1, space="sell",optimize=optimize_fifth, load=True)
    fifth_ott_smoothing = IntParameter(2, 4, default=2, space='sell', optimize=optimize_fifth, load=True)
    fifth_stsk_length = IntParameter(2,6, default=4, space="sell", optimize=optimize_fifth, load=True)
    fifth_stsd_length = IntParameter(2,3, default=2, space="sell", optimize=optimize_fifth, load=True)
    fifth_perc_length = DecimalParameter(0.2, 0.4, default=0.2, decimals=1, space="sell", optimize=optimize_fifth, load=True)
    
    optimize_patches = False
    hhv1_length = IntParameter(2, 6, default=4, space='buy', optimize=optimize_patches, load=True)
    hhv2_length = IntParameter(2, 6, default=4, space='buy', optimize=optimize_patches, load=True)

    hott1_length = DecimalParameter(0.4, 0.6, default=0.4, decimals=1,
                                    space="buy", optimize=optimize_patches, load=True)

    optimize_patches2 = False
    llv1_length = IntParameter(2, 6, default=4, space='buy',
                               optimize=optimize_patches2, load=True)
    llv2_length = IntParameter(2, 6, default=4, space='buy',
                               optimize=optimize_patches2, load=True)
    llv3_length = IntParameter(2, 6, default=4, space='buy',
                               optimize=optimize_patches2, load=True)
    #1250



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

        
        first_check = self.stosk_check(self.first_stsk_length.value, self.first_stsd_length.value, self.first_vidya_length.value, self.first_ott_length.value)
        second_check = self.stosk_check(self.second_stsk_length.value, self.second_stsd_length.value, self.second_vidya_length.value, self.second_ott_length.value)
        third_check = self.stosk_check(self.third_stsk_length.value, self.third_stsd_length.value, self.third_vidya_length.value, self.third_ott_length.value)
        fourth_check = self.stosk_check(self.fourth_stsk_length.value, self.fourth_stsd_length.value, self.fourth_vidya_length.value, self.fourth_ott_length.value)
        fifth_check = self.stosk_check(self.fifth_stsk_length.value, self.fifth_stsd_length.value, self.fifth_vidya_length.value, self.fifth_ott_length.value)
       
        return first_check & second_check & third_check & fourth_check & fifth_check


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        heikinashi["volume"] = dataframe["volume"]

        # 1.Bolge
        dataframe['FIRST_UP'], _= myind.ott_smt_signal(heikinashi, window=int(self.first_vidya_length.value) * 10, percent=self.first_ott_length.value*2.0, smoothing=(self.first_ott_smoothing.value * 0.0002), field='close', matype='var')
        dataframe['FIRST_STSK'] = myind.StochVar_Signal(heikinashi, fastk_period=self.first_stsk_length.value*50, fastd_period=self.first_stsd_length.value*50, smoothing=self.first_perc_length.value)
        
        #2.Bolge
        dataframe['SECOND_UP'], _ = myind.ott_smt_signal(heikinashi, window=int(self.second_vidya_length.value) * 10, percent=self.second_ott_length.value*2.0, smoothing=(self.second_ott_smoothing.value * 0.0002), field='close', matype='var')
        dataframe['SECOND_STSK'] = myind.StochVar_Signal(heikinashi, fastk_period=self.second_stsk_length.value*50, fastd_period=self.second_stsd_length.value*50, smoothing=self.second_perc_length.value)
        
        # 3.Bolge
        _, dataframe['THIRD_DOWN'] = myind.ott_smt_signal(heikinashi, window=int(self.third_vidya_length.value) * 10, percent=self.third_ott_length.value*2.0, smoothing=(self.third_ott_smoothing.value * 0.0002), field='close', matype='var')
        dataframe['THIRD_STSK'] = myind.StochVar_Signal(heikinashi, fastk_period=self.third_stsk_length.value*50, fastd_period=self.third_stsd_length.value*50, smoothing=self.third_perc_length.value)
       
          # 4.Bolge
        _, dataframe['FOURTH_DOWN'] = myind.ott_smt_signal(heikinashi, window=int(self.fourth_vidya_length.value) * 10, percent=self.fourth_ott_length.value * 2.0, smoothing=( self.fourth_ott_smoothing.value * 0.0002), field='close', matype='var')
        dataframe['FOURTH_STSK'] = myind.StochVar_Signal(heikinashi, fastk_period=self.fourth_stsk_length.value*50, fastd_period=self.fourth_stsd_length.value*50, smoothing=self.fourth_perc_length.value)

        # 5.Bolge
        _, dataframe['FIFTH_DOWN'] = myind.ott_smt_signal(heikinashi, window=int(self.fifth_vidya_length.value) * 10, percent=self.fifth_ott_length.value * 2.0, smoothing=( self.fifth_ott_smoothing.value * 0.0002), field='close', matype='var')
        dataframe['FIFTH_STSK'] = myind.StochVar_Signal(heikinashi, fastk_period=self.fifth_stsk_length.value*50, fastd_period=self.fifth_stsd_length.value*50, smoothing=self.fifth_perc_length.value)
            
        # PATCHES
        dataframe['HHV1'] = myind.hhv_signal(dataframe, period=self.hhv1_length.value * 10)
        dataframe['HHV2'] = myind.hhv_signal(dataframe, period=self.hhv2_length.value * 10)

        dataframe['HOTT1'] = myind.hott_signal(dataframe, period=self.hhv1_length.value*5, shft=0, perc=self.hott1_length.value)
        dataframe['HOTT2'] = myind.hott_signal(dataframe, period=self.hhv2_length.value*5, shft=0, perc=self.hott1_length.value)


    
        dataframe['LLV1'] = myind.llv_signal(dataframe, period=self.llv1_length.value * 10)
        dataframe['LLV2'] = myind.llv_signal(dataframe, period=self.llv2_length.value * 10)
        dataframe['LLV3'] = myind.llv_signal(dataframe, period=self.llv3_length.value * 10)


        dataframe['LOTT1'] = myind.lott_signal(dataframe, period=self.llv1_length.value*5, shft=0, perc=self.hott1_length.value)
        dataframe['LOTT2'] = myind.lott_signal(dataframe, period=self.llv2_length.value*5, shft=0, perc=self.hott1_length.value)
        dataframe['LOTT3'] = myind.lott_signal(dataframe, period=self.llv3_length.value*5, shft=0, perc=self.hott1_length.value)



       
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        if self.validty_check() == False:
            return dataframe
        # # Heiken Ashi
        # heikinashi = qtpylib.heikinashi(dataframe)
        # heikinashi["volume"] = dataframe["volume"]


        
        dataframe['TREND_REGION'] = myind.ott_bolge_signal(dataframe, window=int(self.trend_atr_length.value)*10, percent=self.trend_mult_length.value/2, percent2=self.minor_trend_mult_length.value) 


        # MAV > OTT:
        #     MAV > ROTT : Bolge1 (Trend Long)  (Trend LongExit) - (NoTrendShortExit)
        #     MAV < ROTT : Bolge2 (Trend Long)  (Trend LongExit) (Firsatci ShortEntry) (NoTrendShortExit)
        # MAV < OTT:
        #     MAV > ROTT : Bolge3 (Firsatci Long) (NoTrend LongExit) -- (Short Entry) - (ShortExit)
        #     MAV < ROTT : Bolge4  (NoTrendLongExit)      -- (ShortEntry)  - (ShortExit)  

        dataframe.loc[
            (
                (
                    (
                        ((dataframe['TREND_REGION']== 1) | (dataframe['TREND_REGION']== 2) )
                        &
                        (dataframe['FIRST_UP'] == "buy")
                        &
                        (dataframe['FIRST_STSK'] == "buy")
                        &
                        (dataframe['HHV1'] == "buy")
                        &
                        (dataframe['HOTT1'] == "buy")
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_1')

        dataframe.loc[
            (
                (
                    (
                        (dataframe['TREND_REGION'] == 3)
                        &
                        (dataframe['SECOND_UP'] == "buy")
                        &
                        (dataframe['SECOND_STSK'] == "buy")
                        &
                        (dataframe['HHV2'] == "buy")
                        &
                        (dataframe['HOTT2'] == "buy")
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_2')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.validty_check() == False:
            return dataframe



          

        # MAV > OTT:
        #     MAV > ROTT : Bolge1 (Trend Long)  (Trend LongExit) - (NoTrendShortExit)
        #     MAV < ROTT : Bolge2 (Trend Long)  (Trend LongExit) (Firsatci ShortEntry) (NoTrendShortExit)
        # MAV < OTT:
        #     MAV > ROTT : Bolge3 (Firsatci Long) (NoTrend LongExit) -- (Short Entry) - (ShortExit)
        #     MAV < ROTT : Bolge4  (NoTrendLongExit)      -- (ShortEntry)  - (ShortExit)  

        dataframe.loc[
            (
                (
                    (
                        (dataframe['TREND_REGION']== 1)
                        &
                        (dataframe['THIRD_DOWN'] == "sell")
                        &
                        (dataframe['THIRD_STSK'] == "sell")
                        &
                        (dataframe['LLV1'] == "sell")
                        &
                        (dataframe['LOTT1'] == "sell")
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'sell_3')

        dataframe.loc[
            (
                (
                    (
                        ((dataframe['TREND_REGION']== 2))
                        &
                        (dataframe['FOURTH_DOWN'] == "sell")
                        &
                        (dataframe['FOURTH_STSK'] == "sell")
                        &
                        (dataframe['LLV2'] == "sell")
                        &
                        (dataframe['LOTT2'] == "sell")
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'sell_4')
    
        dataframe.loc[
            (
                (
                    (
                        ((dataframe['TREND_REGION']== 3) | (dataframe['TREND_REGION']== 4))
                        &
                        (dataframe['FIFTH_DOWN'] == "sell")
                        &
                        (dataframe['FIFTH_STSK'] == "sell")
                        &
                        (dataframe['LLV3'] == "sell")
                        &
                        (dataframe['LOTT3'] == "sell")
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'sell_5')
        return dataframe

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote ressource for comparison)

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, this simply does nothing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass

    def custom_stake_amount(self, pair: str, current_time: 'datetime', current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        """
        Customize stake size for each new trade. This method is not called when edge module is
        enabled.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param proposed_stake: A stake amount proposed by the bot.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :return: A stake size, which is between min_stake and max_stake.
        """
        return proposed_stake

    use_custom_stoploss = False

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the current_rate
        """
        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> 'Optional[Union[str, bool]]':
        """
        Custom sell signal logic indicating that specified position should be sold. Returning a
        string or True from this method is equal to setting sell signal on a candle at specified
        time. This method is not called when sell signal is set.

        This method should be overridden to create sell signals that depend on trade parameters. For
        example you could implement a sell relative to the candle when the trade was opened,
        or a custom 1:2 risk-reward ROI.

        Custom sell reason max length is 64. Exceeding characters will be removed.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: To execute sell, return a string with custom sell reason or True. Otherwise return
        None or False.
        """
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: 'datetime', **kwargs) -> bool:
        """
        Called right before placing a buy order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (quote) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: 'datetime', **kwargs) -> bool:
        """
        Called right before placing a regular sell order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in quote currency.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param sell_reason: Sell reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                            'sell_signal', 'force_sell', 'emergency_sell']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the sell-order is placed on the exchange.
            False aborts the process
        """
        return True

    def check_entry_timeout(self, pair: str, trade: 'Trade', order: dict, **kwargs) -> bool:
        """
        Check buy timeout function callback.
        This method can be used to override the buy-timeout.
        It is called whenever a limit buy order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: trade object.
        :param order: Order dictionary as returned from CCXT.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is cancelled.
        """
        return False

    def check_exit_timeout(self, pair: str, trade: 'Trade', order: dict, **kwargs) -> bool:
        """
        Check sell timeout function callback.
        This method can be used to override the sell-timeout.
        It is called whenever a limit sell order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: trade object.
        :param order: Order dictionary as returned from CCXT.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the sell-order is cancelled.
        """
        return False
