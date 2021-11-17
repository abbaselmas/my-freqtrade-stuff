# --- Do not remove these libs ---
# --- Do not remove these libs ---
from logging import FATAL
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt



# Buy hyperspace params:
buy_params = {
    "low_offset": 0.981,
    "base_nb_candles_buy": 8,  # value loaded from strategy
    "ewo_high": 3.553,  # value loaded from strategy
    "ewo_high_2": -5.585,  # value loaded from strategy
    "ewo_low": -14.378,  # value loaded from strategy
    "lookback_candles": 32,  # value loaded from strategy
    "low_offset_2": 0.942,  # value loaded from strategy
    "profit_threshold": 1.037,  # value loaded from strategy
    "rsi_buy": 78,  # value loaded from strategy
    "rsi_fast_buy": 37,  # value loaded from strategy
}

# Sell hyperspace params:
sell_params = {
        "sell_custom_dec_profit_1": 0.002,
        "sell_custom_dec_profit_2": 0.162,
        "sell_custom_profit_0": 0.041,
        "sell_custom_profit_1": 0.02,
        "sell_custom_profit_2": 0.084,
        "sell_custom_profit_3": 0.176,
        "sell_custom_profit_4": 0.511,
        "sell_custom_profit_under_rel_1": 0.015,
        "sell_custom_profit_under_rsi_diff_1": 1.337,
        "sell_custom_rsi_0": 31.342,
        "sell_custom_rsi_1": 36.57,
        "sell_custom_rsi_2": 49.69,
        "sell_custom_rsi_3": 47.78,
        "sell_custom_rsi_4": 42.64,
        "sell_custom_stoploss_under_rel_1": 0.015,
        "sell_custom_stoploss_under_rsi_diff_1": 12.341,
        "sell_custom_under_profit_1": 0.081,
        "sell_custom_under_profit_2": 0.078,
        "sell_custom_under_profit_3": 0.045,
        "sell_custom_under_rsi_1": 45.8,
        "sell_custom_under_rsi_2": 62.8,
        "sell_custom_under_rsi_3": 51.6,
        "sell_trail_down_1": 0.114,
        "sell_trail_down_2": 0.146,
        "sell_trail_down_3": 0.015,
        "sell_trail_profit_max_1": 0.23,
        "sell_trail_profit_max_2": 0.22,
        "sell_trail_profit_max_3": 0.11,
        "sell_trail_profit_min_1": 0.249,
        "sell_trail_profit_min_2": 0.088,
        "sell_trail_profit_min_3": 0.021,
        "base_nb_candles_sell": 16,  # value loaded from strategy
        "high_offset": 1.097,  # value loaded from strategy
        "high_offset_2": 1.472,  # value loaded from strategy
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


class tesla(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        # "0": 0.283,
        # "40": 0.086,
        # "99": 0.036,
        "0": 10,
        
    }

    # Stoploss:
    stoploss = -0.15

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        2, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(
        2, 25, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=False)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=False)
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=False)
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=False)

    sell_custom_profit_0 = DecimalParameter(0.001, 0.1, default=sell_params['sell_custom_profit_0'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_0 = DecimalParameter(30.0, 40.0, default=sell_params['sell_custom_rsi_0'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_profit_1 = DecimalParameter(0.005, 0.1, default=sell_params['sell_custom_profit_1'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_1 = DecimalParameter(30.0, 50.0, default=sell_params['sell_custom_rsi_1'], space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_2 = DecimalParameter(0.007, 0.1, default=sell_params['sell_custom_profit_2'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_2 = DecimalParameter(34.0, 50.0, default=sell_params['sell_custom_rsi_2'], space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_3 = DecimalParameter(0.009, 0.30, default=sell_params['sell_custom_profit_3'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_3 = DecimalParameter(38.0, 55.0, default=sell_params['sell_custom_rsi_3'], space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_4 = DecimalParameter(0.01, 0.6, default=sell_params['sell_custom_profit_4'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_4 = DecimalParameter(40.0, 58.0, default=sell_params['sell_custom_under_profit_1'], space='sell', decimals=2, optimize=True, load=True)

    sell_custom_under_profit_1 = DecimalParameter(0.001, 0.10, default=sell_params['sell_custom_under_profit_1'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_1 = DecimalParameter(36.0, 60.0, default=sell_params['sell_custom_under_rsi_1'], space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_2 = DecimalParameter(0.001, 0.10, default=sell_params['sell_custom_under_profit_2'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_2 = DecimalParameter(46.0, 66.0, default=sell_params['sell_custom_under_rsi_2'], space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_3 = DecimalParameter(0.001, 0.10, default=sell_params['sell_custom_under_profit_3'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_3 = DecimalParameter(50.0, 68.0, default=sell_params['sell_custom_under_rsi_3'], space='sell', decimals=1, optimize=True, load=True)

    sell_custom_dec_profit_1 = DecimalParameter(0.001, 0.10, default=sell_params['sell_custom_dec_profit_1'], space='sell', decimals=3, optimize=True, load=True)
    sell_custom_dec_profit_2 = DecimalParameter(0.05, 0.2, default=sell_params['sell_custom_dec_profit_2'], space='sell', decimals=3, optimize=True, load=True)

    sell_trail_profit_min_1 = DecimalParameter(0.001, 0.25, default=sell_params['sell_trail_profit_min_1'], space='sell', decimals=3, optimize=True, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.03, 0.5, default=sell_params['sell_trail_profit_max_1'], space='sell', decimals=2, optimize=True, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.2, default=sell_params['sell_trail_down_1'], space='sell', decimals=3, optimize=True, load=True)

    sell_trail_profit_min_2 = DecimalParameter(0.004, 0.1, default=sell_params['sell_trail_profit_min_2'], space='sell', decimals=3, optimize=True, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=sell_params['sell_trail_profit_max_2'], space='sell', decimals=2, optimize=True, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=sell_params['sell_trail_down_2'], space='sell', decimals=3, optimize=True, load=True)

    sell_trail_profit_min_3 = DecimalParameter(0.006, 0.1, default=sell_params['sell_trail_profit_min_3'], space='sell', decimals=3, optimize=True, load=True)
    sell_trail_profit_max_3 = DecimalParameter(0.08, 0.16, default=sell_params['sell_trail_profit_max_3'], space='sell', decimals=2, optimize=True, load=True)
    sell_trail_down_3 = DecimalParameter(0.01, 0.04, default=sell_params['sell_trail_down_3'], space='sell', decimals=3, optimize=True, load=True)

    sell_custom_profit_under_rel_1 = DecimalParameter(0.01, 0.04, default=sell_params['sell_custom_profit_under_rel_1'], space='sell', optimize=True, load=True)
    sell_custom_profit_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=sell_params['sell_custom_profit_under_rsi_diff_1'], space='sell', optimize=True, load=True)

    sell_custom_stoploss_under_rel_1 = DecimalParameter(0.001, 0.02, default=sell_params['sell_custom_stoploss_under_rel_1'], space='sell', optimize=True, load=True)
    sell_custom_stoploss_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=sell_params['sell_custom_stoploss_under_rsi_diff_1'], space='sell', optimize=True, load=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    lookback_candles = IntParameter(
        1, 36, default=buy_params['lookback_candles'], space='buy', optimize=False)

    profit_threshold = DecimalParameter(0.99, 1.05,
                                        default=buy_params['profit_threshold'], space='buy', optimize=False)

    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=False)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=False)

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=False)

    rsi_buy = IntParameter(10, 80, default=buy_params['rsi_buy'], space='buy', optimize=False)
    rsi_fast_buy = IntParameter(
        10, 50, default=buy_params['rsi_fast_buy'], space='buy', optimize=False)

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_15m = '15m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = True

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
        'subplots': {
            'rsi': {
                'rsi': {'color': 'orange'},
                'rsi_fast': {'color': 'red'},
                'rsi_slow': {'color': 'green'},
            },
            'ewo': {
                'EWO': {'color': 'orange'}
            },
        }
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if (current_profit > 0.3):
            return 0.05
        elif (current_profit > 0.1):
            return 0.03
        elif (current_profit > 0.06):
            return 0.02
        elif (current_profit > 0.04):
            return 0.01
        elif (current_profit > 0.025):
            return 0.005
        elif (current_profit > 0.018):
            return 0.005

        return 0.15

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)
        if(current_time < trade.open_date_utc + timedelta(minutes=360)) : return None

        if (last_candle is not None):
            if (current_profit > self.sell_custom_profit_4.value) & (last_candle['rsi'] < self.sell_custom_rsi_4.value):
                return 'signal_profit_4'
            elif (current_profit > self.sell_custom_profit_3.value) & (last_candle['rsi'] < self.sell_custom_rsi_3.value):
                return 'signal_profit_3'
            elif (current_profit > self.sell_custom_profit_2.value) & (last_candle['rsi'] < self.sell_custom_rsi_2.value):
                return 'signal_profit_2'
            elif (current_profit > self.sell_custom_profit_1.value) & (last_candle['rsi'] < self.sell_custom_rsi_1.value):
                return 'signal_profit_1'
            elif (current_profit > self.sell_custom_profit_0.value) & (last_candle['rsi'] < self.sell_custom_rsi_0.value):
                return 'signal_profit_0'

            elif (current_profit > self.sell_custom_under_profit_1.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_1.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_1'
            elif (current_profit > self.sell_custom_under_profit_2.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_2.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_2'
            elif (current_profit > self.sell_custom_under_profit_3.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_3.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_3'

            elif (current_profit > self.sell_custom_dec_profit_1.value) & (last_candle['sma_200_dec']):
                return 'signal_profit_d_1'
            elif (current_profit > self.sell_custom_dec_profit_2.value) & (last_candle['close'] < last_candle['ema_100']):
                return 'signal_profit_d_2'

            elif (current_profit > self.sell_trail_profit_min_1.value) & (current_profit < self.sell_trail_profit_max_1.value) & (max_profit > (current_profit + self.sell_trail_down_1.value)):
                return 'signal_profit_t_1'
            elif (current_profit > self.sell_trail_profit_min_2.value) & (current_profit < self.sell_trail_profit_max_2.value) & (max_profit > (current_profit + self.sell_trail_down_2.value)):
                return 'signal_profit_t_2'

            elif (last_candle['close'] < last_candle['ema_200']) & (current_profit > self.sell_trail_profit_min_3.value) & (current_profit < self.sell_trail_profit_max_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)):
                return 'signal_profit_u_t_1'

            

        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951):  # *1.2
                    return False

        # slippage
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '15m') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)
        # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_15m

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma_200_dec'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # informative_1h = self.informative_1h_indicators(dataframe, metadata)
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                # don't buy if there isn't 3% profit to be made
                (dataframe['close_15m'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['rsi'] < 25)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            ((dataframe['close'] > dataframe['sma_9']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
             )
            |
            (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
