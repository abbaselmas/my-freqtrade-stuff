# --- Do not remove these libs ---
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
from freqtrade.exchange import timeframe_to_prev_date

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 12,
    "ewo_high": 4.149,
    "ewo_high_2": 7.366,
    "ewo_low": -8.407,
    "low_offset": 0.977,
    "low_offset_2": 0.924,
    "rsi_buy": 43,
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 38,
    "high_offset": 1.022,
    "high_offset_2": 1.004,
    "high_offset_ema": 1.07,
    "sell_custom_dec_profit_1": 0.089,
    "sell_custom_dec_profit_2": 0.191,
    "sell_custom_profit_0": 0.089,
    "sell_custom_profit_1": 0.033,
    "sell_custom_profit_2": 0.058,
    "sell_custom_profit_3": 0.078,
    "sell_custom_profit_4": 0.553,
    "sell_custom_profit_under_rel_1": 0.012,
    "sell_custom_profit_under_rsi_diff_1": 4.856,
    "sell_custom_rsi_0": 37.362,
    "sell_custom_rsi_1": 38.11,
    "sell_custom_rsi_2": 38.67,
    "sell_custom_rsi_3": 54.36,
    "sell_custom_rsi_4": 43.72,
    "sell_custom_stoploss_under_rel_1": 0.005,
    "sell_custom_stoploss_under_rsi_diff_1": 14.02,
    "sell_custom_under_profit_1": 0.083,
    "sell_custom_under_profit_2": 0.097,
    "sell_custom_under_profit_3": 0.083,
    "sell_custom_under_rsi_1": 37.6,
    "sell_custom_under_rsi_2": 59.6,
    "sell_custom_under_rsi_3": 63.6,
    "sell_trail_down_1": 0.062,
    "sell_trail_down_2": 0.167,
    "sell_trail_down_3": 0.017,
    "sell_trail_profit_max_1": 0.16,
    "sell_trail_profit_max_2": 0.23,
    "sell_trail_profit_max_3": 0.15,
    "sell_trail_profit_min_1": 0.003,
    "sell_trail_profit_min_2": 0.037,
    "sell_trail_profit_min_3": 0.054,
}

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

class abbas6(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.103
    }

    # Stoploss:
    stoploss = -0.072

    # SMAOffset
    high_offset_ema = DecimalParameter(0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    base_nb_candles_buy = IntParameter(2, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(10, 40, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=True)
    high_offset = DecimalParameter(0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)

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
    ewo_low = DecimalParameter(-20.0, -8.0,default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)

    ewo_high_2 = DecimalParameter(-6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=True)

    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = False

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = False

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.002
    }

    buy_signals = {}

    # Custom Trailing Stoploss by Perkmeister
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:

        if (current_profit > 0.3):
            return 0.05
        elif (current_profit > 0.2):
            return 0.04
        elif (current_profit > 0.1):
            return 0.03
        elif (current_profit > 0.05):
            return 0.02
        elif (current_profit > 0.02):
            return 0.01
        return 0.99

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)

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

            #elif (current_profit > 0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_profit_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_profit_under_rsi_diff_1.value):
                #return 'signal_profit_u_e_1'

            #elif (current_profit < -0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_stoploss_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_under_rsi_diff_1.value):
                #return 'signal_stoploss_u_1'

        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, sell_reason: str, current_time: datetime, **kwargs) -> bool:

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
        current_profit = trade.calc_profit_ratio(rate)
        if (sell_reason.startswith('sell signal (') and (current_profit > 0.018)):
            # Reject sell signal when trailing stoplosses
            return False
        return True

    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma_200_dec'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        return informative_1h

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma_200_dec'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < 35) &
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
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                (dataframe['close_1h'].rolling(24).max() < (dataframe['close'] * 1.03 )) # don't buy if there isn't 3% profit to be made
            )
        )

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

        dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell.value)) *self.high_offset_ema.value

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe

def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif
