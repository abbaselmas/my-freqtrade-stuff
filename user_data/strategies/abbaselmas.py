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
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real

abbas_params = {
    "startup_candle_count": 200,
    "hma_50X": 1.149,
    "ema_100X": 0.951,
    "current_profitX": 0.01102
}

# Protection hyperspace params:
protection_params = {
    "cooldown_stop_duration_candles": 0,
    "lowprofit2_lookback_period_candles": 179,
    "lowprofit2_required_profit": 0.018,
    "lowprofit2_stop_duration_candles": 28,
    "lowprofit2_trade_limit": 37,
    "lowprofit_lookback_period_candles": 11,
    "lowprofit_required_profit": 0.037,
    "lowprofit_stop_duration_candles": 115,
    "lowprofit_trade_limit": 49,
    "maxdrawdown_lookback_period_candles": 25,
    "maxdrawdown_max_allowed_drawdown": 0.21,
    "maxdrawdown_stop_duration_candles": 47,
    "maxdrawdown_trade_limit": 8,
    "stoplossguard_lookback_period_candles": 270,
    "stoplossguard_stop_duration_candles": 7,
    "stoplossguard_trade_limit": 9,
}

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 17,
    "ewo_high": 2.182,
    "ewo_high_2": -3.44,
    "ewo_low": -10.26,
    "low_offset": 1.066,
    "low_offset_2": 0.961,
    "rsi_buy": 68,
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 9,
    "high_offset": 1.01,
    "high_offset_2": 1.233,
    "high_offset_ema": 0.931,
    "sell_custom_dec_profit_1": 0.084,
    "sell_custom_dec_profit_2": 0.124,
    "sell_custom_profit_0": 0.048,
    "sell_custom_profit_1": 0.083,
    "sell_custom_profit_2": 0.082,
    "sell_custom_profit_3": 0.107,
    "sell_custom_profit_4": 0.016,
    "sell_custom_rsi_0": 39.351,
    "sell_custom_rsi_1": 37.42,
    "sell_custom_rsi_2": 41.15,
    "sell_custom_rsi_3": 47.89,
    "sell_custom_rsi_4": 50.37,
    "sell_custom_under_profit_1": 0.103,
    "sell_custom_under_profit_2": 0.109,
    "sell_custom_under_profit_3": 0.077,
    "sell_custom_under_rsi_1": 49.8,
    "sell_custom_under_rsi_2": 64.5,
    "sell_custom_under_rsi_3": 56.8,
    "sell_trail_down_1": 0.079,
    "sell_trail_down_2": 0.113,
    "sell_trail_down_3": 0.044,
    "sell_trail_profit_max_1": 0.47,
    "sell_trail_profit_max_2": 0.13,
    "sell_trail_profit_max_3": 0.15,
    "sell_trail_profit_min_1": 0.137,
    "sell_trail_profit_min_2": 0.065,
    "sell_trail_profit_min_3": 0.093,
}

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

class abbaselmas(IStrategy):
    INTERFACE_VERSION = 2

    cooldown_stop_duration_candles = IntParameter(0, 20, default=protection_params['cooldown_stop_duration_candles'], space="protection", optimize=True)

    maxdrawdown_optimize = True
    maxdrawdown_lookback_period_candles = IntParameter(5, 40, default=protection_params['maxdrawdown_lookback_period_candles'], space="protection", optimize=maxdrawdown_optimize)
    maxdrawdown_trade_limit = IntParameter(1, 20, default=protection_params['maxdrawdown_trade_limit'], space="protection", optimize=maxdrawdown_optimize)
    maxdrawdown_stop_duration_candles = IntParameter(10, 60, default=protection_params['maxdrawdown_stop_duration_candles'], space="protection", optimize=maxdrawdown_optimize)
    maxdrawdown_max_allowed_drawdown = DecimalParameter(0.05, 0.40, default=protection_params['maxdrawdown_max_allowed_drawdown'], space="protection", decimals=2, optimize=maxdrawdown_optimize)

    stoplossguard_optimize = True
    stoplossguard_lookback_period_candles = IntParameter(1, 300, default=protection_params['stoplossguard_lookback_period_candles'], space="protection", optimize=stoplossguard_optimize)
    stoplossguard_trade_limit = IntParameter(1, 20, default=protection_params['stoplossguard_trade_limit'], space="protection", optimize=stoplossguard_optimize)
    stoplossguard_stop_duration_candles = IntParameter(1, 10, default=protection_params['stoplossguard_stop_duration_candles'], space="protection", optimize=stoplossguard_optimize)

    lowprofit_optimize = True
    lowprofit_lookback_period_candles = IntParameter(10, 60, default=protection_params['lowprofit_lookback_period_candles'], space="protection", optimize=lowprofit_optimize)
    lowprofit_trade_limit = IntParameter(1, 50, default=protection_params['lowprofit_trade_limit'], space="protection", optimize=lowprofit_optimize)
    lowprofit_stop_duration_candles = IntParameter(10, 200, default=protection_params['lowprofit_stop_duration_candles'], space="protection", optimize=lowprofit_optimize)
    lowprofit_required_profit = DecimalParameter(0.000, 0.050, default=protection_params['lowprofit_required_profit'], space="protection", decimals=3, optimize=lowprofit_optimize)

    lowprofit2_optimize = True
    lowprofit2_lookback_period_candles = IntParameter(10, 300, default=protection_params['lowprofit2_lookback_period_candles'], space="protection", optimize=lowprofit2_optimize)
    lowprofit2_trade_limit = IntParameter(1, 70, default=protection_params['lowprofit2_trade_limit'], space="protection", optimize=lowprofit2_optimize)
    lowprofit2_stop_duration_candles = IntParameter(1, 40, default=protection_params['lowprofit2_stop_duration_candles'], space="protection", optimize=lowprofit2_optimize)
    lowprofit2_required_profit = DecimalParameter(0, 0.030, default=protection_params['lowprofit2_required_profit'], space="protection", decimals=3, optimize=lowprofit2_optimize)

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
        prot.append({
            "method": "LowProfitPairs",
            "lookback_period_candles": self.lowprofit_lookback_period_candles.value,
            "trade_limit": self.lowprofit_trade_limit.value,
            "stop_duration_candles": self.lowprofit_stop_duration_candles.value,
            "required_profit": self.lowprofit_required_profit.value
        })
        prot.append({
            "method": "LowProfitPairs",
            "lookback_period_candles": self.lowprofit2_lookback_period_candles.value,
            "trade_limit": self.lowprofit2_trade_limit.value,
            "stop_duration_candles": self.lowprofit2_stop_duration_candles.value,
            "required_profit": self.lowprofit2_required_profit.value
        })

        return prot

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.085, -0.040, decimals=3, name='stoploss')]

        # Define custom ROI space
        def roi_space() -> List[Dimension]:
            return [
                Integer(10, 120, name='roi_t1'),
                Integer(10, 60, name='roi_t2'),
                Integer(10, 40, name='roi_t3'),
                SKDecimal(0.01, 0.04, decimals=3, name='roi_p1'),
                SKDecimal(0.01, 0.07, decimals=3, name='roi_p2'),
                SKDecimal(0.01, 0.20, decimals=3, name='roi_p3'),
            ]
        # Define custom trailing space
        def trailing_space() -> List[Dimension]:
            return[
                Categorical([True], name='trailing_stop'),
                SKDecimal(0.0001, 0.0010, decimals=5, name='trailing_stop_positive'),
                SKDecimal(0.0060, 0.0180, decimals=4, name='trailing_stop_positive_offset_p1'),
                Categorical([True], name='trailing_only_offset_is_reached'),
            ]

        def 

    # ROI table:
    minimal_roi = {
        "200": 0
    }

    # Stoploss:
    stoploss = -0.078

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.00012
    trailing_stop_positive_offset = 0.01102
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = False

    # SMAOffset
    smaoffset_optimize = True
    high_offset_ema = DecimalParameter(0.90, 1.1, default=sell_params['high_offset_ema'], load=True, space='sell', decimals=3, optimize=smaoffset_optimize)
    base_nb_candles_buy = IntParameter(15, 30, default=buy_params['base_nb_candles_buy'], space='buy', optimize=smaoffset_optimize)
    base_nb_candles_sell = IntParameter(5, 30, default=sell_params['base_nb_candles_sell'], space='sell', optimize=smaoffset_optimize)
    low_offset = DecimalParameter(1.0, 1.1, default=buy_params['low_offset'], space='buy', decimals=3, optimize=smaoffset_optimize)
    low_offset_2 = DecimalParameter(0.94, 0.98, default=buy_params['low_offset_2'], space='buy', decimals=3, optimize=smaoffset_optimize)
    high_offset = DecimalParameter(1.0, 1.1, default=sell_params['high_offset'], space='sell', decimals=3, optimize=smaoffset_optimize)
    high_offset_2 = DecimalParameter(1.2, 1.5, default=sell_params['high_offset_2'], space='sell', decimals=3, optimize=smaoffset_optimize)

    sell_custom_profit_optimize = True
    sell_custom_profit_0 = DecimalParameter(0.010, 0.100, default=sell_params['sell_custom_profit_0'], space='sell', decimals=3, optimize=sell_custom_profit_optimize, load=True)
    sell_custom_profit_1 = DecimalParameter(0.013, 0.100, default=sell_params['sell_custom_profit_1'], space='sell', decimals=3, optimize=sell_custom_profit_optimize, load=True)
    sell_custom_profit_2 = DecimalParameter(0.015, 0.100, default=sell_params['sell_custom_profit_2'], space='sell', decimals=3, optimize=sell_custom_profit_optimize, load=True)
    sell_custom_profit_3 = DecimalParameter(0.009, 0.300, default=sell_params['sell_custom_profit_3'], space='sell', decimals=3, optimize=sell_custom_profit_optimize, load=True)
    sell_custom_profit_4 = DecimalParameter(0.010, 0.600, default=sell_params['sell_custom_profit_4'], space='sell', decimals=3, optimize=sell_custom_profit_optimize, load=True)

    sell_custom_rsi_optimize = True
    sell_custom_rsi_0 = DecimalParameter(30.0, 40.0, default=sell_params['sell_custom_rsi_0'], space='sell', decimals=3, optimize=sell_custom_rsi_optimize, load=True)
    sell_custom_rsi_1 = DecimalParameter(30.0, 50.0, default=sell_params['sell_custom_rsi_1'], space='sell', decimals=2, optimize=sell_custom_rsi_optimize, load=True)
    sell_custom_rsi_2 = DecimalParameter(34.0, 50.0, default=sell_params['sell_custom_rsi_2'], space='sell', decimals=2, optimize=sell_custom_rsi_optimize, load=True)
    sell_custom_rsi_3 = DecimalParameter(38.0, 55.0, default=sell_params['sell_custom_rsi_3'], space='sell', decimals=2, optimize=sell_custom_rsi_optimize, load=True)
    sell_custom_rsi_4 = DecimalParameter(40.0, 58.0, default=sell_params['sell_custom_under_profit_1'], space='sell', decimals=2, optimize=sell_custom_rsi_optimize, load=True)

    sell_custom_under_profit_optimize = True
    sell_custom_under_profit_1 = DecimalParameter(0.070, 0.120, default=sell_params['sell_custom_under_profit_1'], space='sell', decimals=3, optimize=sell_custom_under_profit_optimize, load=True)
    sell_custom_under_profit_2 = DecimalParameter(0.070, 0.120, default=sell_params['sell_custom_under_profit_2'], space='sell', decimals=3, optimize=sell_custom_under_profit_optimize, load=True)
    sell_custom_under_profit_3 = DecimalParameter(0.070, 0.120, default=sell_params['sell_custom_under_profit_3'], space='sell', decimals=3, optimize=sell_custom_under_profit_optimize, load=True)

    sell_custom_under_rsi_optimize = True
    sell_custom_under_rsi_1 = DecimalParameter(36.0, 60.0, default=sell_params['sell_custom_under_rsi_1'], space='sell', decimals=1, optimize=sell_custom_under_rsi_optimize, load=True)
    sell_custom_under_rsi_2 = DecimalParameter(46.0, 66.0, default=sell_params['sell_custom_under_rsi_2'], space='sell', decimals=1, optimize=sell_custom_under_rsi_optimize, load=True)
    sell_custom_under_rsi_3 = DecimalParameter(48.0, 68.0, default=sell_params['sell_custom_under_rsi_3'], space='sell', decimals=1, optimize=sell_custom_under_rsi_optimize, load=True)

    sell_custom_dec_profit_optimize = True
    sell_custom_dec_profit_1 = DecimalParameter(0.001, 0.100, default=sell_params['sell_custom_dec_profit_1'], space='sell', decimals=3, optimize=sell_custom_dec_profit_optimize, load=True)
    sell_custom_dec_profit_2 = DecimalParameter(0.100, 0.200, default=sell_params['sell_custom_dec_profit_2'], space='sell', decimals=3, optimize=sell_custom_dec_profit_optimize, load=True)

    sell_trail_optimize = True
    sell_trail_profit_min_1 = DecimalParameter(0.001, 0.25, default=sell_params['sell_trail_profit_min_1'], space='sell', decimals=3, optimize=sell_trail_optimize, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.03, 0.5, default=sell_params['sell_trail_profit_max_1'], space='sell', decimals=2, optimize=sell_trail_optimize, load=True)
    sell_trail_down_1 = DecimalParameter(0.040, 0.100, default=sell_params['sell_trail_down_1'], space='sell', decimals=3, optimize=sell_trail_optimize, load=True)
    sell_trail_profit_min_2 = DecimalParameter(0.004, 0.100, default=sell_params['sell_trail_profit_min_2'], space='sell', decimals=3, optimize=sell_trail_optimize, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=sell_params['sell_trail_profit_max_2'], space='sell', decimals=2, optimize=sell_trail_optimize, load=True)
    sell_trail_down_2 = DecimalParameter(0.040, 0.120, default=sell_params['sell_trail_down_2'], space='sell', decimals=3, optimize=sell_trail_optimize, load=True)
    sell_trail_profit_min_3 = DecimalParameter(0.006, 0.100, default=sell_params['sell_trail_profit_min_3'], space='sell', decimals=3, optimize=sell_trail_optimize, load=True)
    sell_trail_profit_max_3 = DecimalParameter(0.08, 0.16, default=sell_params['sell_trail_profit_max_3'], space='sell', decimals=2, optimize=sell_trail_optimize, load=True)
    sell_trail_down_3 = DecimalParameter(0.03, 0.05, default=sell_params['sell_trail_down_3'], space='sell', decimals=3, optimize=sell_trail_optimize, load=True)

    # Protection
    protection_optimize = True
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-12.0, -8.0,default=buy_params['ewo_low'], space='buy', decimals=2, optimize=protection_optimize)
    ewo_high = DecimalParameter(1.0, 2.2, default=buy_params['ewo_high'], space='buy', decimals=3, optimize=protection_optimize)
    ewo_high_2 = DecimalParameter(-4.0, -2.0, default=buy_params['ewo_high_2'], space='buy', decimals=2, optimize=protection_optimize)
    rsi_buy = IntParameter(55, 85, default=buy_params['rsi_buy'], space='buy', optimize=protection_optimize)

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'


    process_only_new_candles = True
    startup_candle_count = IntParameter(100, 300, default=abbas_params['startup_candle_count'], space='abbas', optimize=True)

    plot_config = {
        'main_plot': {
            "bb_upperband28": {"color": "#bc281d","type": "line"},
            'bb_midband28': {'color': "orange", "type": "line"},
            "bb_lowerband28": {"color": "#792bbb","type": "line"}
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'yellow'},
            }
        }
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.002
    }

    buy_signals = {}

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

        return None

    hma_50X = DecimalParameter(0, 2, default=abbas_params['hma_50X'], space='abbas', decimals=3, optimize=True)
    ema_100X = DecimalParameter(0, 2, default=abbas_params['ema_100X'], space='abbas', decimals=3, optimize=True)
    current_profitX =  DecimalParameter(0, 0.02, default=abbas_params['current_profitX'], space='abbas', decimals=3, optimize=True)
    

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, sell_reason: str, current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*new_params['hma_50X'] > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*new_params[ema_100X]):  # *1.2
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
        if (sell_reason.startswith('sell signal (') and (current_profit > abbas_params['current_profitX'])):
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

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2.8)
        dataframe['bb_lowerband28'] = bollinger2['lower']
        dataframe['bb_middleband28'] = bollinger2['mid']
        dataframe['bb_upperband28'] = bollinger2['upper']

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

# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma1_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif

# PMAX
def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx

# smoothed Heiken Ashi
def HA(dataframe, smoothing=None):
    df = dataframe.copy()

    df['HA_Close']=(df['open'] + df['high'] + df['low'] + df['close'])/4

    df.reset_index(inplace=True)

    ha_open = [ (df['open'][0] + df['close'][0]) / 2 ]
    [ ha_open.append((ha_open[i] + df['HA_Close'].values[i]) / 2) for i in range(0, len(df)-1) ]
    df['HA_Open'] = ha_open

    df.set_index('index', inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','high']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','low']].min(axis=1)

    if smoothing is not None:
        sml = abs(int(smoothing))
        if sml > 0:
            df['Smooth_HA_O']=ta.EMA(df['HA_Open'], sml)
            df['Smooth_HA_C']=ta.EMA(df['HA_Close'], sml)
            df['Smooth_HA_H']=ta.EMA(df['HA_High'], sml)
            df['Smooth_HA_L']=ta.EMA(df['HA_Low'], sml)
            
    return df

def pump_warning(dataframe, perc=15):
    df = dataframe.copy()    
    df["change"] = df["high"] - df["low"]
    df["test1"] = (df["close"] > df["open"])
    df["test2"] = ((df["change"]/df["low"]) > (perc/100))
    df["result"] = (df["test1"] & df["test2"]).astype('int')
    return df['result']

def tv_wma(dataframe, length = 9, field="close") -> DataFrame:
    """
    Source: Tradingview "Moving Average Weighted"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : WMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_wma'
    """

    norm = 0
    sum = 0

    for i in range(1, length - 1):
        weight = (length - i) * length
        norm = norm + weight
        sum = sum + dataframe[field].shift(i) * weight

    dataframe["tv_wma"] = (sum / norm) if norm > 0 else 0
    return dataframe["tv_wma"]

def tv_hma(dataframe, length = 9, field="close") -> DataFrame:
    """
    Source: Tradingview "Hull Moving Average"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : HMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_hma'
    """

    dataframe["h"] = 2 * tv_wma(dataframe, math.floor(length / 2), field) - tv_wma(dataframe, length, field)

    dataframe["tv_hma"] = tv_wma(dataframe, math.floor(math.sqrt(length)), "h")
    # dataframe.drop("h", inplace=True, axis=1)

    return dataframe["tv_hma"]

"""
TA Indicators
"""

def zema(dataframe, period, field='close'):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/overlap_studies.py#L79
    Modified slightly to use ta.EMA instead of technical ema
    """
    df = dataframe.copy()

    df['ema1'] = ta.EMA(df[field], timeperiod=period)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=period)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']

    return df['zema']

def RMI(dataframe, *, length=20, mom=5):
    """
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L912
    """
    df = dataframe.copy()

    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)

    df.fillna(0, inplace=True)

    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)

    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))

    return df["RMI"]

def mastreak(dataframe: DataFrame, period: int = 4, field='close') -> Series:
    """
    MA Streak
    Port of: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
    """    
    df = dataframe.copy()

    avgval = zema(df, period, field)

    arr = np.diff(avgval)
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                    -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))

    res = same_length(df['close'], streak)

    return res

def pcc(dataframe: DataFrame, period: int = 20, mult: int = 2):
    """
    Percent Change Channel
    PCC is like KC unless it uses percentage changes in price to set channel distance.
    https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
    """
    df = dataframe.copy()

    df['previous_close'] = df['close'].shift()

    df['close_change'] = (df['close'] - df['previous_close']) / df['previous_close'] * 100
    df['high_change'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_change'] = (df['low'] - df['close']) / df['close'] * 100

    df['delta'] = df['high_change'] - df['low_change']

    mid = zema(df, period, 'close_change')
    rangema = zema(df, period, 'delta')

    upper = mid + rangema * mult
    lower = mid - rangema * mult

    return upper, rangema, lower

def SSLChannels(dataframe, length=10, mode='sma'):
    """
    Source: https://www.tradingview.com/script/xzIoaIJC-SSL-channel/
    Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/indicators.py#L1025
    Usage:
        dataframe['sslDown'], dataframe['sslUp'] = SSLChannels(dataframe, 10)
    """
    if mode not in ('sma'):
        raise ValueError(f"Mode {mode} not supported yet")

    df = dataframe.copy()

    if mode == 'sma':
        df['smaHigh'] = df['high'].rolling(length).mean()
        df['smaLow'] = df['low'].rolling(length).mean()

    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1,
                         np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()

    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

def SSLChannels_ATR(dataframe, length=7):
    """
    SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
    Credit to @JimmyNixx for python
    """
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

def WaveTrend(dataframe, chlen=10, avg=21, smalen=4):
    """
    WaveTrend Ocillator by LazyBear
    https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
    """
    df = dataframe.copy()

    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['esa'] = ta.EMA(df['hlc3'], timeperiod=chlen)
    df['d'] = ta.EMA((df['hlc3'] - df['esa']).abs(), timeperiod=chlen)
    df['ci'] = (df['hlc3'] - df['esa']) / (0.015 * df['d'])
    df['tci'] = ta.EMA(df['ci'], timeperiod=avg)

    df['wt1'] = df['tci']
    df['wt2'] = ta.SMA(df['wt1'], timeperiod=smalen)
    df['wt1-wt2'] = df['wt1'] - df['wt2']

    return df['wt1'], df['wt2']

def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b*b*b
    c2 = 3*b*b+3*b*b*b
    c3 = -6*b*b-3*b-3*b*b*b
    c4 = 1+3*b+b*b*b+3*b*b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']


def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc
