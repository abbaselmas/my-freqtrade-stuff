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
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real

# Protection hyperspace params:
protection_params = {
    "cooldown_stop_duration_candles": 0,
    "lowprofit2_lookback_period_candles": 91,
    "lowprofit2_required_profit": 0.019,
    "lowprofit2_stop_duration_candles": 21,
    "lowprofit2_trade_limit": 16,
    "lowprofit_lookback_period_candles": 28,
    "lowprofit_required_profit": 0.044,
    "lowprofit_stop_duration_candles": 127,
    "lowprofit_trade_limit": 30,
    "maxdrawdown_lookback_period_candles": 21,
    "maxdrawdown_max_allowed_drawdown": -0.18,
    "maxdrawdown_stop_duration_candles": 27,
    "maxdrawdown_trade_limit": 9,
    "stoplossguard_lookback_period_candles": 82,
    "stoplossguard_stop_duration_candles": 8,
    "stoplossguard_trade_limit": 10,
}

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 17,
    "ewo_high": 1.964,
    "ewo_high_2": -2.91,
    "ewo_low": -8.27,
    "low_offset": 1.068,
    "low_offset_2": 0.941,
    "rsi_buy": 84,
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 15,
    "high_offset": 1.064,
    "high_offset_2": 1.228,
    "high_offset_ema": 1.087,
    "sell_custom_dec_profit_1": 0.077,
    "sell_custom_dec_profit_2": 0.152,
    "sell_custom_profit_0": 0.01,
    "sell_custom_profit_1": 0.063,
    "sell_custom_profit_2": 0.025,
    "sell_custom_profit_3": 0.182,
    "sell_custom_profit_4": 0.07,
    "sell_custom_rsi_0": 38.476,
    "sell_custom_rsi_1": 41.21,
    "sell_custom_rsi_2": 40.58,
    "sell_custom_rsi_3": 45.81,
    "sell_custom_rsi_4": 55.28,
    "sell_custom_under_profit_1": 0.087,
    "sell_custom_under_profit_2": 0.079,
    "sell_custom_under_profit_3": 0.075,
    "sell_custom_under_rsi_1": 41.3,
    "sell_custom_under_rsi_2": 53.4,
    "sell_custom_under_rsi_3": 64.2,
    "sell_trail_down_1": 0.065,
    "sell_trail_down_2": 0.05,
    "sell_trail_down_3": 0.04,
    "sell_trail_profit_max_1": 0.3,
    "sell_trail_profit_max_2": 0.1,
    "sell_trail_profit_max_3": 0.13,
    "sell_trail_profit_min_1": 0.204,
    "sell_trail_profit_min_2": 0.021,
    "sell_trail_profit_min_3": 0.088,
}

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

class abbas(IStrategy):
    INTERFACE_VERSION = 2

    cooldown_stop_duration_candles = IntParameter(0, 48, default=0, space="protection", optimize=False)

    maxdrawdown_optimize = False
    maxdrawdown_lookback_period_candles = IntParameter(5, 30, default=21, space="protection", optimize=maxdrawdown_optimize)
    maxdrawdown_trade_limit = IntParameter(1, 20, default=9, space="protection", optimize=maxdrawdown_optimize)
    maxdrawdown_stop_duration_candles = IntParameter(10, 70, default=27, space="protection", optimize=maxdrawdown_optimize)
    maxdrawdown_max_allowed_drawdown = DecimalParameter(-0.25, 0.05, default=-0.18, space="protection", decimals=2, optimize=maxdrawdown_optimize)

    stoplossguard_optimize = False
    stoplossguard_lookback_period_candles = IntParameter(1, 300, default=82, space="protection", optimize=stoplossguard_optimize)
    stoplossguard_trade_limit = IntParameter(1, 20, default=10, space="protection", optimize=stoplossguard_optimize)
    stoplossguard_stop_duration_candles = IntParameter(1, 20, default=8, space="protection", optimize=stoplossguard_optimize)

    lowprofit_optimize = False
    lowprofit_lookback_period_candles = IntParameter(2, 60, default=28, space="protection", optimize=lowprofit_optimize)
    lowprofit_trade_limit = IntParameter(1, 50, default=30, space="protection", optimize=lowprofit_optimize)
    lowprofit_stop_duration_candles = IntParameter(1, 200, default=127, space="protection", optimize=lowprofit_optimize)
    lowprofit_required_profit = DecimalParameter(0, 0.05, default=0.0044, space="protection", decimals=3, optimize=lowprofit_optimize)

    lowprofit2_optimize = False
    lowprofit2_lookback_period_candles = IntParameter(1, 300, default=91, space="protection", optimize=lowprofit2_optimize)
    lowprofit2_trade_limit = IntParameter(1, 70, default=16, space="protection", optimize=lowprofit2_optimize)
    lowprofit2_stop_duration_candles = IntParameter(1, 25, default=21, space="protection", optimize=lowprofit2_optimize)
    lowprofit2_required_profit = DecimalParameter(0, 0.02, default=0.0019, space="protection", decimals=3, optimize=lowprofit2_optimize)

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
            return [SKDecimal(-0.075, -0.050, decimals=3, name='stoploss')]

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
                SKDecimal(0.00010, 0.00011, decimals=5, name='trailing_stop_positive'),
                SKDecimal(0.0130, 0.0173, decimals=4, name='trailing_stop_positive_offset_p1'),
                Categorical([True], name='trailing_only_offset_is_reached'),
            ]

    # ROI table:
    minimal_roi = {
        "0": 0.098,
        "32": 0.058,
        "92": 0.039,
        "210": 0
    }

    # Stoploss:
    stoploss = -0.072

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0001
    trailing_stop_positive_offset = 0.0152
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = False

    # SMAOffset
    smaoffset_optimize = True
    high_offset_ema = DecimalParameter(0.90, 1.1, default=1.012, load=True, space='sell', decimals=3, optimize=smaoffset_optimize)
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
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200

    plot_config = {
        'main_plot': {
            'bb_upperband': {'color': 'green'},
            'bb_midband': {'color': 'orange'},
            'bb_lowerband': {'color': 'red'},
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
