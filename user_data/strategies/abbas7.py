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
    "base_nb_candles_buy": 28,
    "ewo_high": 1.476,
    "ewo_high_2": -2.06,
    "ewo_low": -11.99,
    "low_offset": 1.034,
    "low_offset_2": 0.974,
    "rsi_buy": 71,
}

class abbas7(IStrategy):
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
            return [SKDecimal(-0.090, -0.030, decimals=3, name='stoploss')]

        # Define custom trailing space
        def trailing_space() -> List[Dimension]:
            return[
                Categorical([True], name='trailing_stop'),
                SKDecimal(0.00010, 0.00040, decimals=5, name='trailing_stop_positive'),
                SKDecimal(0.0080, 0.0180, decimals=4, name='trailing_stop_positive_offset_p1'),
                Categorical([True], name='trailing_only_offset_is_reached'),
            ]

    # ROI table:
    minimal_roi = {
        "200": 0
    }

    # Stoploss:
    stoploss = -0.078  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0001
    trailing_stop_positive_offset = 0.0082
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = False
    sell_profit_only = True
    sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = False

    # SMAOffset
    smaoffset_optimize = True
    base_nb_candles_buy = IntParameter(15, 30, default=buy_params['base_nb_candles_buy'], space='buy', optimize=smaoffset_optimize)
    low_offset = DecimalParameter(1.0, 1.1, default=buy_params['low_offset'], space='buy', decimals=3, optimize=smaoffset_optimize)
    low_offset_2 = DecimalParameter(0.94, 0.98, default=buy_params['low_offset_2'], space='buy', decimals=3, optimize=smaoffset_optimize)

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
    startup_candle_count = 200

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

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, sell_reason: str, current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

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

        # Bollinger bands
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
                (dataframe['volume'] > 0)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi'] < 25)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                (dataframe['close_1h'].rolling(24).max() < (dataframe['close'] * 1.03 )) # don't buy if there isn't 3% profit within 24h
            )
        )

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma1_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif
