from freqtrade.strategy import (IStrategy, informative)
from typing import Dict, List
from pandas import DataFrame, Series
import pandas_ta as pta
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 15,
    "buy_V_bb_width": 0.01,
    "buy_V_cti": -0.6,
    "buy_V_mfi": 30,
    "buy_V_r14": -60,
    "buy_clucha_bbdelta_close": 0.041,
    "buy_clucha_bbdelta_tail": 0.72,
    "buy_clucha_closedelta_close": 0.01,
    "buy_clucha_rocr_1h": 0.07,
    "buy_vwap_closedelta": 12.63,
    "buy_vwap_cti": -0.87,
    "buy_vwap_width": 0.16,
    "ewo_high": 8.32,
    "ewo_high_2": -4.44,
    "ewo_low": -6.42,
    "fast_ewo": 9,
    "low_offset": 1.18,
    "low_offset_2": 0.93,
    "rsi_buy": 67,
    "rsi_ewo2": 18,
    "rsi_fast_ewo1": 56,
    "slow_ewo": 198
}
# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 15,
    "high_offset": 1.11
}

def EWO(dataframe, ema_length=5, ema2_length=35):
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe["low"] * 100

# VWAP bands
def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df["vwap"] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df["vwap"].rolling(window=window_size).std()
    df["vwap_low"] = df["vwap"] - (rolling_std * num_of_std)
    df["vwap_high"] = df["vwap"] + (rolling_std * num_of_std)
    return df["vwap_low"], df["vwap"], df["vwap_high"]

def ha_typical_price(bars):
    res = (bars["ha_high"] + bars["ha_low"] + bars["ha_close"]) / 3.
    return Series(index=bars.index, data=res)

class abbas8(IStrategy):
    def version(self) -> str:
        return "v9.8.5"
    INTERFACE_VERSION = 3
    class HyperOpt:
        # Define custom ROI space
        def roi_space() -> List[Dimension]:
            return [
                Integer(  6, 60, name="roi_t1"),
                Integer( 60, 120, name="roi_t2"),
                Integer(120, 200, name="roi_t3")
            ]
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            roi_table = {}
            roi_table[params["roi_t1"]] = 0
            roi_table[params["roi_t2"]] = -0.015
            roi_table[params["roi_t3"]] = -0.030
            return roi_table
    timeframe = "5m"
    info_timeframes = ["15m", "30m", "1h"]
    minimal_roi = {
        "99": 0,
        "140": -0.015,
        "232": -0.03
    }
    order_time_in_force = {
        "entry": "gtc",
        "exit": "ioc"
    }
    slippage_protection = {
        "retries": 3,
        "max_slippage": -0.002
    }
    stoploss = -0.067
    trailing_stop = True
    trailing_stop_positive = 0.0003
    trailing_stop_positive_offset = 0.0146
    trailing_only_offset_is_reached = True
    use_exit_signal = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True
    startup_candle_count = 449

    smaoffset_optimize = False
    base_nb_candles_buy = IntParameter(10, 50, default=buy_params["base_nb_candles_buy"], space="buy", optimize=smaoffset_optimize)
    base_nb_candles_sell = IntParameter(10, 50, default=sell_params["base_nb_candles_sell"], space="sell", optimize=smaoffset_optimize)
    low_offset = DecimalParameter(0.8, 1.2, default=buy_params["low_offset"], space="buy", decimals=2, optimize=smaoffset_optimize)
    low_offset_2 = DecimalParameter(0.8, 1.2, default=buy_params["low_offset_2"], space="buy", decimals=2, optimize=smaoffset_optimize)
    high_offset = DecimalParameter(0.8, 1.2, default=sell_params["high_offset"], space="sell", decimals=2, optimize=smaoffset_optimize)

    ewo_optimize = False
    fast_ewo = IntParameter(5,40, default=buy_params["fast_ewo"], space="buy", optimize=ewo_optimize)
    slow_ewo = IntParameter(80,250, default=buy_params["slow_ewo"], space="buy", optimize=ewo_optimize)
    rsi_fast_ewo1 = IntParameter(20, 60, default=buy_params["rsi_fast_ewo1"], space="buy", optimize=ewo_optimize)
    rsi_ewo2 = IntParameter(10, 40, default=buy_params["rsi_ewo2"], space="buy", optimize=ewo_optimize)

    protection_optimize = False
    ewo_low = DecimalParameter(-20.0, -4.0, default=buy_params["ewo_low"], space="buy", decimals=2, optimize=protection_optimize)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params["ewo_high"], space="buy", decimals=2, optimize=protection_optimize)
    ewo_high_2 = DecimalParameter(-6.0, 12.0, default=buy_params["ewo_high_2"], space="buy", decimals=2, optimize=protection_optimize)
    rsi_buy = IntParameter(50, 85, default=buy_params["rsi_buy"], space="buy", optimize=protection_optimize)

    is_optimize_clucha = False
    buy_clucha_bbdelta_close    = DecimalParameter(0.001, 0.042,  default=0.034, space="buy", decimals=3, optimize = is_optimize_clucha)
    buy_clucha_bbdelta_tail     = DecimalParameter(0.70,   1.10,  default=0.95,  space="buy", decimals=2, optimize = is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.001,  0.025, default=0.02,  space="buy", decimals=3, optimize = is_optimize_clucha)
    buy_clucha_rocr_1h          = DecimalParameter(0.01,   1.00,  default=0.13,  space="buy", decimals=2, optimize = is_optimize_clucha)

    is_optimize_vwap = False
    buy_vwap_width      = DecimalParameter(0.05, 10.0,   default=0.80,  space="buy", decimals=2, optimize = is_optimize_vwap)
    buy_vwap_closedelta = DecimalParameter(10.0, 30.0,   default=15.0,  space="buy", decimals=2, optimize = is_optimize_vwap)
    buy_vwap_cti        = DecimalParameter(-0.90, -0.00, default=-0.60, space="buy", decimals=2, optimize = is_optimize_vwap)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for info_timeframe in self.info_timeframes:
            informative_pairs.extend([(pair, info_timeframe) for pair in pairs])
        return informative_pairs
    
    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")

        # Heikin Ashi
        inf_heikinashi = qtpylib.heikinashi(informative_1h)
        informative_1h["ha_close"] = inf_heikinashi["close"]
        informative_1h["rocr"] = ta.ROCR(informative_1h["ha_close"], timeperiod=168)

        return informative_1h
    
    def informative_30m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_30m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="30m")    
        return informative_30m
    
    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="15m")
        return informative_15m

    def base_tf_5m_indicators(self, metadata: dict, dataframe: DataFrame) -> DataFrame:
        dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_buy.value))
        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_sell.value))
        dataframe["ewo"] = EWO(dataframe, int(self.fast_ewo.value), int(self.slow_ewo.value))
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)
        dataframe["volume_mean_slow"] = dataframe["volume"].rolling(window=30).mean()

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["ha_high"] = heikinashi["high"]
        dataframe["ha_low"] = heikinashi["low"]
        ## BB 40
        bollinger2_40 = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=40, stds=2)
        dataframe["bb_lowerband2_40"] = bollinger2_40["lower"]
        dataframe["bb_middleband2_40"] = bollinger2_40["mid"]
        # RSI
        dataframe["rsi_84"] = ta.RSI(dataframe, timeperiod=84)
        dataframe["rsi_112"] = ta.RSI(dataframe, timeperiod=112)
        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe["vwap_upperband"] = vwap_high
        dataframe["vwap_middleband"] = vwap
        dataframe["vwap_lowerband"] = vwap_low
        dataframe["vwap_width"] = ( (dataframe["vwap_upperband"] - dataframe["vwap_lowerband"]) / dataframe["vwap_middleband"] ) * 100
        # ClucHA
        dataframe["bb_delta_cluc"] = (dataframe["bb_middleband2_40"] - dataframe["bb_lowerband2_40"]).abs()
        dataframe["ha_closedelta"] = (dataframe["ha_close"] - dataframe["ha_close"].shift()).abs()
        dataframe["tail"] = (dataframe["ha_close"] - dataframe["ha_low"]).abs()
        dataframe["rocr"] = ta.ROCR(dataframe["ha_close"], timeperiod=28)
        # CTI
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)
        # BinH
        dataframe["closedelta"] = (dataframe["close"] - dataframe["close"].shift()).abs()
        return dataframe
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        val = super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs)        

        if val:
            if self.trailing_sell_order_enabled:
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if(len(dataframe) >= 1):
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_sell= self.trailing_sell(pair)
                    trailing_sell_offset = self.trailing_sell_offset(dataframe, pair, current_price)

                    if trailing_sell['allow_sell_trailing']:
                        if (not trailing_sell['trailing_sell_order_started'] and (last_candle['sell'] != 0)):
                            trailing_sell['trailing_sell_order_started'] = True
                            trailing_sell['trailing_sell_order_downlimit'] = last_candle['close']
                            trailing_sell['start_trailing_sell_price'] = last_candle['close']
                            trailing_sell['sell_tag'] = last_candle['sell_tag']
                            trailing_sell['start_trailing_time'] = datetime.now(timezone.utc)
                            trailing_sell['offset'] = 0
                            
                            self.trailing_sell_info(pair, current_price)

                        elif trailing_sell['trailing_sell_order_started']:
                            if trailing_sell_offset == 'forcesell':
                                # sell in custom conditions
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_sell_profit_ratio(pair, current_price)) * 100)
                                self.trailing_sell_info(pair, current_price)

                            elif trailing_sell_offset is None:
                                # stop trailing sell custom conditions
                                self.trailing_sell(pair, reinit=True)

                            elif current_price > trailing_sell['trailing_sell_order_downlimit']:
                                # update downlimit
                                old_downlimit = trailing_sell["trailing_sell_order_downlimit"]
                                self.custom_info_trail_sell[pair]['trailing_sell']['trailing_sell_order_downlimit'] = max(current_price * (1 - trailing_sell_offset), self.custom_info_trail_sell[pair]['trailing_sell']['trailing_sell_order_downlimit'])
                                self.custom_info_trail_sell[pair]['trailing_sell']['offset'] = trailing_sell_offset
                                self.trailing_sell_info(pair, current_price)

                            elif current_price > (trailing_sell['start_trailing_sell_price'] * (1 - self.trailing_sell_max_sell)):
                                # sell! current price < downlimit && higher than starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_sell_profit_ratio(pair, current_price)) * 100)
                                self.trailing_sell_info(pair, current_price)

                            elif current_price < (trailing_sell['start_trailing_sell_price'] * (1 - self.trailing_sell_max_stop)):
                                # stop trailing, sell fast, price too low
                                val = True                                
                                self.trailing_sell_info(pair, current_price)
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_sell_info(pair, current_price)

                    else:
                        logger.info(f"Wait for next sell signal for {pair}")

                if (val == True):
                    self.trailing_sell_info(pair, rate)
                    self.trailing_sell(pair, reinit=True)

        #if (sell_reason != 'sell_signal') | (sell_reason!='force_sell'):
        if (sell_reason != 'sell_signal'):
            val = True

        return val
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe       = self.base_tf_5m_indicators(metadata, dataframe)
        informative_1h  = self.informative_1h_indicators(dataframe, metadata)
        dataframe       = merge_informative_pair(dataframe, informative_1h, self.timeframe, "1h", ffill=True)
        informative_30m = self.informative_30m_indicators(dataframe, metadata)
        dataframe       = merge_informative_pair(dataframe, informative_30m, self.timeframe, "30m", ffill=True)
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe       = merge_informative_pair(dataframe, informative_15m, self.timeframe, "15m", ffill=True)
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_tag"] = ""
        dataframe.loc[
            (
                (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
                (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
                (dataframe["ewo"] > self.ewo_high.value) &
                (dataframe["rsi"] < self.rsi_buy.value) &
                (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
            ),
            ["enter_long", "enter_tag"]] = (1, "ewo1")
        dataframe.loc[
            (
                (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
                (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
                (dataframe["ewo"] < self.ewo_low.value) &
                (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
            ),
            ["enter_long", "enter_tag"]] = (1, "ewolow")
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["vwap_lowerband"]) &
                (dataframe["vwap_width"] > self.buy_vwap_width.value) &
                (dataframe["closedelta"] > dataframe["close"] * self.buy_vwap_closedelta.value / 1000 ) &
                (dataframe["cti"] < self.buy_vwap_cti.value) &
                (dataframe["rsi_84"] < 60) &
                (dataframe["rsi_112"] < 60)
            ),
            ["enter_long", "enter_tag"]] = (1, "vwap")
        dataframe.loc[
            (
                (dataframe["rocr_1h"] > self.buy_clucha_rocr_1h.value ) &
                (dataframe["bb_lowerband2_40"].shift() > 0) &
                (dataframe["bb_delta_cluc"] > dataframe["ha_close"] * self.buy_clucha_bbdelta_close.value) &
                (dataframe["ha_closedelta"] > dataframe["ha_close"] * self.buy_clucha_closedelta_close.value) &
                (dataframe["tail"] < dataframe["bb_delta_cluc"] * self.buy_clucha_bbdelta_tail.value) &
                (dataframe["ha_close"] < dataframe["bb_lowerband2_40"].shift()) &
                (dataframe["ha_close"] < dataframe["ha_close"].shift()) &
                (dataframe["rsi_84"] < 60) &
                (dataframe["rsi_112"] < 60)
            ),
            ["enter_long", "enter_tag"]] = (1, "clucha")

        # trailing buy sell part
        if self.trailing_buy_order_enabled: 
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_buy = self.trailing_buy(metadata['pair'])
            if (last_candle['enter_long'] == 1):
                if not trailing_buy['trailing_buy_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                    if not open_trades:
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_buy['allow_trailing'] = True
                        initial_buy_tag = last_candle['entry_tag'] if 'enter_tag' in last_candle else 'entry signal'
                        dataframe.loc[:, 'enter_tag'] = f"{initial_buy_tag} (start trail price {last_candle['close']})"                        
            else:
                if (trailing_buy['trailing_buy_order_started'] == True):
                    dataframe.loc[:,'enter_long'] = 1
                    dataframe.loc[:, 'enter_tag'] = trailing_buy['enter_tag']

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.trailing_buy_order_enabled and self.abort_trailing_when_sell_signal_triggered:
            last_candle = dataframe.iloc[-1].squeeze()
            if (last_candle['exit_long'] != 0):
                trailing_buy = self.trailing_buy(metadata['pair'])
                if trailing_buy['trailing_buy_order_started']:
                    self.trailing_buy(metadata['pair'], reinit=True)        

        if self.trailing_sell_order_enabled:
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_sell = self.trailing_sell(metadata['pair'])
            if (last_candle['exit_long'] != 0):
                if not trailing_sell['trailing_sell_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                    #if not open_trades: 
                    if open_trades:
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_sell['allow_sell_trailing'] = True
                        initial_sell_tag = last_candle['exit_tag'] if 'exit_tag' in last_candle else 'exit signal'
                        dataframe.loc[:, 'exit_tag'] = f"{initial_sell_tag} (start trail price {last_candle['close']})"
            else:
                if (trailing_sell['trailing_sell_order_started'] == True):
                    dataframe.loc[:,'exit_long'] = 1
                    dataframe.loc[:, 'exit_tag'] = trailing_sell['exit_tag']
        return dataframe
    
    # trailing buy sell part
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
        
        if val:
            if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if(len(dataframe) >= 1):
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_buy = self.trailing_buy(pair)
                    trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)

                    if trailing_buy['allow_trailing']:
                        if (not trailing_buy['trailing_buy_order_started'] and (last_candle['buy'] == 1)):
                            # start trailing buy
                            
                            trailing_buy['trailing_buy_order_started'] = True
                            trailing_buy['trailing_buy_order_uplimit'] = last_candle['close']
                            trailing_buy['start_trailing_price'] = last_candle['close']
                            trailing_buy['entry_tag'] = last_candle['entry_tag']
                            trailing_buy['start_trailing_time'] = datetime.now(timezone.utc)
                            trailing_buy['offset'] = 0
                            
                            self.trailing_buy_info(pair, current_price)

                        elif trailing_buy['trailing_buy_order_started']:
                            if trailing_buy_offset == 'forcebuy':
                                # buy in custom conditions
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_buy_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)

                            elif trailing_buy_offset is None:
                                # stop trailing buy custom conditions
                                self.trailing_buy(pair, reinit=True)

                            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                # update uplimit
                                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                            elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                                # buy ! current price > uplimit && lower thant starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_buy_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)

                            elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                                # stop trailing buy because price is too high
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_buy_info(pair, current_price)

                    else:
                        logger.info(f"Wait for next buy signal for {pair}")

                if (val == True):
                    self.trailing_buy_info(pair, rate)
                    self.trailing_buy(pair, reinit=True)
        
        return val

    process_only_new_candles = True

    custom_info_trail_buy = dict()
    custom_info_trail_sell = dict()    

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_sell_order_enabled = True    
    trailing_expire_seconds = 1800      #NOTE 5m timeframe
    #trailing_expire_seconds = 1800/5    #NOTE 1m timeframe
    #trailing_expire_seconds = 1800*3    #NOTE 15m timeframe

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = True
    trailing_sell_uptrend_enabled = True    
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.02  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.000  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    trailing_sell_max_stop = 0.02   # stop trailing sell if current_price < starting_price * (1+trailing_buy_max_stop)
    trailing_sell_max_sell = 0.000  # sell if price between downlimit (=max of serie (current_price * (1 + trailing_sell_offset())) and (start_price * 1+trailing_sell_max_sell))

    abort_trailing_when_sell_signal_triggered = False


    init_trailing_buy_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,  
        'start_trailing_price': 0,
        'entry_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_trailing': False,
    }

    init_trailing_sell_dict = {
        'trailing_sell_order_started': False,
        'trailing_sell_order_downlimit': 0,        
        'start_trailing_sell_price': 0,
        'exit_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_sell_trailing': False,
    }    

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if (reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]):
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_buy_dict.copy()
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_sell(self, pair, reinit=False):
        # returns trailing sell info for pair (init if necessary)
        if not pair in self.custom_info_trail_sell:
            self.custom_info_trail_sell[pair] = dict()
        if (reinit or not 'trailing_sell' in self.custom_info_trail_sell[pair]):
            self.custom_info_trail_sell[pair]['trailing_sell'] = self.init_trailing_sell_dict.copy()
        return self.custom_info_trail_sell[pair]['trailing_sell']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0

    def trailing_sell_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_sell = self.trailing_sell(pair)

        duration = 0
        try:
            duration = (current_time - trailing_sell['start_trailing_time'])
        except TypeError:
            duration = 0

    def current_trailing_buy_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def current_trailing_sell_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_sell = self.trailing_sell(pair)
        if trailing_sell['trailing_sell_order_started']:
            return (current_price - trailing_sell['start_trailing_sell_price'])/ trailing_sell['start_trailing_sell_price']
            #return 0-((trailing_sell['start_trailing_sell_price'] - current_price) / trailing_sell['start_trailing_sell_price'])
        else:
            return 0

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_buy_profit_ratio(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt  = (last_candle['perc_norm']).round(5)
        default_offset = 0.0045 * (1 + adapt)        #NOTE: default_offset 0.0045 <--> 0.009
        

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if ((current_trailing_profit_ratio > 0) and (last_candle['entry_long'] == 1)):
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    def trailing_sell_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_sell_profit_ratio = self.current_trailing_sell_profit_ratio(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt  = (last_candle['perc_norm']).round(5)
        default_offset = 0.003 * (1 + adapt)        #NOTE: default_offset 0.003 <--> 0.006
        
        trailing_sell  = self.trailing_sell(pair)
        if not trailing_sell['trailing_sell_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration =  current_time - trailing_sell['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if ((current_trailing_sell_profit_ratio > 0) and (last_candle['sell'] != 0)):
                # more than 1h, price over first signal, sell signal still active -> sell
                return 'forcesell'
            else:
                # wait for next signal
                return None
        elif (self.trailing_sell_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_sell_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is falling, sell 
            return 'forcesell'

        if current_trailing_sell_profit_ratio > 0:
            # current price is lower than initial price
            return default_offset

        trailing_sell_offset = {
            # 0.06: 0.02,
            # 0.03: 0.01,
            0.1: default_offset,
        }

        for key in trailing_sell_offset:
            if current_trailing_sell_profit_ratio < key:
                return trailing_sell_offset[key]

        return default_offset

    # end of trailing sell parameters
    # -----------------------------------------------------