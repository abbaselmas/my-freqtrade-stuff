from freqtrade.strategy import (IStrategy, informative)
from typing import Dict, List
from pandas import DataFrame
import pandas_ta as pta
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 15,
    "ewo_high": 8.32,
    "ewo_high_2": -4.44,
    "ewo_low": -6.42,
    "fast_ewo": 9,
    "low_offset": 1.18,
    "low_offset_2": 0.93,
    "rsi_buy": 67,
    "rsi_ewo2": 18,
    "rsi_fast_ewo1": 56,
    "slow_ewo": 198,
    "buy_bb20_close_bblowerband_safe_1": 0.983,
    "buy_bb20_close_bblowerband_safe_2": 0.998,
    "buy_macd_1": 0.09,
    "buy_macd_2": 0.01,
    "buy_rsi_1h_1": 26,
    "buy_rsi_1h_2": 38.4,
    "buy_rsi_1h_5": 58.5,
    "buy_rsi_3": 18.4,
    "buy_volume_drop_1": 10,
    "buy_volume_drop_2": 1.9,
    "buy_volume_drop_3": 10.0,
    "buy_volume_pump_1": 0.3
}
# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 15,
    "high_offset": 1.11
}
class abbas8(IStrategy):
    def version(self) -> str:
        return "v9.8.3"
    INTERFACE_VERSION = 3
    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.08, -0.05, decimals=3, name="stoploss")]
        # Define custom trailing space
        def trailing_space() -> List[Dimension]:
            return[
                Categorical([True], name="trailing_stop"),
                SKDecimal(0.0002, 0.0006, decimals=4, name="trailing_stop_positive"),
                SKDecimal(0.010,  0.020, decimals=3, name="trailing_stop_positive_offset_p1"),
                Categorical([True], name="trailing_only_offset_is_reached"),
            ]
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

    order_time_in_force = {
        "entry": "gtc",
        "exit": "ioc"
    }
    slippage_protection = {
        "retries": 3,
        "max_slippage": -0.002
    }
    
    buy_bb20_close_bblowerband_safe_1 = DecimalParameter(0.9, 1.1, default=buy_params["buy_bb20_close_bblowerband_safe_1"], space="buy", decimals=2, optimize=True)
    buy_bb20_close_bblowerband_safe_2 = DecimalParameter(0.9, 1.1, default=buy_params["buy_bb20_close_bblowerband_safe_2"], space="buy", decimals=2, optimize=True)
    
    buy_volume_optimize = True
    buy_volume_pump_1 = DecimalParameter(0.10, 0.60, default=buy_params["buy_volume_pump_1"], space="buy", decimals=2, optimize=buy_volume_optimize)
    buy_volume_drop_1 = DecimalParameter(1, 10, default=buy_params["buy_volume_drop_1"], space="buy", decimals=1, optimize=buy_volume_optimize)
    
    buy_rsi_optimize = True
    buy_rsi_1h_1 = IntParameter(10, 40, default=buy_params["buy_rsi_1h_1"], space="buy", optimize=buy_rsi_optimize)
    buy_rsi_1h_2 = IntParameter(10, 40, default=buy_params["buy_rsi_1h_2"], space="buy", optimize=buy_rsi_optimize)
    buy_rsi_1h_5 = IntParameter(10, 60, default=buy_params["buy_rsi_1h_5"], space="buy", optimize=buy_rsi_optimize)
    
    buy_macd_1 = DecimalParameter(0.01, 0.09, default=buy_params["buy_macd_1"], space="buy", decimals=2, optimize=True)
    buy_macd_2 = DecimalParameter(0.01, 0.09, default=buy_params["buy_macd_2"], space="buy", decimals=2, optimize=True)

    is_optimize_vwap = True
    buy_vwap_width      = DecimalParameter(0.05, 10.0, default=0.80, optimize = is_optimize_vwap)
    buy_vwap_closedelta = DecimalParameter(10.0, 30.0, default=15.0, optimize = is_optimize_vwap)
    buy_vwap_cti        = DecimalParameter(-0.9, -0.0, default=-0.6, optimize = is_optimize_vwap)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for info_timeframe in self.info_timeframes:
            informative_pairs.extend([(pair, info_timeframe) for pair in pairs])
        return informative_pairs
    
    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")

        informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h["rsi"] = ta.RSI(informative_1h, timeperiod=14)

        # Heikin Ashi
        inf_heikinashi = qtpylib.heikinashi(informative_1h)
        informative_1h['ha_close'] = inf_heikinashi['close']
        informative_1h['rocr'] = ta.ROCR(informative_1h['ha_close'], timeperiod=168)

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
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["volume_mean_slow"] = dataframe["volume"].rolling(window=30).mean()

        ## BB 40
        bollinger2_40 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        dataframe['bb_upperband2_40'] = bollinger2_40['upper']
        # EMA
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_6'] = ta.RSI(dataframe, timeperiod=6)
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)
        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['vwap_width'] = ( (dataframe['vwap_upperband'] - dataframe['vwap_lowerband']) / dataframe['vwap_middleband'] ) * 100
        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        # ClucHA
        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)
        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        # BinH
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        return dataframe
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, sell_reason: str, current_time: datetime, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        try:
            state = self.slippage_protection["__pair_retries"]
        except KeyError:
            state = self.slippage_protection["__pair_retries"] = {}
        candle = dataframe.iloc[-1].squeeze()
        slippage = (rate / candle["close"]) - 1
        if slippage < self.slippage_protection["max_slippage"]:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection["retries"]:
                state[pair] = pair_retries + 1
                return False
        state[pair] = 0
        return True
    
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
        # dataframe.loc[
        #     (
        #         (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
        #         (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
        #         (dataframe["ewo"] > self.ewo_high.value) &
        #         (dataframe["rsi"] < self.rsi_buy.value) &
        #         (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "ewo1")
        
        # dataframe.loc[
        #     (
        #         (dataframe["rsi_fast"] < self.rsi_fast_ewo1.value) &
        #         (dataframe["close"] < (dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value)) &
        #         (dataframe["ewo"] < self.ewo_low.value) &
        #         (dataframe["close"] < (dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "ewolow")
        
        # dataframe.loc[
        #     (   
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] < dataframe['bb_lowerband'] *  self.buy_bb20_close_bblowerband_safe_2.value) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 2")
        
        # dataframe.loc[
        #     (   
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_1.value) &
        #         (dataframe['close'] < dataframe['bb_lowerband']) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 4")
        
        # dataframe.loc[
        #     (   
        #         (dataframe['ema_26'] > dataframe['ema_12']) &
        #         ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_2.value)) &
        #         ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
        #         (dataframe['close'] < (dataframe['bb_lowerband'])) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 6")
        # dataframe.loc[
        #     (
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_2.value) &
        #         (dataframe['ema_26'] > dataframe['ema_12']) &
        #         ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
        #         ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(30) * self.buy_volume_pump_1.value)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 7")
        
        # dataframe.loc[
        #     (
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] < dataframe['bb_lowerband'] *  self.buy_bb20_close_bblowerband_safe_2.value) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         (dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 10")
        
        # dataframe.loc[
        #     (
        #         (dataframe['close'] > dataframe['ema_200']) &
        #         (dataframe['close'] > dataframe['ema_200_1h']) &
        #         (dataframe['close'] < dataframe['bb_lowerband'] * self.buy_bb20_close_bblowerband_safe_1.value) &
        #         (dataframe['low'] < dataframe['bb_lowerband'] * self.buy_bb20_close_bblowerband_safe_2.value) &
        #         (dataframe['close'].shift() > dataframe['bb_lowerband']) &
        #         (dataframe['rsi_1h'] < 72.8) &
        #         (dataframe['open'] > dataframe['close']) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
        #         ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 12")
        
        # dataframe.loc[
        #     (
        #         (dataframe['rsi_1h'] < self.buy_rsi_1h_5.value) &
        #         (dataframe['ema_26'] > dataframe['ema_12']) &
        #         ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_2.value)) &
        #         ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
        #         (dataframe['close'] < (dataframe['bb_lowerband'])) &
        #         (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
        #         (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
        #         (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "cond 16")
        

        dataframe.loc[
            (
                (dataframe['close'] < dataframe['vwap_lowerband']) &
                (dataframe['vwap_width'] > self.buy_vwap_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_vwap_closedelta.value / 1000 ) &
                (dataframe['cti'] < self.buy_vwap_cti.value) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60)
            ),
            ["enter_long", "enter_tag"]] = (1, "vwap")
        dataframe.loc[
            (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60)
            ),
            ["enter_long", "enter_tag"]] = (1, "cofi")
        dataframe.loc[
            (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h.value )
                (dataframe['bb_lowerband2_40'].shift() > 0) &
                (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close.value) &
                (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close.value) &
                (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail.value) &
                (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                (dataframe['ha_close'] < dataframe['ha_close'].shift()) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60)
            ),
            ["enter_long", "enter_tag"]] = (1, "clucha")

        dont_buy_conditions = []
        
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, "enter_long"] = 0
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
def EWO(dataframe, ema_length=5, ema2_length=35):
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe["low"] * 100

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
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']

# VWAP bands
def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']