from freqtrade.strategy import (IStrategy, informative)
from typing import Dict, List
from pandas import DataFrame, Series
import pandas_ta as pta
import numpy as np
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
import logging
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter, stoploss_from_open
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real
from datetime import datetime, timedelta, timezone
from technical.indicators import RMI,vwmacd

logger = logging.getLogger(__name__)

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 15,
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
    "high_offset": 1.11,
    "pHSL": -0.073,
    "pPF_1": 0.016,
    "pPF_2": 0.076,
    "pSL_1": 0.011,
    "pSL_2": 0.028
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

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc
def SSLChannels_ATR(dataframe, length=7):
    df = dataframe.copy()

    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

    return df['sslDown'], df['sslUp']

class abbas8(IStrategy):
    def version(self) -> str:
        return "v9.8.6"
    INTERFACE_VERSION = 3
    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.12, -0.03, decimals=3, name="stoploss")]
        # Define custom trailing space
        def trailing_space() -> List[Dimension]:
            return[
                Categorical([True], name="trailing_stop"),
                SKDecimal(0.0002, 0.0020, decimals=4, name="trailing_stop_positive"),
                SKDecimal(0.010,  0.030, decimals=3, name="trailing_stop_positive_offset_p1"),
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

    # BeastBotXBLR
    ###########################################################################
    # Buy
    optc1 = True
    buy_rmi_length = IntParameter(8, 20, default=8, optimize = optc1, load=True)
    buy_rmi = IntParameter(30, 50, default=35, optimize= optc1, load=True)
    buy_cci_length = IntParameter(25, 45, default=25, optimize = optc1, load=True)
    buy_cci = IntParameter(-135, -90, default=-133, optimize= optc1, load=True)
    buy_srsi_fk = IntParameter(30, 50, default=25, optimize= optc1, load=True)
    buy_bb_width   = DecimalParameter(0.065, 0.135, default=0.095, decimals=3, optimize = optc1, load=True)
    buy_bb_delta   = DecimalParameter(0.018, 0.035, default=0.025, decimals=3, optimize = optc1, load=True)
    buy_bb_factor  = DecimalParameter(0.990, 0.999, default=0.995, decimals=3, optimize = optc1, load=True)
    buy_closedelta = DecimalParameter( 12.0, 18.0,  default=15.0,  decimals=1, optimize = optc1, load=True)

    optc2 = True
    buy_c2_1 = DecimalParameter(0.010, 0.025, default=0.018, space='buy', decimals=3, optimize=optc2, load=True)
    buy_c2_2 = DecimalParameter(0.980, 0.995, default=0.982, space='buy', decimals=3, optimize=optc2, load=True)
    buy_c2_3 = DecimalParameter(-0.8, -0.3,   default=-0.5,  space='buy', decimals=1, optimize=optc2, load=True)
    
    optc3 = True
    buy_con3_1 = DecimalParameter(0.010, 0.025, default=0.017, space='buy', decimals=3, optimize=optc3, load=True)
    buy_con3_2 = DecimalParameter(0.980, 0.995, default=0.984, space='buy', decimals=3, optimize=optc3, load=True)
    buy_con3_3 = DecimalParameter(0.955, 0.975, default=0.965, space='buy', decimals=3, optimize=optc3, load=True)
    buy_con3_4 = DecimalParameter(-0.95, -0.70, default=-0.85, space='buy', decimals=2, optimize=optc3, load=True)

    optc4 = True
    buy_rsi_1h_42      = IntParameter(10, 50, default=15, space='buy', optimize=optc4, load=True)
    buy_macd_41        = DecimalParameter(0.01, 0.09, default=0.02, space='buy', decimals=2, optimize=optc4, load=True)
    buy_volume_pump_41 = DecimalParameter(0.1, 0.9,   default=0.4,  space='buy', decimals=1, optimize=optc4, load=True)
    buy_volume_drop_41 = DecimalParameter(1, 10,      default=3.8,  space='buy', decimals=1, optimize=optc4, load=True)

    optc6 = True
    buy_c6_2 = DecimalParameter(0.980, 0.999, default=0.985, space='buy', decimals=3, optimize=optc6, load=True)
    buy_c6_1 = DecimalParameter(0.08, 0.20,   default=0.12,  space='buy', decimals=2, optimize=optc6, load=True) 
    buy_c6_2 = DecimalParameter(0.02, 0.40,   default=0.28,  space='buy', decimals=2, optimize=optc6, load=True)
    buy_c6_3 = DecimalParameter(0.005, 0.040, default=0.031, space='buy', decimals=3, optimize=optc6, load=True) 
    buy_c6_4 = DecimalParameter(0.010, 0.030, default=0.021, space='buy', decimals=3, optimize=optc6, load=True)
    buy_c6_5 = DecimalParameter(0.20, 0.40,   default=0.26,  space='buy', decimals=2, optimize=optc6, load=True)

    optc7 = True
    buy_c7_1 = DecimalParameter(0.95, 1.10, default=1.01, space='buy', decimals=2, optimize=optc7, load=True)
    buy_c7_2 = DecimalParameter(0.95, 1.10, default=0.99, space='buy', decimals=2, optimize=optc7, load=True)
    buy_c7_3 = IntParameter(-100, -80, default=-94, space='buy', optimize= optc7, load=True)
    buy_c7_4 = IntParameter(-90, -60,  default=-75, space='buy', optimize= optc7, load=True)
    buy_c7_5 = IntParameter(75, 90,    default= 80, space='buy', optimize= optc7, load=True)

    optc8 = True
    buy_min_inc_1 = DecimalParameter(0.010, 0.050, default=0.022, space='buy', decimals=3, optimize=optc8, load=True)
    buy_rsi_1h_min_1 = IntParameter(25, 40, default=30, space='buy', optimize=optc8, load=True)
    buy_rsi_1h_max_1 = IntParameter(70, 90, default=84, space='buy', optimize=optc8, load=True)
    buy_rsi_1 = IntParameter(20, 40, default=36.0, space='buy', optimize=optc8, load=True)
    buy_mfi_1 = IntParameter(20, 40, default=26.0, space='buy', optimize=optc8, load=True)

    optc9 = False
    buy_c9_1 = IntParameter(25, 44,   default=36,  space='buy',  optimize=optc9, load=True)
    buy_c9_2 = IntParameter(-80, -67, default=-75, space='buy',  optimize=optc9, load=True)
    buy_c9_3 = IntParameter(-80, -67, default=-75, space='buy',  optimize=optc9, load=True)
    buy_c9_4 = IntParameter(35, 54,   default=46,  space='buy',  optimize=optc9, load=True)
    buy_c9_5 = IntParameter(20, 44,   default=30,  space='buy',  optimize=optc9, load=True)
    buy_c9_6 = IntParameter(65, 94,   default=84,  space='buy',  optimize=optc9, load=True)
    buy_c9_7 = IntParameter(-110, -80, default=-99, space='buy',  optimize=optc9, load=True)

    optc10 = False
    buy_c10_1 = IntParameter(-110, -80, default=-99, space='buy', optimize=optc10, load=True)
    buy_c10_2 = DecimalParameter(-1.00, -0.50, default=-0.78, space='buy', decimals=2, optimize=optc10, load=True)

    dip_optimize = False
    buy_dip_threshold_5 = DecimalParameter(0.001, 0.050, default=0.015, space='buy', decimals=3, optimize=dip_optimize, load=True)
    buy_dip_threshold_6 = DecimalParameter(0.010, 0.200, default=0.06, space='buy', decimals=3, optimize=dip_optimize, load=True)
    buy_dip_threshold_7 = DecimalParameter(0.050, 0.400, default=0.24, space='buy', decimals=3, optimize=dip_optimize, load=True)
    buy_dip_threshold_8 = DecimalParameter(0.200, 0.500, default=0.4, space='buy', decimals=3, optimize=dip_optimize, load=True)
    # 24 hours
    buy_pump_pull_threshold_1 = DecimalParameter(1.50, 3.00, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_1 = DecimalParameter(0.400, 1.000, default=0.5, space='buy', decimals=2, optimize=False, load=True)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for info_timeframe in self.info_timeframes:
            informative_pairs.extend([(pair, info_timeframe) for pair in pairs])
        return informative_pairs
    
    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")

        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)
        #informative_1h['not_downtrend'] = ((informative_1h['close'] > informative_1h['close'].shift(2)) | (informative_1h['rsi'] > 50))
        informative_1h['r_480'] = williams_r(dataframe, period=480)
        informative_1h['safe_pump_24'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) /
            informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_1.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) /
            self.buy_pump_pull_threshold_1.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))

        informative_1h['cti'] = pta.cti(informative_1h["close"], length=20) 

        ssldown, sslup = SSLChannels_ATR(informative_1h, 14)
        informative_1h['ssl-dir'] = np.where(sslup > ssldown,'up','down')

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
        dataframe["volume_mean_slow_30"] = dataframe["volume"].rolling(window=30).mean()

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        # nuevo #
        
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe['bb_delta'] = ((dataframe['bb_lowerband'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband'])
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        # CCI hyperopt
        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cci'] = ta.CCI(dataframe, 26)

        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)

        # SRSI hyperopt ?
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']

        # Volume
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=48).mean()

        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_32'] = williams_r(dataframe, period=32)
        dataframe['r_64'] = williams_r(dataframe, period=64)

        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        
        dataframe['safe_dips_strict'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_5.value) &
                                  (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_6.value) &
                                  (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_7.value) &
                                  (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_8.value))

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
        dataframe["ha_tail"] = (dataframe["ha_close"] - dataframe["ha_low"]).abs()
        dataframe["rocr"] = ta.ROCR(dataframe["ha_close"], timeperiod=28)
        # CTI
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)
        # BinH
        dataframe["closedelta"] = (dataframe["close"] - dataframe["close"].shift()).abs()
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

        ssldown, sslup = SSLChannels_ATR(dataframe, 64)
        dataframe['ssl-up'] = sslup
        dataframe['ssl-down'] = ssldown
        dataframe['ssl-dir'] = np.where(sslup > ssldown,'up','down')
        dataframe['rmi'] =  RMI(dataframe, length=24, mom=5)
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
        #         (dataframe["close"] < dataframe["vwap_lowerband"]) &
        #         (dataframe["vwap_width"] > self.buy_vwap_width.value) &
        #         (dataframe["closedelta"] > dataframe["close"] * self.buy_vwap_closedelta.value / 1000 ) &
        #         (dataframe["cti"] < self.buy_vwap_cti.value) &
        #         (dataframe["rsi_84"] < 60) &
        #         (dataframe["rsi_112"] < 60)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "vwap")
        # dataframe.loc[
        #     (
        #         (dataframe["rocr_1h"] > self.buy_clucha_rocr_1h.value ) &
        #         (dataframe["bb_lowerband2_40"].shift() > 0) &
        #         (dataframe["bb_delta_cluc"] > dataframe["ha_close"] * self.buy_clucha_bbdelta_close.value) &
        #         (dataframe["ha_closedelta"] > dataframe["ha_close"] * self.buy_clucha_closedelta_close.value) &
        #         (dataframe["ha_tail"] < dataframe["bb_delta_cluc"] * self.buy_clucha_bbdelta_tail.value) &
        #         (dataframe["ha_close"] < dataframe["bb_lowerband2_40"].shift()) &
        #         (dataframe["ha_close"] < dataframe["ha_close"].shift()) &
        #         (dataframe["rsi_84"] < 60) &
        #         (dataframe["rsi_112"] < 60)
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "clucha")
        
        dataframe.loc[
            (
                (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value) &
                ((dataframe['bb_delta'] > self.buy_bb_delta.value) & (dataframe['bb_width'] > self.buy_bb_width.value)) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &    
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con1")
        dataframe.loc[
            (
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_c2_1.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_c2_2.value)) &
                (dataframe['cti_1h'] > self.buy_c2_3.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con2")
        dataframe.loc[
            (
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_con3_1.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_con3_2.value)) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_con3_3.value) &
                (dataframe['cti'] < self.buy_con3_4.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con3")
        dataframe.loc[
            (
                (dataframe['rsi_1h'] < self.buy_rsi_1h_42.value) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_41.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_41.value)) &
                (dataframe['volume_mean_slow_30'] > dataframe['volume_mean_slow_30'].shift(48) * self.buy_volume_pump_41.value) &
                (dataframe['volume_mean_slow_30'] * self.buy_volume_pump_41.value < dataframe['volume_mean_slow_30'].shift(48))
            ),
            ["enter_long", "enter_tag"]] = (1, "con4")
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_c6_1.value) &
                (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_c6_2.value) &
                dataframe['bb_lowerband'].shift().gt(0) &
                dataframe['bb_delta'].gt(dataframe['close'] * self.buy_c6_3.value) &
                dataframe['closedelta'].gt(dataframe['close'] * self.buy_c6_4.value) &
                dataframe['tail'].lt(dataframe['bb_delta'] * self.buy_c6_5.value) &
                dataframe['close'].lt(dataframe['bb_lowerband'].shift()) &
                dataframe['close'].le(dataframe['close'].shift())
            ),
            ["enter_long", "enter_tag"]] = (1, "con6")
        dataframe.loc[
            (
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * self.buy_c7_1.value)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_c7_2.value)) &
                (dataframe['r_14'] < self.buy_c7_3.value) &
                (dataframe['r_64'] < self.buy_c7_4.value) &
                (dataframe['rsi_1h'] < self.buy_c7_5.value) 
            ),
            ["enter_long", "enter_tag"]] = (1, "con7")
        dataframe.loc[
            (
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (dataframe['sma_200'] > dataframe['sma_200'].shift(50)) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_24_1h']) &

                (((dataframe['close'] - dataframe['open'].rolling(36).min()) / dataframe['open'].rolling(36).min()) > self.buy_min_inc_1.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_1.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_1.value) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con8")
        dataframe.loc[
            (
                (((dataframe['close'] - dataframe['open'].rolling(12).min()) / dataframe['open'].rolling(12).min()) > 0.032) &
                (dataframe['rsi'] < self.buy_c9_1.value) &
                (dataframe['r_14'] < self.buy_c9_2.value) &
                (dataframe['r_32'] < self.buy_c9_3.value) &
                (dataframe['mfi'] < self.buy_c9_4.value) &
                (dataframe['rsi_1h'] > self.buy_c9_5.value) &
                (dataframe['rsi_1h'] < self.buy_c9_6.value) &
                (dataframe['r_480_1h'] > self.buy_c9_7.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con9")
        dataframe.loc[
            (
                (dataframe['close'].shift(4) < (dataframe['close'].shift(3))) &
                (dataframe['close'].shift(3) < (dataframe['close'].shift(2))) &
                (dataframe['close'].shift(2) < (dataframe['close'].shift())) &
                (dataframe['close'].shift(1) < (dataframe['close'])) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['close'] > (dataframe['open'])) &
                (dataframe['cci'].shift() < dataframe['cci']) &
                (dataframe['ssl-dir_1h'] == 'up') &
                (dataframe['cci'] < self.buy_c10_1.value) &
                (dataframe['cti'] < self.buy_c10_2.value)
            ),
            ["enter_long", "enter_tag"]] = (1, "con10")
        # BCMBİGZ
        # dataframe.loc[
        #     (
        #         (dataframe["close"] > dataframe["ema_200_1h"]) & 
        #         (dataframe["ema_50"] > dataframe["ema_200"]) &
        #         (dataframe["ema_50_1h"] > dataframe["ema_200_1h"]) &
        #         (((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]) < self.buy_dip_threshold_1.value) &
        #         (((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"])< self.buy_dip_threshold_2.value) &
        #         dataframe["lower"].shift().gt(0) &
        #         dataframe["bbdelta"].gt(dataframe["close"] * self.buy_bb40_bbdelta_close.value) &
        #         dataframe["closedelta"].gt(dataframe["close"] * self.buy_bb40_closedelta_close.value) &
        #         dataframe["tail"].lt(dataframe["bbdelta"] * self.buy_bb40_tail_bbdelta.value) &
        #         dataframe["close"].lt(dataframe["lower"].shift()) &
        #         dataframe["close"].le(dataframe["close"].shift())
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 0")
        # dataframe.loc[
        #     (
        #         (dataframe["close"] > dataframe["ema_200"]) &
        #         (dataframe["close"] > dataframe["ema_200_1h"]) &
        #         (dataframe["ema_50_1h"] > dataframe["ema_100_1h"]) &
        #         (dataframe["ema_50_1h"] > dataframe["ema_200_1h"]) &
        #         (((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]) < self.buy_dip_threshold_1.value) &
        #         (((dataframe["open"].rolling(12).max() - dataframe["close"])/ dataframe["close"]) < self.buy_dip_threshold_2.value) &
        #         (dataframe["close"] < dataframe["ema_50"]) &
        #         (dataframe["close"] < self.buy_bb20_close_bblowerband.value * dataframe["bb_lowerband"]) &
        #         (dataframe["volume"] < (dataframe["volume_mean_slow"].shift(1) * self.buy_bb20_volume.value))
        #     ),
        #     ["enter_long", "enter_tag"]] = (1, "BCMBİGZ 1")

        dont_buy_conditions = []
        
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, "enter_long"] = 0
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    # ## Trailing params
    # use_custom_stoploss = True
    # # hard stoploss profit
    # pHSL = DecimalParameter(-0.10, -0.040, default=-0.08, decimals=3, space='sell', load=True, optimize=True)
    # # profit threshold 1, trigger point, SL_1 is used
    # pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True, optimize=True)
    # pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True, optimize=True)
    # # profit threshold 2, SL_2 is used
    # pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True, optimize=True)
    # pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True, optimize=True)
    # # Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
    #     # hard stoploss profit
    #     HSL = self.pHSL.value
    #     PF_1 = self.pPF_1.value
    #     SL_1 = self.pSL_1.value
    #     PF_2 = self.pPF_2.value
    #     SL_2 = self.pSL_2.value
    #     # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
    #     # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
    #     # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.
    #     if (current_profit > PF_2):
    #         sl_profit = SL_2 + (current_profit - PF_2)
    #     elif (current_profit > PF_1):
    #         sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
    #     else:
    #         sl_profit = HSL
    #     # Only for hyperopt invalid return
    #     if (sl_profit >= current_profit):
    #         return -0.99
    #     return stoploss_from_open(sl_profit, current_profit)