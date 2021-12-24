# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from numpy.core.records import ndarray

# --------------------------------
import talib.abstract as ta
#from technical.indicators import mmar

class multiRisMMAR(IStrategy):
    """

    author@: Creslin

    based on work from Creslin

    """
    minimal_roi = {
        "0": 0.80
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.80

    # Optimal ticker interval for the strategy
    ticker_interval = '4h'

    def get_ticker_indicator(self):
        return int(self.ticker_interval[:-1])

    def mmar(self, dataframe, matype="EMA", src="close", debug=False):
        """
        Madrid Moving Average Ribbon

        Returns: MMAR
        """
        """
        Author(Freqtrade): Creslinux
        Original Author(TrdingView): "Madrid"

        Pinescript from TV Source Code and Description 
        //
        // Madrid : 17/OCT/2014 22:51M: Moving Average Ribbon : 2.0 : MMAR
        // http://madridjourneyonws.blogspot.com/
        //
        // This plots a moving average ribbon, either exponential or standard.
        // This study is best viewed with a dark background.  It provides an easy
        // and fast way to determine the trend direction and possible reversals.
        //
        // Lime : Uptrend. Long trading
        // Green : Reentry (buy the dip) or downtrend reversal warning
        // Red : Downtrend. Short trading
        // Maroon : Short Reentry (sell the peak) or uptrend reversal warning
        //
        // To best determine if this is a reentry point or a trend reversal
        // the MMARB (Madrid Moving Average Ribbon Bar) study is used.
        // This is the bar located at the bottom.  This bar signals when a
        // current trend reentry is found (partially filled with opposite dark color)
        // or when a trend reversal is ahead (completely filled with opposite dark color).
        //

        study(title="Madrid Moving Average Ribbon", shorttitle="MMAR", overlay=true)
        exponential = input(true, title="Exponential MA")

        src = close

        ma05 = exponential ? ema(src, 05) : sma(src, 05)
        ma10 = exponential ? ema(src, 10) : sma(src, 10)
        ma15 = exponential ? ema(src, 15) : sma(src, 15)
        ma20 = exponential ? ema(src, 20) : sma(src, 20)
        ma25 = exponential ? ema(src, 25) : sma(src, 25)
        ma30 = exponential ? ema(src, 30) : sma(src, 30)
        ma35 = exponential ? ema(src, 35) : sma(src, 35)
        ma40 = exponential ? ema(src, 40) : sma(src, 40)
        ma45 = exponential ? ema(src, 45) : sma(src, 45)
        ma50 = exponential ? ema(src, 50) : sma(src, 50)
        ma55 = exponential ? ema(src, 55) : sma(src, 55)
        ma60 = exponential ? ema(src, 60) : sma(src, 60)
        ma65 = exponential ? ema(src, 65) : sma(src, 65)
        ma70 = exponential ? ema(src, 70) : sma(src, 70)
        ma75 = exponential ? ema(src, 75) : sma(src, 75)
        ma80 = exponential ? ema(src, 80) : sma(src, 80)
        ma85 = exponential ? ema(src, 85) : sma(src, 85)
        ma90 = exponential ? ema(src, 90) : sma(src, 90)
        ma100 = exponential ? ema(src, 100) : sma(src, 100)

        leadMAColor = change(ma05)>=0 and ma05>ma100 ? lime
                    : change(ma05)<0  and ma05>ma100 ? maroon
                    : change(ma05)<=0 and ma05<ma100 ? red
                    : change(ma05)>=0 and ma05<ma100 ? green
                    : gray
        maColor(ma, maRef) =>
                      change(ma)>=0 and ma05>maRef ? lime
                    : change(ma)<0  and ma05>maRef ? maroon
                    : change(ma)<=0 and ma05<maRef ? red
                    : change(ma)>=0 and ma05<maRef ? green
                    : gray

        plot( ma05, color=leadMAColor, style=line, title="MMA05", linewidth=3)
        plot( ma10, color=maColor(ma10,ma100), style=line, title="MMA10", linewidth=1)
        plot( ma15, color=maColor(ma15,ma100), style=line, title="MMA15", linewidth=1)
        plot( ma20, color=maColor(ma20,ma100), style=line, title="MMA20", linewidth=1)
        plot( ma25, color=maColor(ma25,ma100), style=line, title="MMA25", linewidth=1)
        plot( ma30, color=maColor(ma30,ma100), style=line, title="MMA30", linewidth=1)
        plot( ma35, color=maColor(ma35,ma100), style=line, title="MMA35", linewidth=1)
        plot( ma40, color=maColor(ma40,ma100), style=line, title="MMA40", linewidth=1)
        plot( ma45, color=maColor(ma45,ma100), style=line, title="MMA45", linewidth=1)
        plot( ma50, color=maColor(ma50,ma100), style=line, title="MMA50", linewidth=1)
        plot( ma55, color=maColor(ma55,ma100), style=line, title="MMA55", linewidth=1)
        plot( ma60, color=maColor(ma60,ma100), style=line, title="MMA60", linewidth=1)
        plot( ma65, color=maColor(ma65,ma100), style=line, title="MMA65", linewidth=1)
        plot( ma70, color=maColor(ma70,ma100), style=line, title="MMA70", linewidth=1)
        plot( ma75, color=maColor(ma75,ma100), style=line, title="MMA75", linewidth=1)
        plot( ma80, color=maColor(ma80,ma100), style=line, title="MMA80", linewidth=1)
        plot( ma85, color=maColor(ma85,ma100), style=line, title="MMA85", linewidth=1)
        plot( ma90, color=maColor(ma90,ma100), style=line, title="MMA90", linewidth=3)
        :return:
        """
        import talib as ta

        matype = matype
        src = src
        df = dataframe
        debug = debug

        # Default to EMA, allow SMA if passed to def.
        if matype == "EMA" or matype == "ema":
            ma = ta.EMA
        elif matype == "SMA" or matype == "sma":
            ma = ta.SMA
        else:
            ma = ta.EMA

        # Get MAs, also last MA in own column to pass to def later
        df["ma05"] = ma(df[src], 5)
        df['ma05l'] = df['ma05'].shift(+1)
        df["ma10"] = ma(df[src], 10)
        df['ma10l'] = df['ma10'].shift(+1)
        df["ma20"] = ma(df[src], 20)
        df['ma20l'] = df['ma20'].shift(+1)
        df["ma30"] = ma(df[src], 30)
        df['ma30l'] = df['ma30'].shift(+1)
        df["ma40"] = ma(df[src], 40)
        df['ma40l'] = df['ma40'].shift(+1)
        df["ma50"] = ma(df[src], 50)
        df['ma50l'] = df['ma50'].shift(+1)
        df["ma60"] = ma(df[src], 60)
        df['ma60l'] = df['ma60'].shift(+1)
        df["ma70"] = ma(df[src], 70)
        df['ma70l'] = df['ma70'].shift(+1)
        df["ma80"] = ma(df[src], 80)
        df['ma80l'] = df['ma80'].shift(+1)
        df["ma90"] = ma(df[src], 90)
        df['ma90l'] = df['ma90'].shift(+1)
        df["ma100"] = ma(df[src], 100)
        df['ma100l'] = df['ma100'].shift(+1)

        """ logic for LeadMA
        : change(ma05)>=0 and ma05>ma100 ? lime    +2
        : change(ma05)<0  and ma05>ma100 ? maroon  -1
        : change(ma05)<=0 and ma05<ma100 ? red     -2
        : change(ma05)>=0 and ma05<ma100 ? green   +1
        : gray
        """

        def leadMAc(x):
            if (x['ma05'] - x['ma05l']) >= 0 and (x['ma05'] > x['ma100']):
                # Lime: Uptrend.Long trading
                x["leadMA"] = "lime"
                return x["leadMA"]
            elif (x['ma05'] - x['ma05l']) < 0 and (x['ma05'] > x['ma100']):
                # Maroon : Short Reentry (sell the peak) or uptrend reversal warning
                x["leadMA"] = "maroon"
                return x["leadMA"]
            elif (x['ma05'] - x['ma05l']) <= 0 and (x['ma05'] < x['ma100']):
                # Red : Downtrend. Short trading
                x["leadMA"] = "red"
                return x["leadMA"]
            elif (x['ma05'] - x['ma05l']) >= 0 and (x['ma05'] < x['ma100']):
                # Green: Reentry(buy the dip) or downtrend reversal warning
                x["leadMA"] = "green"
                return x["leadMA"]
            else:
                # If its great it means not enough ticker data for lookback
                x["leadMA"] = "grey"
                return x["leadMA"]

        df['leadMA'] = df.apply(leadMAc, axis=1)

        """   Logic for MAs 
        : change(ma)>=0 and ma05>ma100 ? lime
        : change(ma)<0  and ma05>ma100 ? maroon
        : change(ma)<=0 and ma05<ma100 ? red
        : change(ma)>=0 and ma05<ma100 ? green
        : gray
        """

        def maColor(x, ma):
            col_label = '_'.join([ma, "c"])
            col_lable_l = ''.join([ma, "l"])

            if (x[ma] - x[col_lable_l]) >= 0 and (x[ma] > x['ma100']):
                # Lime: Uptrend.Long trading
                x[col_label] = "lime"
                return x[col_label]
            elif (x[ma] - x[col_lable_l]) < 0 and (x[ma] > x['ma100']):
                # Maroon : Short Reentry (sell the peak) or uptrend reversal warning
                x[col_label] = "maroon"
                return x[col_label]

            elif (x[ma] - x[col_lable_l]) <= 0 and (x[ma] < x['ma100']):
                # Red : Downtrend. Short trading
                x[col_label] = "red"
                return x[col_label]

            elif (x[ma] - x[col_lable_l]) >= 0 and (x[ma] < x['ma100']):
                # Green: Reentry(buy the dip) or downtrend reversal warning
                x[col_label] = "green"
                return x[col_label]
            else:
                # If its great it means not enough ticker data for lookback
                x[col_label] = 'grey'
                return x[col_label]

        df['ma05_c'] = df.apply(maColor, ma="ma05", axis=1)
        df['ma10_c'] = df.apply(maColor, ma="ma10", axis=1)
        df['ma20_c'] = df.apply(maColor, ma="ma20", axis=1)
        df['ma30_c'] = df.apply(maColor, ma="ma30", axis=1)
        df['ma40_c'] = df.apply(maColor, ma="ma40", axis=1)
        df['ma50_c'] = df.apply(maColor, ma="ma50", axis=1)
        df['ma60_c'] = df.apply(maColor, ma="ma60", axis=1)
        df['ma70_c'] = df.apply(maColor, ma="ma70", axis=1)
        df['ma80_c'] = df.apply(maColor, ma="ma80", axis=1)
        df['ma90_c'] = df.apply(maColor, ma="ma90", axis=1)

        if debug:
            from pandas import set_option
            set_option('display.max_rows', 2000)
            print(df[["date","leadMA",
                      "ma05", "ma05l", "ma05_c",
                      "ma10", "ma10l", "ma10_c",
                      # "ma20", "ma20l", "ma20_c",
                      # "ma30", "ma30l", "ma30_c",
                      # "ma40", "ma40l", "ma40_c",
                      # "ma50", "ma50l", "ma50_c",
                      # "ma60", "ma60l", "ma60_c",
                      # "ma70", "ma70l", "ma70_c",
                      # "ma80", "ma80l", "ma80_c",
                      "ma90", "ma90l", "ma90_c",
                       "ma100", "leadMA" ]].tail(200))

            print(df[["date", 'close',
                      "leadMA",
                      "ma10_c",
                      "ma20_c",
                      "ma30_c",
                      "ma40_c",
                      "ma50_c",
                      "ma60_c",
                      "ma70_c",
                      "ma80_c",
                      "ma90_c"
                      ]].tail(684))

        return df['leadMA'], df['ma10_c'], df['ma20_c'], df['ma30_c'], \
               df['ma40_c'], df['ma50_c'], df['ma60_c'], df['ma70_c'], \
               df['ma80_c'], df['ma90_c']

    def madrid_sqz(self, datafame, length=34, src='close', ref=13, sqzLen=5, debug=False):
        """
        Squeeze Madrid Indicator

        Author: Creslinux
        Original Author: Madrid - Tradingview
        https://www.tradingview.com/script/9bUUSzM3-Madrid-Trend-Squeeze/

        :param datafame:
        :param lenght: min 14 - default 34
        :param src: default close
        :param ref: default 13
        :param sqzLen: default 5
        :return: df['sqz_cma_c'], df['sqz_rma_c'], df['sqz_sma_c']


        There are seven colors used for the study

        Green : Uptrend in general
        Lime : Spots the current uptrend leg
        Aqua : The maximum profitability of the leg in a long trade
        The Squeeze happens when Green+Lime+Aqua are aligned (the larger the values the better)

        Maroon : Downtrend in general
        Red : Spots the current downtrend leg
        Fuchsia: The maximum profitability of the leg in a short trade
        The Squeeze happens when Maroon+Red+Fuchsia are aligned (the larger the values the better)

        Yellow : The trend has come to a pause and it is either a reversal warning or a continuation. These are the entry, re-entry or closing position points.
        """

        """ 
        Original Pinescript source code
        
        ma = ema(src, len)
        closema = close - ma
        refma = ema(src, ref) - ma
        sqzma = ema(src, sqzLen) - ma
        
        hline(0)
        plotcandle(0, closema, 0, closema, color=closema >= 0?aqua: fuchsia)
        plotcandle(0, sqzma, 0, sqzma, color=sqzma >= 0?lime: red)
        plotcandle(0, refma, 0, refma, color=(refma >= 0 and closema < refma) or (
                    refma < 0 and closema > refma) ? yellow: refma >= 0 ? green: maroon)
        """
        import talib as ta
        from numpy import where

        len = length
        src = src
        ref = ref
        sqzLen = sqzLen
        df = datafame
        ema = ta.EMA
        debug = debug

        """ Original code logic
        ma = ema(src, len)
        closema = close - ma
        refma = ema(src, ref) - ma
        sqzma = ema(src, sqzLen) - ma
        """
        df['sqz_ma'] = ema(df[src], len)
        df['sqz_cma'] = df['close'] - df['sqz_ma']
        df['sqz_rma'] = ema(df[src], ref) - df['sqz_ma']
        df['sqz_sma'] = ema(df[src], sqzLen) - df['sqz_ma']

        """ Original code logic
        plotcandle(0, closema, 0, closema, color=closema >= 0?aqua: fuchsia)
        plotcandle(0, sqzma, 0, sqzma, color=sqzma >= 0?lime: red)
        
        plotcandle(0, refma, 0, refma, color=
        (refma >= 0 and closema < refma) or (refma < 0 and closema > refma) ? yellow: 
        refma >= 0 ? green: maroon)
        """

        #print(df[['sqz_cma', 'sqz_rma', 'sqz_sma']])

        def sqz_cma_c(x):
            if x['sqz_cma'] >= 0 :
                x['sqz_cma_c'] = "aqua"
                return x['sqz_cma_c']
            else:
                x['sqz_cma_c'] = "fuchsia"
                return x['sqz_cma_c']
        df['sqz_cma_c'] = df.apply(sqz_cma_c, axis=1)

        def sqz_sma_c(x):
            if x['sqz_sma'] >= 0:
                x['sqz_sma_c'] = "lime"
                return x['sqz_sma_c']
            else:
                x['sqz_sma_c'] = "red"
                return x['sqz_sma_c']
        df['sqz_sma_c'] = df.apply(sqz_sma_c, axis=1)


        def sqz_rma_c(x):
            if x['sqz_rma'] >= 0 and  x['sqz_cma'] < x['sqz_rma']:
                x['sqz_rma_c'] = "yellow"
                return x['sqz_rma_c']
            elif x['sqz_rma'] < 0 and x['sqz_cma'] > x['sqz_rma']:
                x['sqz_rma_c'] = "yellow"
                return x['sqz_rma_c']
            elif x['sqz_rma'] >= 0 :
                x['sqz_rma_c'] = "green"
                return x['sqz_rma_c']
            else:
                x['sqz_rma_c'] = "maroon"
                return x['sqz_rma_c']
        df['sqz_rma_c'] = df.apply(sqz_rma_c, axis=1)

        if debug:
            from pandas import set_option
            set_option('display.max_rows', 2000)
            print(df[['sqz_cma_c', 'sqz_rma_c', 'sqz_sma_c']])

        return df['sqz_cma_c'], df['sqz_rma_c'], df['sqz_sma_c']


    def madrid_momentum(self, dataframe, fastMALenght=34, slowMALength=89, signalMALen=13, debug=False):
        """
        Madrid Momentum Indicator


        Author: Creslin
        Original Author: Marditd - Trading View

        https://www.tradingview.com/script/mJoNRvkS-Madrid-Trend-Trading/
        :param datafame:
        :param src: hl2
        :param fastMAlength: default 34
        :param slowMAlength: default 89

        :param signalMALen: 13
        :return:
        """
        """
        Original Source Code - Pine Script
        src =  hl2
        
        // Input parameters
        fastMALength = input(34, minval=1, title="Fast Length")
        slowMALength = input(89, minval=1, title="Slow Length")
        signalMALen = input(13, title="Signal MA")
        
        fastMA = ema(src, fastMALength )
        slowMA = ema(src, slowMALength )
        trendStrength = (fastMA - slowMA)*100/slowMA
        signalMA = sma(trendStrength, signalMALen)
        
        momentum = trendStrength-signalMA
        momColor = momentum>0 and change(momentum)>0 ? lime
                 : momentum>0 and change(momentum)<0 ? green
                 : momentum<0 and change(momentum)>0 ? maroon
                 : momentum<0 and change(momentum)<0 ? red
                 : gray
                 
        plot(momentum, style=histogram, linewidth=2, color=momColor)
        plot(0, color=change(momentum)>=0?green:red, linewidth=3)
        """
        import talib as ta
        ema = ta.EMA
        sma = ta.SMA

        df = dataframe
        fastMAlength = fastMALenght
        slowMAlength = slowMALength
        signalMALen = signalMALen
        debug = debug


        """ Orginal Pinescript Logic
        fastMA = ema(src, fastMALength )
        slowMA = ema(src, slowMALength )
        trendStrength = (fastMA - slowMA)*100/slowMA
        signalMA = sma(trendStrength, signalMALen)
        momentum = trendStrength-signalMA
        """

        df['hl2'] = (df['close'] + df['open']) /2
        df['fastMA'] = ema(df['hl2'], fastMAlength )
        df['slowMA'] = ema(df['hl2'], slowMAlength )
        df['trendStrength'] = (df['fastMA'] - df['slowMA'])*100/df['slowMA']
        df['signalMA'] = sma(df['trendStrength'], signalMALen)

        df['momentum'] = df['trendStrength'] - df['signalMA']
        df['mom_change'] =  df['momentum'] -  df['momentum'].shift(+1)

        """  Original Pinescript Logic 
        momColor = momentum > 0 and change(momentum) > 0 ? lime
        : momentum > 0 and change(momentum) < 0 ? green
        : momentum < 0 and change(momentum) > 0 ? maroon
        : momentum < 0 and change(momentum) < 0 ? red
        : gray
        """
        def mom_color(x):
            if x['momentum'] > 0 and x['mom_change'] > 0 :
                x['mom_color'] = "lime"
                return x['mom_color']
            elif x['momentum'] > 0 and x['mom_change'] < 0 :
                x['mom_color'] = "green"
                return x['mom_color']
            elif x['momentum'] < 0 and x['mom_change'] > 0 :
                x['mom_color'] = "maroon"
                return x['mom_color']
            elif x['momentum'] < 0 and x['mom_change'] < 0 :
                x['mom_color'] = "red"
                return x['mom_color']
            else:
                x['mom_color'] = "grey"
                return x['mom_color']
        df['mom_color'] = df.apply(mom_color, axis=1)

        if debug:
            from pandas import set_option
            set_option('display.max_rows', 2000)
            print(df[['date', 'momentum', 'mom_change', 'mom_color']])

        return df['momentum'], df['mom_change'], df['mom_color']


    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        # MMAR - Moving Average Ribb - defaults - EMA / close
        # lime bullish, green pivot, maroon pivot, red bearish
        dataframe['leadMA'], dataframe['ma10_c'], dataframe['ma20_c'], dataframe['ma30_c'], \
        dataframe['ma40_c'], dataframe['ma50_c'], dataframe['ma60_c'], dataframe['ma70_c'], \
        dataframe['ma80_c'], dataframe['ma90_c'] = \
            self.mmar(dataframe, matype='EMA', src='close', debug=False)

        # Madrid Squeeze
        dataframe['sqz_cma_c'], \
        dataframe['sqz_rma_c'], dataframe['sqz_sma_c'] = \
            self.madrid_sqz(dataframe, length=34, src='close', ref=13, sqzLen=5)

        # Madrid Momentum
        dataframe['momentum'], dataframe['mom_change'], dataframe['mom_color'] = \
            self.madrid_momentum(dataframe, fastMALenght=34, slowMALength=89, signalMALen=13, debug=True)

        # Use Weighted MA in combination for trend - no buys when not upwards
        dataframe['wma20'] = ta.WMA(dataframe, timeperiod=20)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:

        """
        (dataframe['ma90_c'] == "green")  | (dataframe['ma90_c'] == "lime")
        (dataframe['ma80_c'] == "green") | (dataframe['ma80_c'] == "lime") |
        (dataframe['ma70_c'] == "green") | (dataframe['ma70_c'] == "lime") |
        (dataframe['ma60_c'] == "green") | (dataframe['ma60_c'] == "lime") |
        (dataframe['ma50_c'] == "green") | (dataframe['ma50_c'] == "lime") |
        (dataframe['ma40_c'] == "green") | (dataframe['ma40_c'] == "lime") |
        (dataframe['ma30_c'] == "green") | (dataframe['ma30_c'] == "lime") |
        (dataframe['ma20_c'] == "green") | (dataframe['ma20_c'] == "lime") |
        (dataframe['ma10_c'] == "green") | (dataframe['ma10_c'] == "lime") |
        (dataframe['leadMA'] == "green") | (dataframe['leadMA'] == "lime")
        """

        dataframe.loc[
            (
                    # Only consider buying when in trend - use wma 20
                    ((dataframe['wma20'] <= dataframe['close']))&
                    ((dataframe['leadMA'] == 'lime') | (dataframe['leadMA'] == 'green' ))&

                    #Check squeeze is in the good before longs
                    (dataframe['sqz_rma_c'] == "green") &
                    (dataframe['sqz_sma_c'] == "lime") &
                    (dataframe['sqz_cma_c'] == "aqua") &

                    #Check Momentum is bullish and ticking upwards
                    ((dataframe['mom_color'] == 'green') | (dataframe['mom_color'] == 'lime')) &
                    (dataframe['mom_change'] > 0) &

                    # Go long when 20 AND 30 AND 40 are bullish
                    ((dataframe['ma20_c'] == "lime") | (dataframe['ma20_c'] == "green")) &
                    ((dataframe['ma30_c'] == "lime") | (dataframe['ma30_c'] == "green")) &
                    ((dataframe['ma40_c'] == "lime") | (dataframe['ma40_c'] == "green"))
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                #
                #
                # Close when 20 OR  30 OR 40 are bearish
                ((dataframe['ma20_c'] == "red") | (dataframe['ma20_c'] == "maroon")) |
                ((dataframe['ma30_c'] == "red") | (dataframe['ma30_c'] == "maroon")) |
                ((dataframe['ma40_c'] == "red") | (dataframe['ma40_c'] == "maroon"))
            ),
            'sell'] = 1
        return dataframe