# RokoNeural strategy
# Stupid simple triple layer NN strategy
# IT SHOULD BE OPTIMIZED BEFORE FIRST RUN USING HYPEROPT
# For better results in terms of profit or/and win ratio
# I recommend not to optimize sell space at beginning.
# Optimize roi and trailing instead.

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (DecimalParameter,
                                IStrategy, IntParameter)


class RokoNeuralV3(IStrategy):

    minimal_roi = {
        "0": 10000
    }

    stoploss = -0.9

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Hyperoptable parameters
    # Buy neural network weights
    buy_w_0_0_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_0_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_0_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_0_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_0_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_0_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_0_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_0_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_1_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_1_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_1_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_1_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_1_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_1_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_1_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_1_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_2_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_2_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_2_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_2_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_2_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_2_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_2_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_2_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_3_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_3_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_3_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_3_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_3_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_3_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_3_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_3_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_4_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_4_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_4_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_4_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_4_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_4_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_4_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_0_4_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_0_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_0_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_0_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_0_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_0_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_0_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_0_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_0_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_1_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_1_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_1_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_1_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_1_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_1_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_1_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_1_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_2_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_2_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_2_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_2_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_2_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_2_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_2_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_2_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_3_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_3_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_3_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_3_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_3_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_3_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_3_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_3_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_4_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_4_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_4_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_4_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_1_4_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_4_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_4_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_1_4_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_0_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_0_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_0_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_0_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_0_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_0_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_0_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_0_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_1_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_1_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_1_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_1_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_1_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_1_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_1_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_1_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_2_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_2_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_2_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_2_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_2_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_2_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_2_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_2_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_3_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_3_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_3_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_3_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_3_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_3_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_3_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_3_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_4_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_4_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_4_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_4_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_w_2_4_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_4_5 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_4_6 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    #buy_w_2_4_7 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_exit_0 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_exit_1 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_exit_2 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_exit_3 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)
    buy_exit_4 = DecimalParameter(low=-100, high=100, default=0, space='buy', optimize=True, load=True)

    buy_activation = IntParameter(low=0, high=100000, default=0, space='buy', optimize=True, load=True)

    # Sell neural network weights
    sell_w_0_0_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_0_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_0_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_0_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_0_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_0_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_0_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_0_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_1_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_1_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_1_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_1_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_1_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_1_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_1_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_1_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_2_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_2_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_2_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_2_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_2_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_2_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_2_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_2_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_3_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_3_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_3_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_3_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_3_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_3_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_3_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_3_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_4_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_4_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_4_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_4_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_4_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_4_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_4_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_0_4_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_0_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_0_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_0_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_0_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_0_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_0_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_0_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_0_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_1_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_1_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_1_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_1_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_1_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_1_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_1_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_1_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_2_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_2_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_2_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_2_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_2_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_2_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_2_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_2_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_3_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_3_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_3_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_3_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_3_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_3_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_3_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_3_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_4_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_4_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_4_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_4_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_1_4_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_4_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_4_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_1_4_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_0_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_0_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_0_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_0_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_0_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_0_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_0_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_0_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_1_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_1_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_1_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_1_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_1_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_1_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_1_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_1_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_2_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_2_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_2_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_2_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_2_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_2_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_2_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_2_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_3_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_3_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_3_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_3_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_3_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_3_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_3_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_3_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_4_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_4_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_4_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_4_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_w_2_4_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_4_5 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_4_6 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    #sell_w_2_4_7 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_exit_0 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_exit_1 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_exit_2 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_exit_3 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)
    sell_exit_4 = DecimalParameter(low=-100, high=100, default=0, space='sell', optimize=True, load=True)

    sell_activation = IntParameter(low=0, high=100000, default=0, space='sell', optimize=True, load=True)
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def informative_pairs(self):

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Momentum Indicators
        # ------------------------------------

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_norm'] = (dataframe['rsi'] - 50) * 2
        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['mfi_norm'] = (dataframe['mfi'] - 50) * 2
        # Williams %R
        dataframe['willr'] = ta.WILLR(dataframe)
        dataframe['willr_norm'] = (dataframe['willr'] + 50) * 2
        # ROC
        dataframe['roc'] = ta.ROC(dataframe)

        dataframe['uo'] = ta.ULTOSC(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        nn_input_0 = [dataframe['cci'], dataframe['rsi_norm'], dataframe['mfi_norm'], dataframe['willr_norm'],
                      dataframe['roc'], dataframe['uo'], 1]

        buy_weight_0_0 = [self.buy_w_0_0_0.value, self.buy_w_0_0_1.value, self.buy_w_0_0_2.value,
                          self.buy_w_0_0_3.value, self.buy_w_0_0_4.value, self.buy_w_0_0_5.value,
                          self.buy_w_0_0_6.value, self.buy_w_0_0_7.value, ]
        buy_weight_0_1 = [self.buy_w_0_1_0.value, self.buy_w_0_1_1.value, self.buy_w_0_1_2.value,
                          self.buy_w_0_1_3.value, self.buy_w_0_1_4.value, self.buy_w_0_1_5.value,
                          self.buy_w_0_1_6.value, self.buy_w_0_1_7.value, ]
        buy_weight_0_2 = [self.buy_w_0_2_0.value, self.buy_w_0_2_1.value, self.buy_w_0_2_2.value,
                          self.buy_w_0_2_3.value, self.buy_w_0_2_4.value, self.buy_w_0_2_5.value,
                          self.buy_w_0_2_6.value, self.buy_w_0_2_7.value, ]
        buy_weight_0_3 = [self.buy_w_0_3_0.value, self.buy_w_0_3_1.value, self.buy_w_0_3_2.value,
                          self.buy_w_0_3_3.value, self.buy_w_0_3_4.value, self.buy_w_0_3_5.value,
                          self.buy_w_0_3_6.value, self.buy_w_0_3_7.value, ]
        buy_weight_0_4 = [self.buy_w_0_4_0.value, self.buy_w_0_4_1.value, self.buy_w_0_4_2.value,
                          self.buy_w_0_4_3.value, self.buy_w_0_4_4.value, self.buy_w_0_4_5.value,
                          self.buy_w_0_4_6.value, self.buy_w_0_4_7.value, ]
        buy_weight_1_0 = [self.buy_w_1_0_0.value, self.buy_w_1_0_1.value, self.buy_w_1_0_2.value,
                          self.buy_w_1_0_3.value, self.buy_w_1_0_4.value]
        buy_weight_1_1 = [self.buy_w_1_1_0.value, self.buy_w_1_1_1.value, self.buy_w_1_1_2.value,
                          self.buy_w_1_1_3.value, self.buy_w_1_1_4.value]
        buy_weight_1_2 = [self.buy_w_1_2_0.value, self.buy_w_1_2_1.value, self.buy_w_1_2_2.value,
                          self.buy_w_1_2_3.value, self.buy_w_1_2_4.value]
        buy_weight_1_3 = [self.buy_w_1_3_0.value, self.buy_w_1_3_1.value, self.buy_w_1_3_2.value,
                          self.buy_w_1_3_3.value, self.buy_w_1_3_4.value]
        buy_weight_1_4 = [self.buy_w_1_4_0.value, self.buy_w_1_4_1.value, self.buy_w_1_4_2.value,
                          self.buy_w_1_4_3.value, self.buy_w_1_4_4.value]
        buy_weight_2_0 = [self.buy_w_2_0_0.value, self.buy_w_2_0_1.value, self.buy_w_2_0_2.value,
                          self.buy_w_2_0_3.value, self.buy_w_2_0_4.value]
        buy_weight_2_1 = [self.buy_w_2_1_0.value, self.buy_w_2_1_1.value, self.buy_w_2_1_2.value,
                          self.buy_w_2_1_3.value, self.buy_w_2_1_4.value]
        buy_weight_2_2 = [self.buy_w_2_2_0.value, self.buy_w_2_2_1.value, self.buy_w_2_2_2.value,
                          self.buy_w_2_2_3.value, self.buy_w_2_2_4.value]
        buy_weight_2_3 = [self.buy_w_2_3_0.value, self.buy_w_2_3_1.value, self.buy_w_2_3_2.value,
                          self.buy_w_2_3_3.value, self.buy_w_2_3_4.value]
        buy_weight_2_4 = [self.buy_w_2_4_0.value, self.buy_w_2_4_1.value, self.buy_w_2_4_2.value,
                          self.buy_w_2_4_3.value, self.buy_w_2_4_4.value]
        #buy_weight_1_0 = [self.buy_w_1_0_0.value, self.buy_w_1_0_1.value, self.buy_w_1_0_2.value,
                          #self.buy_w_1_0_3.value, self.buy_w_1_0_4.value, self.buy_w_1_0_5.value,
                          #self.buy_w_1_0_6.value, self.buy_w_1_0_7.value, ]
        #buy_weight_1_1 = [self.buy_w_1_1_0.value, self.buy_w_1_1_1.value, self.buy_w_1_1_2.value,
                          #self.buy_w_1_1_3.value, self.buy_w_1_1_4.value, self.buy_w_1_1_5.value,
                          #self.buy_w_1_1_6.value, self.buy_w_1_1_7.value, ]
        #buy_weight_1_2 = [self.buy_w_1_2_0.value, self.buy_w_1_2_1.value, self.buy_w_1_2_2.value,
                          #self.buy_w_1_2_3.value, self.buy_w_1_2_4.value, self.buy_w_1_2_5.value,
                          #self.buy_w_1_2_6.value, self.buy_w_1_2_7.value, ]
        #buy_weight_1_3 = [self.buy_w_1_3_0.value, self.buy_w_1_3_1.value, self.buy_w_1_3_2.value,
                          #self.buy_w_1_3_3.value, self.buy_w_1_3_4.value, self.buy_w_1_3_5.value,
                          #self.buy_w_1_3_6.value, self.buy_w_1_3_7.value, ]
        #buy_weight_1_4 = [self.buy_w_1_4_0.value, self.buy_w_1_4_1.value, self.buy_w_1_4_2.value,
                          #self.buy_w_1_4_3.value, self.buy_w_1_4_4.value, self.buy_w_1_4_5.value,
                          #self.buy_w_1_4_6.value, self.buy_w_1_4_7.value, ]
        #buy_weight_2_0 = [self.buy_w_2_0_0.value, self.buy_w_2_0_1.value, self.buy_w_2_0_2.value,
                          #self.buy_w_2_0_3.value, self.buy_w_2_0_4.value, self.buy_w_2_0_5.value,
                          #self.buy_w_2_0_6.value, self.buy_w_2_0_7.value, ]
        #buy_weight_2_1 = [self.buy_w_2_1_0.value, self.buy_w_2_1_1.value, self.buy_w_2_1_2.value,
                          #self.buy_w_2_1_3.value, self.buy_w_2_1_4.value, self.buy_w_2_1_5.value,
                          #self.buy_w_2_1_6.value, self.buy_w_2_1_7.value, ]
        #buy_weight_2_2 = [self.buy_w_2_2_0.value, self.buy_w_2_2_1.value, self.buy_w_2_2_2.value,
                          #self.buy_w_2_2_3.value, self.buy_w_2_2_4.value, self.buy_w_2_2_5.value,
                          #self.buy_w_2_2_6.value, self.buy_w_2_2_7.value, ]
        #buy_weight_2_3 = [self.buy_w_2_3_0.value, self.buy_w_2_3_1.value, self.buy_w_2_3_2.value,
                          #self.buy_w_2_3_3.value, self.buy_w_2_3_4.value, self.buy_w_2_3_5.value,
                          #self.buy_w_2_3_6.value, self.buy_w_2_3_7.value, ]
        #buy_weight_2_4 = [self.buy_w_2_4_0.value, self.buy_w_2_4_1.value, self.buy_w_2_4_2.value,
                          #self.buy_w_2_4_3.value, self.buy_w_2_4_4.value, self.buy_w_2_4_5.value,
                          #self.buy_w_2_4_6.value, self.buy_w_2_4_7.value, ]
        buy_neuron_0_0 = 0
        buy_neuron_0_1 = 0
        buy_neuron_0_2 = 0
        buy_neuron_0_3 = 0
        buy_neuron_0_4 = 0
        buy_neuron_1_0 = 0
        buy_neuron_1_1 = 0
        buy_neuron_1_2 = 0
        buy_neuron_1_3 = 0
        buy_neuron_1_4 = 0
        buy_neuron_2_0 = 0
        buy_neuron_2_1 = 0
        buy_neuron_2_2 = 0
        buy_neuron_2_3 = 0
        buy_neuron_2_4 = 0

        for i in range(len(nn_input_0)):
            buy_neuron_0_0 += nn_input_0[i] * buy_weight_0_0[i]
            buy_neuron_0_1 += nn_input_0[i] * buy_weight_0_1[i]
            buy_neuron_0_2 += nn_input_0[i] * buy_weight_0_2[i]
            buy_neuron_0_3 += nn_input_0[i] * buy_weight_0_3[i]
            buy_neuron_0_4 += nn_input_0[i] * buy_weight_0_4[i]
        nn_input_1 = [buy_neuron_0_0, buy_neuron_0_1, buy_neuron_0_2, buy_neuron_0_3, buy_neuron_0_4]

        for i in range(len(nn_input_1)):
            buy_neuron_1_0 += nn_input_1[i] * buy_weight_1_0[i]
            buy_neuron_1_1 += nn_input_1[i] * buy_weight_1_1[i]
            buy_neuron_1_2 += nn_input_1[i] * buy_weight_1_2[i]
            buy_neuron_1_3 += nn_input_1[i] * buy_weight_1_3[i]
            buy_neuron_1_4 += nn_input_1[i] * buy_weight_1_4[i]
        nn_input_2 = [buy_neuron_1_0, buy_neuron_1_1, buy_neuron_1_2, buy_neuron_1_3, buy_neuron_1_4]

        for i in range(len(nn_input_2)):
            buy_neuron_2_0 += nn_input_2[i] * buy_weight_2_0[i]
            buy_neuron_2_1 += nn_input_2[i] * buy_weight_2_1[i]
            buy_neuron_2_2 += nn_input_2[i] * buy_weight_2_2[i]
            buy_neuron_2_3 += nn_input_2[i] * buy_weight_2_3[i]
            buy_neuron_2_4 += nn_input_2[i] * buy_weight_2_4[i]
        nn_input_3 = [buy_neuron_2_0, buy_neuron_2_1, buy_neuron_2_2, buy_neuron_2_3, buy_neuron_2_4, ]
        buy_exit_neuron = self.buy_exit_0.value * buy_neuron_2_0 + self.buy_exit_1.value * buy_neuron_2_1 + self.buy_exit_2.value * buy_neuron_2_2 + self.buy_exit_3.value * buy_neuron_2_3 + self.buy_exit_4.value * buy_neuron_2_4
        dataframe.loc[
            (
                    (buy_exit_neuron > self.buy_activation.value) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        nn_input_0 = [dataframe['cci'], dataframe['rsi_norm'], dataframe['mfi_norm'], dataframe['willr_norm'],
                    dataframe['roc'], dataframe['uo'], 1]
        sell_weight_0_0 = [self.sell_w_0_0_0.value, self.sell_w_0_0_1.value, self.sell_w_0_0_2.value,
                           self.sell_w_0_0_3.value, self.sell_w_0_0_4.value, self.sell_w_0_0_5.value,
                           self.sell_w_0_0_6.value, self.sell_w_0_0_7.value, ]
        sell_weight_0_1 = [self.sell_w_0_1_0.value, self.sell_w_0_1_1.value, self.sell_w_0_1_2.value,
                           self.sell_w_0_1_3.value, self.sell_w_0_1_4.value, self.sell_w_0_1_5.value,
                           self.sell_w_0_1_6.value, self.sell_w_0_1_7.value, ]
        sell_weight_0_2 = [self.sell_w_0_2_0.value, self.sell_w_0_2_1.value, self.sell_w_0_2_2.value,
                           self.sell_w_0_2_3.value, self.sell_w_0_2_4.value, self.sell_w_0_2_5.value,
                           self.sell_w_0_2_6.value, self.sell_w_0_2_7.value, ]
        sell_weight_0_3 = [self.sell_w_0_3_0.value, self.sell_w_0_3_1.value, self.sell_w_0_3_2.value,
                           self.sell_w_0_3_3.value, self.sell_w_0_3_4.value, self.sell_w_0_3_5.value,
                           self.sell_w_0_3_6.value, self.sell_w_0_3_7.value, ]
        sell_weight_0_4 = [self.sell_w_0_4_0.value, self.sell_w_0_4_1.value, self.sell_w_0_4_2.value,
                           self.sell_w_0_4_3.value, self.sell_w_0_4_4.value, self.sell_w_0_4_5.value,
                           self.sell_w_0_4_6.value, self.sell_w_0_4_7.value, ]
        sell_weight_1_0 = [self.sell_w_1_0_0.value, self.sell_w_1_0_1.value, self.sell_w_1_0_2.value,
                           self.sell_w_1_0_3.value, self.sell_w_1_0_4.value]
        sell_weight_1_1 = [self.sell_w_1_1_0.value, self.sell_w_1_1_1.value, self.sell_w_1_1_2.value,
                           self.sell_w_1_1_3.value, self.sell_w_1_1_4.value]
        sell_weight_1_2 = [self.sell_w_1_2_0.value, self.sell_w_1_2_1.value, self.sell_w_1_2_2.value,
                           self.sell_w_1_2_3.value, self.sell_w_1_2_4.value]
        sell_weight_1_3 = [self.sell_w_1_3_0.value, self.sell_w_1_3_1.value, self.sell_w_1_3_2.value,
                           self.sell_w_1_3_3.value, self.sell_w_1_3_4.value]
        sell_weight_1_4 = [self.sell_w_1_4_0.value, self.sell_w_1_4_1.value, self.sell_w_1_4_2.value,
                           self.sell_w_1_4_3.value, self.sell_w_1_4_4.value]
        sell_weight_2_0 = [self.sell_w_2_0_0.value, self.sell_w_2_0_1.value, self.sell_w_2_0_2.value,
                           self.sell_w_2_0_3.value, self.sell_w_2_0_4.value]
        sell_weight_2_1 = [self.sell_w_2_1_0.value, self.sell_w_2_1_1.value, self.sell_w_2_1_2.value,
                           self.sell_w_2_1_3.value, self.sell_w_2_1_4.value]
        sell_weight_2_2 = [self.sell_w_2_2_0.value, self.sell_w_2_2_1.value, self.sell_w_2_2_2.value,
                           self.sell_w_2_2_3.value, self.sell_w_2_2_4.value]
        sell_weight_2_3 = [self.sell_w_2_3_0.value, self.sell_w_2_3_1.value, self.sell_w_2_3_2.value,
                           self.sell_w_2_3_3.value, self.sell_w_2_3_4.value]
        sell_weight_2_4 = [self.sell_w_2_4_0.value, self.sell_w_2_4_1.value, self.sell_w_2_4_2.value,
                           self.sell_w_2_4_3.value, self.sell_w_2_4_4.value]
        #sell_weight_1_0 = [self.sell_w_1_0_0.value, self.sell_w_1_0_1.value, self.sell_w_1_0_2.value,
                           #self.sell_w_1_0_3.value, self.sell_w_1_0_4.value, self.sell_w_1_0_5.value,
                           #self.sell_w_1_0_6.value, self.sell_w_1_0_7.value, ]
        #sell_weight_1_1 = [self.sell_w_1_1_0.value, self.sell_w_1_1_1.value, self.sell_w_1_1_2.value,
                           #self.sell_w_1_1_3.value, self.sell_w_1_1_4.value, self.sell_w_1_1_5.value,
                           #self.sell_w_1_1_6.value, self.sell_w_1_1_7.value, ]
        #sell_weight_1_2 = [self.sell_w_1_2_0.value, self.sell_w_1_2_1.value, self.sell_w_1_2_2.value,
                           #self.sell_w_1_2_3.value, self.sell_w_1_2_4.value, self.sell_w_1_2_5.value,
                           #self.sell_w_1_2_6.value, self.sell_w_1_2_7.value, ]
        #sell_weight_1_3 = [self.sell_w_1_3_0.value, self.sell_w_1_3_1.value, self.sell_w_1_3_2.value,
                           #self.sell_w_1_3_3.value, self.sell_w_1_3_4.value, self.sell_w_1_3_5.value,
                           #self.sell_w_1_3_6.value, self.sell_w_1_3_7.value, ]
        #sell_weight_1_4 = [self.sell_w_1_4_0.value, self.sell_w_1_4_1.value, self.sell_w_1_4_2.value,
                           #self.sell_w_1_4_3.value, self.sell_w_1_4_4.value, self.sell_w_1_4_5.value,
                           #self.sell_w_1_4_6.value, self.sell_w_1_4_7.value, ]
        #sell_weight_2_0 = [self.sell_w_2_0_0.value, self.sell_w_2_0_1.value, self.sell_w_2_0_2.value,
                           #self.sell_w_2_0_3.value, self.sell_w_2_0_4.value, self.sell_w_2_0_5.value,
                           #self.sell_w_2_0_6.value, self.sell_w_2_0_7.value, ]
        #sell_weight_2_1 = [self.sell_w_2_1_0.value, self.sell_w_2_1_1.value, self.sell_w_2_1_2.value,
                           #self.sell_w_2_1_3.value, self.sell_w_2_1_4.value, self.sell_w_2_1_5.value,
                           #self.sell_w_2_1_6.value, self.sell_w_2_1_7.value, ]
        #sell_weight_2_2 = [self.sell_w_2_2_0.value, self.sell_w_2_2_1.value, self.sell_w_2_2_2.value,
                           #self.sell_w_2_2_3.value, self.sell_w_2_2_4.value, self.sell_w_2_2_5.value,
                           #self.sell_w_2_2_6.value, self.sell_w_2_2_7.value, ]
        #sell_weight_2_3 = [self.sell_w_2_3_0.value, self.sell_w_2_3_1.value, self.sell_w_2_3_2.value,
                           #self.sell_w_2_3_3.value, self.sell_w_2_3_4.value, self.sell_w_2_3_5.value,
                           #self.sell_w_2_3_6.value, self.sell_w_2_3_7.value, ]
        #sell_weight_2_4 = [self.sell_w_2_4_0.value, self.sell_w_2_4_1.value, self.sell_w_2_4_2.value,
                           #self.sell_w_2_4_3.value, self.sell_w_2_4_4.value, self.sell_w_2_4_5.value,
                           #self.sell_w_2_4_6.value, self.sell_w_2_4_7.value, ]
        sell_neuron_0_0 = 0
        sell_neuron_0_1 = 0
        sell_neuron_0_2 = 0
        sell_neuron_0_3 = 0
        sell_neuron_0_4 = 0
        sell_neuron_1_0 = 0
        sell_neuron_1_1 = 0
        sell_neuron_1_2 = 0
        sell_neuron_1_3 = 0
        sell_neuron_1_4 = 0
        sell_neuron_2_0 = 0
        sell_neuron_2_1 = 0
        sell_neuron_2_2 = 0
        sell_neuron_2_3 = 0
        sell_neuron_2_4 = 0
        for i in range(len(nn_input_0)):
            sell_neuron_0_0 += nn_input_0[i] * sell_weight_0_0[i]
            sell_neuron_0_1 += nn_input_0[i] * sell_weight_0_1[i]
            sell_neuron_0_2 += nn_input_0[i] * sell_weight_0_2[i]
            sell_neuron_0_3 += nn_input_0[i] * sell_weight_0_3[i]
            sell_neuron_0_4 += nn_input_0[i] * sell_weight_0_4[i]
        nn_input_1 = [sell_neuron_0_0, sell_neuron_0_1, sell_neuron_0_2, sell_neuron_0_3, sell_neuron_0_4]
        for i in range(len(nn_input_1)):
            sell_neuron_1_0 += nn_input_1[i] * sell_weight_1_0[i]
            sell_neuron_1_1 += nn_input_1[i] * sell_weight_1_1[i]
            sell_neuron_1_2 += nn_input_1[i] * sell_weight_1_2[i]
            sell_neuron_1_3 += nn_input_1[i] * sell_weight_1_3[i]
            sell_neuron_1_4 += nn_input_1[i] * sell_weight_1_4[i]
        nn_input_2 = [sell_neuron_1_0, sell_neuron_1_1, sell_neuron_1_2, sell_neuron_1_3, sell_neuron_1_4]
        for i in range(len(nn_input_2)):
            sell_neuron_2_0 += nn_input_2[i] * sell_weight_2_0[i]
            sell_neuron_2_1 += nn_input_2[i] * sell_weight_2_1[i]
            sell_neuron_2_2 += nn_input_2[i] * sell_weight_2_2[i]
            sell_neuron_2_3 += nn_input_2[i] * sell_weight_2_3[i]
            sell_neuron_2_4 += nn_input_2[i] * sell_weight_2_4[i]
        sell_exit_neuron = self.sell_exit_0.value * sell_neuron_2_0 + self.sell_exit_1.value * sell_neuron_2_1 + self.sell_exit_2.value * sell_neuron_2_2 + self.sell_exit_3.value * sell_neuron_2_3 + self.sell_exit_4.value * sell_neuron_2_4
        dataframe.loc[
            (
                    (sell_exit_neuron < self.sell_activation.value) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
