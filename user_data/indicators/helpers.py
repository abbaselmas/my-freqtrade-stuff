
def int_range(low_value: int, high_value: int):
    return [n for n in range(low_value, high_value+1)]


def int_range_step(low_value: int, high_value: int, step: int):
    return [n * step for n in range(low_value, high_value+1)]


def decimal_range(low_value: int, high_value: int, decimals: int):
    low = int(low_value * pow(10, decimals))
    high = int(high_value * pow(10, decimals)) + 1
    return [round(n * pow(0.1, decimals), decimals) for n in range(low, high)]


def get_var_string(var_length: int):
    return f"var-{var_length}"


def get_ott_string(var_length: int, ott_perc: float):
    return f"ott-{var_length}-{ott_perc:.1f}"


def get_ott_up_string(var_length: int, ott_perc: float,  smoothing: int):
    return f"ott-{var_length}-{ott_perc:.1f}-{smoothing}-up"


def get_ott_down_string(var_length: int, ott_perc: float,  smoothing: int):
    return f"ott-{var_length}-{ott_perc:.1f}-{smoothing}-down"


def get_stosk_string(stosk: int, stoskd: int, percent: float):
    return f"stosk-{stosk}-{stoskd}-{percent:.1f}"


def get_stoskd_string(stosk: int, stoskd: int, percent: float):
    return f"sott-{stosk}-{stoskd}-{percent:.1f}"


def get_lag_buy_string(alpha: float):
    return f"lag-buy-{alpha:.3f}"


def get_lag_sell_string(alpha: float):
    return f"lag-sell-{alpha:.3f}"


def get_at_signal_string(period: int, coeff: float):
    return f"at-sig-{period}-{coeff:.1f}"
