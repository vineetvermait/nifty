import pandas as pd
import numpy as np
from sklearn import resample

def SMA(
    close: pd.Series,
    sma_window: int,
    min_periods: int = 1,
    **kwargs,
) -> pd.Series:
    return pd.Series(close).rolling(
        sma_window,
        min_periods=min_periods
    ).mean()


def EWM(
    close: pd.Series,
    ewm_window: int,
    min_periods: int = 1,
    **kwargs,
) -> pd.Series:
    return pd.Series(close).ewm(
        span=ewm_window,
        min_periods=min_periods
    ).mean()


def RSI(
    close: np.ndarray,
    rsi_window: int = 14,
    ema: bool = True,
    **kwargs
):
    close_delta = pd.Series(close).diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema:
        ma_up = up.ewm(com=rsi_window - 1, adjust=True,
                       min_periods=rsi_window).mean()
        ma_down = down.ewm(com=rsi_window - 1, adjust=True,
                           min_periods=rsi_window).mean()
    else:
        ma_up = up.rolling(window=rsi_window, adjust=False).mean()
        ma_down = down.rolling(window=rsi_window, adjust=False).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi


def SUPPORT(
    low: pd.Series,
    support_window: int,
    min_periods: int = 1,
    **kwargs
) -> pd.Series:
    return pd.Series(low[:-1]).rolling(
        support_window,
        min_periods=min_periods
    ).min()


def RESISTANCE(
    high: pd.Series,
    resistance_window: int,
    min_periods: int = 1,
    **kwargs
) -> pd.Series:
    return pd.Series(high[:-1]).rolling(
        resistance_window,
        min_periods=min_periods
    ).max()

def _rolling_mean(x, w):
    return pd.Series(x).rolling(w).mean()


def _rolling_std(x, w):
    return pd.Series(x).rolling(w).std()


def UPPER_BAND(
    close,
    bollinger_window,
    std,
    **kwargs,
):
    return _rolling_mean(close, bollinger_window) + (std * _rolling_std(close, bollinger_window))


def LOWER_BAND(
    close,
    bollinger_window,
    std,
    **kwargs,
):
    return _rolling_mean(close, bollinger_window) - (std * _rolling_std(close, bollinger_window))


def MACD(
        close,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        **kwargs
):
    fast_ema = pd.Series(close).ewm(span=fast_period, adjust=False).mean()
    slow_ema = pd.Series(close).ewm(span=slow_period, adjust=False).mean()

    MACD = fast_ema - slow_ema
    signal = MACD.ewm(span=signal_period, adjust=False).mean()
    histogram = MACD - signal
    return MACD, signal, histogram


def mean_reversion(close: pd.Series, window=20, **kwargs):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    signals = pd.DataFrame(close, columns=['close'])
    signals['signal'] = 0.0

    # Calculate z-score
    zscore = (close - rolling_mean) / rolling_std

    # Generate signals based on z-score
    signals['signal'][zscore > 2.0] = -1.0
    signals['signal'][zscore < -2.0] = 1.0

    # Calculate positions
    signals['position'] = signals['signal'].diff()

    return signals['position']


def stochastic_indicators(
    high,
    low,
    close,
    stochastic_window=14,
    smoothed_stochastic_window=3,
    **kwargs
):

    [high, close, low] = list(map(pd.Series, [high, close, low]))
    low_min = low.rolling(stochastic_window).min()
    high_max = high.rolling(stochastic_window).max()
    k_percent = 100 * (close - low_min) / (high_max - low_min)
    d_percent = k_percent.rolling(smoothed_stochastic_window).mean()

    smoothed_k = k_percent.rolling(smoothed_stochastic_window).mean()
    # smoothed_d = d_percent.rolling(m).mean()

    price_trend = np.sign(close.diff())
    oscillator_trend = np.sign(smoothed_k.diff())
    divergence = np.sign(price_trend - oscillator_trend).diff()
    return k_percent, d_percent, divergence

def calculate_ATR(df, period):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    ATR = true_range.rolling(period).mean()
    return ATR


def load(path):
  df = pd.read_csv(path)
  df['datetime'] = pd.to_datetime(df['datetime'])
  df.set_index('datetime', inplace=True)
  df.sort_index(inplace=True)
  return df

def sliding_window(data, window_size):
    """
    Sliding window of size window_size
    """
    data_slided = []
    for i in range(len(data) - window_size + 1):
        data_slided.append(data[i:i + window_size])
    return np.array(data_slided)



def rebalance(_X, _Y, num_bins=10, clip=5):
  data = list(zip(_X, _Y))
  bins = np.linspace(-clip, clip, num_bins + 1)
  y_discrete = np.digitize([y for _, y in data], bins, right=True) - 1
  data_bins = [[] for _ in range(num_bins)]
  for i in range(len(data)):
      data_bins[y_discrete[i]].append(data[i])

  max_bin_size = max(len(data_bin) for data_bin in data_bins)

  for i in range(num_bins):
      if len(data_bins[i]) < max_bin_size:
          data_bins[i] = resample(data_bins[i], replace=True, n_samples=max_bin_size)
      else:
          # Undersample over-represented bins
          data_bins[i] = resample(data_bins[i], replace=False, n_samples=max_bin_size)

  # Combine all the data back together
  balanced_data = [item for sublist in data_bins for item in sublist]
  sX, sY = zip(*balanced_data)
  return np.array(sX), np.array(sY)

