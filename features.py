"""
Construcción de variables (features) técnicas, temporales y derivadas,
más la definición del target de clasificación (sube / no sube).
"""
import numpy as np
import pandas as pd
import ta


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Retornos ---
    df["return_1d"] = df["close"].pct_change(1)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))

    # --- Volatilidad ---
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # --- Máximos / mínimos recientes ---
    df["max_20d"] = df["high"].rolling(20).max()
    df["min_20d"] = df["low"].rolling(20).min()
    df["dist_to_max_20d"] = (df["close"] - df["max_20d"]) / df["close"]
    df["dist_to_min_20d"] = (df["close"] - df["min_20d"]) / df["close"]

    # --- Medias móviles (EMA) ---
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["ema_20_50_diff"] = (df["ema_20"] - df["ema_50"]) / df["close"]
    df["ema_50_200_diff"] = (df["ema_50"] - df["ema_200"]) / df["close"]
    df["price_vs_ema20"] = (df["close"] - df["ema_20"]) / df["close"]

    # --- RSI ---
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["rsi_7"] = ta.momentum.rsi(df["close"], window=7)

    # --- MACD ---
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()

    # --- ATR ---
    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["atr_pct"] = df["atr_14"] / df["close"]

    # --- ADX ---
    df["adx_14"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    # --- Momentum ---
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["momentum_20"] = df["close"] - df["close"].shift(20)

    # --- Variables temporales ---
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["is_month_end"] = df.index.is_month_end.astype(int)

    return df


def add_targets(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.DataFrame:
    """
    Target de clasificación: ¿el retorno futuro a `horizon` días supera `threshold`?
    threshold=0.0 -> "sube o baja"; threshold=0.005 -> "sube más de 0.5%", etc.
    """
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["future_return"] = future_return
    df["target_direction"] = (future_return > threshold).astype(int)
    return df
