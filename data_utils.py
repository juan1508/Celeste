"""
Descarga de datos históricos de EUR/USD (u otro ticker) desde Yahoo Finance.
"""
import pandas as pd
import yfinance as yf


def download_data(ticker: str = "EURUSD=X", start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """Descarga OHLCV diario y normaliza el DataFrame."""
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No se recibieron datos para el ticker '{ticker}'. Verifica el símbolo.")

    # yfinance a veces devuelve columnas MultiIndex si se piden varios tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.dropna()
    df.index.name = "date"

    if "volume" not in df.columns:
        df["volume"] = 0  # el par EUR/USD spot no siempre trae volumen real

    return df
