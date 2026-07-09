"""
Catálogo de activos disponibles en el selector de la app.
Los tickers siguen la convención de Yahoo Finance.
"""

# Pares de forex: label -> (ticker, valor del pip aproximado en unidades de precio)
FOREX_PAIRS = {
    "EUR/USD": ("EURUSD=X", 0.0001),
    "GBP/USD": ("GBPUSD=X", 0.0001),
    "USD/JPY": ("USDJPY=X", 0.01),
    "USD/CHF": ("USDCHF=X", 0.0001),
    "USD/CAD": ("USDCAD=X", 0.0001),
    "AUD/USD": ("AUDUSD=X", 0.0001),
    "NZD/USD": ("NZDUSD=X", 0.0001),
    "EUR/GBP": ("EURGBP=X", 0.0001),
    "EUR/JPY": ("EURJPY=X", 0.01),
    "GBP/JPY": ("GBPJPY=X", 0.01),
    "USD/COP": ("COP=X", 1.0),
    "USD/MXN": ("MXN=X", 0.01),
}

# Acciones populares: label -> ticker
STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Alphabet / Google (GOOGL)": "GOOGL",
    "Nvidia (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Ecopetrol ADR (EC)": "EC",
    "Bancolombia ADR (CIB)": "CIB",
    "Avianca Holdings (AVHOQ)": "AVHOQ",
}

# Índices bursátiles: label -> ticker
INDICES = {
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
    "Dow Jones (^DJI)": "^DJI",
    "COLCAP (proxy iShares COLCAP: ICOL)": "ICOL",
}

# Criptomonedas: label -> ticker
CRYPTO = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
}

ASSET_CATEGORIES = ["Forex", "Acciones", "Índices", "Cripto", "Personalizado (escribir ticker)"]
