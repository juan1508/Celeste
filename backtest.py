"""
Simulación de operaciones (backtesting) sobre las predicciones del modelo,
incluyendo spread, slippage y comisión, y métricas de desempeño estándar.
"""
import numpy as np
import pandas as pd


def simulate_trades(
    df,
    prob_col="prob_up",
    threshold_up=0.55,
    threshold_down=0.45,
    cost_per_unit=0.00015,
    commission_pct=0.0,
    position_size=1.0,
):
    """
    cost_per_unit: costo de spread + slippage expresado en las mismas unidades que el
    precio (ej. para forex: (spread_pips + slippage_pips) * pip_value; para acciones:
    spread + slippage estimados en $ por acción). Se descuenta como % del precio de cierre.
    commission_pct: comisión por operación como fracción del valor operado (ej. 0.001 = 0.1%).
    """
    df = df.copy()

    df["signal"] = 0
    df.loc[df[prob_col] >= threshold_up, "signal"] = 1     # compra
    df.loc[df[prob_col] <= threshold_down, "signal"] = -1   # venta

    df["strategy_return"] = df["signal"] * df["future_return"] * position_size

    trade_mask = df["signal"] != 0
    df.loc[trade_mask, "strategy_return"] -= cost_per_unit / df.loc[trade_mask, "close"]
    df.loc[trade_mask, "strategy_return"] -= commission_pct

    df["equity_curve"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    return df


def compute_metrics(df, return_col="strategy_return"):
    returns = df[return_col].dropna()
    if len(returns) == 0:
        return {}

    total_return = df["equity_curve"].iloc[-1] - 1
    mean_ret, std_ret = returns.mean(), returns.std()
    downside = returns[returns < 0].std()

    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret and std_ret > 0 else np.nan
    sortino = (mean_ret / downside) * np.sqrt(252) if downside and downside > 0 else np.nan

    running_max = df["equity_curve"].cummax()
    drawdown = (df["equity_curve"] - running_max) / running_max
    max_dd = drawdown.min()

    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    profit_factor = wins / losses if losses > 0 else np.nan

    win_rate = (returns[df["signal"] != 0] > 0).mean() if (df["signal"] != 0).any() else np.nan
    n_trades = int((df["signal"] != 0).sum())

    return {
        "total_return_pct": total_return * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd * 100,
        "profit_factor": profit_factor,
        "win_rate_pct": win_rate * 100 if not np.isnan(win_rate) else np.nan,
        "n_trades": n_trades,
    }
