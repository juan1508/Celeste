"""
EUR/USD ML Trading Lab
-----------------------
Pipeline: Datos -> Features -> Walk-Forward Validation -> Comparación de modelos
-> Backtesting con costos -> Recomendación diaria.

Esto es una herramienta educativa/experimental. No es asesoría financiera.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets import ASSET_CATEGORIES, CRYPTO, FOREX_PAIRS, INDICES, STOCKS
from backtest import compute_metrics, simulate_trades
from data_utils import download_data
from features import add_features, add_targets
from models import generate_walk_forward_splits, get_models, train_and_evaluate

st.set_page_config(page_title="Multi-Asset ML Trading Lab", layout="wide", page_icon="📈")

st.title("📈 Multi-Asset ML Trading Lab")
st.caption(
    "Forex, acciones, índices y cripto. Walk-Forward Validation + Clasificación + "
    "Backtesting con costos reales. Proyecto experimental — no es una recomendación de inversión."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("Activo")
    asset_category = st.selectbox("Categoría", ASSET_CATEGORIES)

    pip_value = None  # solo aplica a forex
    if asset_category == "Forex":
        label = st.selectbox("Par de divisas", list(FOREX_PAIRS.keys()))
        ticker, pip_value = FOREX_PAIRS[label]
    elif asset_category == "Acciones":
        label = st.selectbox("Acción", list(STOCKS.keys()))
        ticker = STOCKS[label]
    elif asset_category == "Índices":
        label = st.selectbox("Índice", list(INDICES.keys()))
        ticker = INDICES[label]
    elif asset_category == "Cripto":
        label = st.selectbox("Cripto", list(CRYPTO.keys()))
        ticker = CRYPTO[label]
    else:
        ticker = st.text_input("Ticker de Yahoo Finance (ej. SPY, BTC-USD, EURUSD=X)", "AAPL")

    st.caption(f"Ticker seleccionado: `{ticker}`")

    start_date = st.date_input("Fecha inicio de datos", pd.to_datetime("2015-01-01"))

    st.subheader("Target de clasificación")
    horizon = st.slider("Horizonte de predicción (días hábiles)", 1, 10, 1)
    threshold_target = st.slider(
        "Umbral de retorno para considerar 'sube' (%)", -1.0, 1.0, 0.0, 0.05
    ) / 100

    st.subheader("Walk-Forward Validation")
    first_val_year = st.number_input("Primer año de validación", 2016, 2025, 2019)

    st.subheader("Backtesting")
    threshold_up = st.slider("Prob. mínima para COMPRAR", 0.50, 0.90, 0.55, 0.01)
    threshold_down = st.slider("Prob. máxima para VENDER", 0.10, 0.50, 0.45, 0.01)

    if asset_category == "Forex":
        spread_pips = st.number_input("Spread (pips)", 0.0, 10.0, 1.0)
        slippage_pips = st.number_input("Slippage (pips)", 0.0, 10.0, 0.5)
        cost_per_unit = (spread_pips + slippage_pips) * pip_value
    else:
        spread_slippage_units = st.number_input(
            "Spread + slippage estimados (en unidades de precio, ej. $ por acción/token)",
            0.0, 1000.0, 0.02, step=0.01,
        )
        cost_per_unit = spread_slippage_units

    commission_pct = st.number_input("Comisión por operación (%)", 0.0, 1.0, 0.0, 0.01) / 100

    run_button = st.button("🚀 Ejecutar pipeline completo", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def load_and_prepare(ticker, start_date, horizon, threshold_target):
    df = download_data(ticker, start=str(start_date))
    df = add_features(df)
    df = add_targets(df, horizon=horizon, threshold=threshold_target)
    df = df.dropna()
    return df


if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

# ---------------------------------------------------------------------------
# Ejecutar pipeline
# ---------------------------------------------------------------------------
if run_button:
    try:
        with st.spinner("Descargando datos y calculando variables..."):
            df = load_and_prepare(ticker, start_date, horizon, threshold_target)

        exclude_cols = {"open", "high", "low", "close", "volume", "future_return", "target_direction"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        splits = generate_walk_forward_splits(df, start_year=start_date.year, first_val_year=first_val_year)
        if not splits:
            st.error("No hay suficientes años de datos para generar splits de Walk-Forward. Ajusta las fechas.")
        else:
            models = get_models()
            all_results, all_val_dfs = {}, {}

            progress = st.progress(0.0, text="Entrenando modelos...")
            total_steps = len(models) * len(splits)
            step = 0

            for model_name, model in models.items():
                metrics_per_year, val_dfs = [], []
                for split in splits:
                    metrics, result_df = train_and_evaluate(model, split["train"], split["val"], feature_cols)
                    metrics["year"] = split["val_year"]
                    metrics["type"] = split["type"]
                    metrics_per_year.append(metrics)
                    val_dfs.append(result_df)
                    step += 1
                    progress.progress(step / total_steps, text=f"Entrenando {model_name}...")
                all_results[model_name] = pd.DataFrame(metrics_per_year)
                all_val_dfs[model_name] = pd.concat(val_dfs)
            progress.empty()

            st.session_state.pipeline_results = {
                "df": df,
                "feature_cols": feature_cols,
                "all_results": all_results,
                "all_val_dfs": all_val_dfs,
            }
            st.success("Pipeline ejecutado correctamente ✅")
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

results = st.session_state.pipeline_results

tab_data, tab_models, tab_backtest, tab_reco = st.tabs(
    ["📊 Datos", "🧠 Modelos & Walk-Forward", "💰 Backtesting", "🔮 Recomendación de hoy"]
)

# ---------------------------------------------------------------------------
# Tab: Datos
# ---------------------------------------------------------------------------
with tab_data:
    if results is None:
        st.info("Configura los parámetros en la barra lateral y presiona **Ejecutar pipeline completo**.")
    else:
        df = results["df"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas de datos", len(df))
        c2.metric("Variables (features)", len(results["feature_cols"]))
        c3.metric("Rango de fechas", f"{df.index.min().date()} → {df.index.max().date()}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(width=1.2)))
        fig.update_layout(title="Precio de cierre", height=400, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Últimas filas del dataset (con features)")
        st.dataframe(df.tail(20))

# ---------------------------------------------------------------------------
# Tab: Modelos & Walk-Forward
# ---------------------------------------------------------------------------
with tab_models:
    if results is None:
        st.info("Ejecuta el pipeline primero.")
    else:
        st.subheader("Resultados por modelo y año de validación")
        summary_rows = []
        for model_name, res_df in results["all_results"].items():
            with st.expander(f"📌 {model_name}", expanded=False):
                st.dataframe(res_df.set_index("year"))
            val_only = res_df[res_df["type"] == "validation"]
            if len(val_only) > 0:
                avg = val_only[["accuracy", "precision", "recall", "f1", "auc"]].mean()
                avg["model"] = model_name
                summary_rows.append(avg)

        st.subheader("📊 Comparación (promedio en validación, excluyendo test final)")
        summary_df = pd.DataFrame(summary_rows).set_index("model")
        st.dataframe(summary_df.style.highlight_max(axis=0, color="lightgreen"))

        best_model = summary_df["accuracy"].idxmax()
        st.success(
            f"Mejor modelo según accuracy promedio en validación: **{best_model}** "
            "(recuerda: en trading, accuracy alto no siempre implica rentabilidad — revisa la pestaña Backtesting)."
        )

# ---------------------------------------------------------------------------
# Tab: Backtesting
# ---------------------------------------------------------------------------
with tab_backtest:
    if results is None:
        st.info("Ejecuta el pipeline primero.")
    else:
        model_choice = st.selectbox("Modelo a backtestear", list(results["all_val_dfs"].keys()))
        val_df = results["all_val_dfs"][model_choice]

        bt_df = simulate_trades(
            val_df,
            threshold_up=threshold_up,
            threshold_down=threshold_down,
            cost_per_unit=cost_per_unit,
            commission_pct=commission_pct,
        )
        metrics = compute_metrics(bt_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno total", f"{metrics.get('total_return_pct', 0):.2f}%")
        c2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        c3.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
        c4.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")

        c5, c6, c7 = st.columns(3)
        c5.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        c6.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%")
        c7.metric("N° operaciones", f"{metrics.get('n_trades', 0)}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["equity_curve"], name="Equity curve"))
        fig.update_layout(title=f"Curva de capital — {model_choice} (incluye costos)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "⚠️ Simulación educativa sobre datos históricos de validación / test final del Walk-Forward. "
            "Rendimientos pasados no garantizan resultados futuros."
        )

# ---------------------------------------------------------------------------
# Tab: Recomendación de hoy
# ---------------------------------------------------------------------------
with tab_reco:
    if results is None:
        st.info("Ejecuta el pipeline primero.")
    else:
        model_choice2 = st.selectbox(
            "Modelo para la recomendación", list(results["all_val_dfs"].keys()), key="reco_model"
        )
        val_df2 = results["all_val_dfs"][model_choice2]
        last_row = val_df2.iloc[-1]
        prob_up = float(last_row["prob_up"])

        c1, c2 = st.columns(2)
        c1.metric("Fecha del último dato disponible", str(val_df2.index[-1].date()))
        c2.metric("Probabilidad estimada de que suba", f"{prob_up * 100:.1f}%")

        if prob_up >= threshold_up:
            st.success("Señal actual: 🟢 Posible COMPRA (según el umbral configurado)")
        elif prob_up <= threshold_down:
            st.error("Señal actual: 🔴 Posible VENTA (según el umbral configurado)")
        else:
            st.warning("Señal actual: ⚪ Sin ventaja estadística clara — mejor no operar")

        st.caption(
            "Esto es la salida de un modelo experimental entrenado con datos históricos, no una recomendación "
            "de inversión. El 'último dato disponible' corresponde al cierre del año/periodo de test más reciente "
            "usado en el pipeline, no necesariamente al día de hoy en tiempo real."
        )
