# 📈 EUR/USD ML Trading Lab

App en Streamlit para experimentar con **clasificación de dirección de precio** (sube/baja) sobre
EUR/USD (o cualquier ticker de Yahoo Finance), usando **Walk-Forward Validation**, comparación de
varios modelos y **backtesting con costos reales** (spread, slippage, comisión).

> ⚠️ Proyecto educativo/experimental. No es asesoría financiera ni una garantía de rentabilidad.

## Estructura del proyecto

```
eurusd-trading-app/
├── app.py            # App de Streamlit (UI)
├── data_utils.py      # Descarga de datos (Yahoo Finance)
├── features.py        # Ingeniería de variables + target de clasificación
├── models.py           # Modelos + generación de splits Walk-Forward
├── backtest.py         # Simulación de operaciones + métricas (Sharpe, Sortino, etc.)
├── requirements.txt
└── README.md
```

## Metodología

1. **Walk-Forward Validation**: en vez de un solo train/test split, se entrena con todo el
   histórico disponible hasta el año N-1 y se valida sobre el año N, avanzando año a año hasta
   llegar al año más reciente (que se usa como test final).
2. **Clasificación, no regresión**: el modelo no predice el precio exacto, predice si el retorno
   futuro (a `horizon` días) supera un umbral configurable.
3. **+30 variables**: precio, retornos, volatilidad, RSI, MACD, EMAs (20/50/200), Bollinger, ATR,
   ADX, momentum y variables temporales (día de la semana, mes, etc.).
4. **Modelos comparados**: Regresión Logística (baseline), Random Forest, XGBoost, LightGBM.
5. **Backtesting con costos**: cada señal de compra/venta descuenta spread + slippage (en pips) y
   comisión (%), y se calculan Sharpe Ratio, Sortino Ratio, Max Drawdown, Profit Factor y Win Rate.

## Instalación local

```bash
git clone https://github.com/TU_USUARIO/eurusd-trading-app.git
cd eurusd-trading-app
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud (igual que tus otros proyectos)

1. Sube esta carpeta a un repositorio de GitHub (ver comandos abajo).
2. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
3. **New app** → selecciona el repo → branch `main` → archivo principal `app.py`.
4. Deploy. Streamlit Cloud instala automáticamente lo que está en `requirements.txt`.

### Subir a GitHub desde cero

```bash
cd eurusd-trading-app
git init
git add .
git commit -m "Primera versión: EUR/USD ML Trading Lab"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/eurusd-trading-app.git
git push -u origin main
```

## Próximos pasos sugeridos (Fase 2 del roadmap)

- Agregar ARIMA/Prophet como baseline de series de tiempo puro.
- Agregar LSTM / Transformer (requiere TensorFlow o PyTorch — más pesado para Streamlit Cloud free tier).
- Automatizar reentrenamiento diario (ej. GitHub Actions + guardar resultados en el repo o en una BD).
- Optimización de tamaño de posición (Kelly Criterion fraccional, por ejemplo).
- Guardar histórico de recomendaciones diarias para evaluar el modelo en producción con el tiempo.
