"""
╔══════════════════════════════════════════════════════════════════╗
║         CRYPTO MARKET REGIME ANALYZER — HMM Edition            ║
║         Quantitative Finance × Machine Learning                ║
║         Hidden Markov Model para Detección de Regímenes        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Regime Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    
    .main-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        color: #8b949e;
        font-size: 0.9rem;
    }
    
    .regime-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .regime-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .metric-box {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .signal-panel {
        background: rgba(22, 27, 34, 0.95);
        border-left: 4px solid;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    
    .stSelectbox label, .stSlider label, .stDateInput label {
        font-family: 'Inter', sans-serif;
        color: #c9d1d9 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────
CRYPTO_PAIRS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Polkadot": "DOT-USD",
    "Chainlink": "LINK-USD",
    "Avalanche": "AVAX-USD",
    "Cardano": "ADA-USD",
    "XRP": "XRP-USD",
    "Hedera": "HBAR-USD",
    "Bittensor": "TAO-USD",
}

TIMEFRAMES = {
    "1 Hora": "1h",
    "4 Horas": "4h",
    "1 Día": "1d",
}

# Regime color palettes (indexed by n_regimes)
REGIME_COLORS = {
    3: ["#ff4444", "#ffaa00", "#00cc66"],
    4: ["#ff4444", "#ff8800", "#00cc66", "#00aaff"],
    5: ["#ff2222", "#ff6644", "#ffaa00", "#00cc66", "#00aaff"],
    6: ["#ff2222", "#ff6644", "#ffaa00", "#88cc00", "#00cc66", "#00aaff"],
    7: ["#ff2222", "#ff4444", "#ff8844", "#ffaa00", "#88cc00", "#00cc66", "#00aaff"],
}

REGIME_NAMES_MAP = {
    3: ["🐻 Bear", "➡️ Neutral", "🐂 Bull"],
    4: ["🐻 Deep Bear", "📉 Bear", "📈 Bull", "🚀 Strong Bull"],
    5: ["💀 Crash", "🐻 Bear", "➡️ Neutral", "📈 Bull", "🚀 Euphoria"],
    6: ["💀 Crash", "🐻 Bear", "📉 Weak Bear", "📈 Weak Bull", "🐂 Bull", "🚀 Euphoria"],
    7: ["💀 Crash", "🐻 Deep Bear", "📉 Bear", "➡️ Neutral", "📈 Bull", "🐂 Strong Bull", "🚀 Euphoria"],
}

# Binance symbol mapping for real-time prices
BINANCE_SYMBOLS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "DOT-USD": "DOTUSDT",
    "LINK-USD": "LINKUSDT",
    "AVAX-USD": "AVAXUSDT",
    "ADA-USD": "ADAUSDT",
    "XRP-USD": "XRPUSDT",
    "HBAR-USD": "HBARUSDT",
    "TAO-USD": "TAOUSDT",
}


def get_realtime_price(ticker: str) -> dict:
    """Fetch real-time price from Binance public API (no auth needed)."""
    symbol = BINANCE_SYMBOLS.get(ticker)
    if not symbol:
        return None
    try:
        # Get 24h ticker for price + change
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "price": float(data["lastPrice"]),
                "change_pct": float(data["priceChangePercent"]),
                "high_24h": float(data["highPrice"]),
                "low_24h": float(data["lowPrice"]),
                "volume_24h": float(data["quoteVolume"]),
                "source": "Binance (real-time)",
            }
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def download_data(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()


def compute_features(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    """Compute log returns, RSI, and relative volume."""
    data = df.copy()
    
    # Log returns
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    
    # RSI
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    
    # Normalize RSI to [-1, 1] for HMM
    data["RSI_norm"] = (data["RSI"] - 50) / 50
    
    # EMA 20
    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
    
    # Relative Volume (current / 20-period SMA)
    vol_sma = data["Volume"].rolling(window=20, min_periods=5).mean()
    data["rel_volume"] = (data["Volume"] / vol_sma.replace(0, np.nan)) - 1
    data["rel_volume"] = data["rel_volume"].clip(-3, 3)  # Clip outliers
    
    # Volatility (20-period rolling std of returns)
    data["volatility"] = data["log_return"].rolling(window=20, min_periods=5).std()
    vol_std = data["volatility"].std()
    vol_std = vol_std if vol_std > 1e-8 else 1e-8
    data["vol_norm"] = (data["volatility"] - data["volatility"].mean()) / vol_std
    data["vol_norm"] = data["vol_norm"].clip(-3, 3)
    
    # Support & Resistance (rolling 20-period low/high)
    data["support"] = data["Low"].rolling(window=20, min_periods=5).min()
    data["resistance"] = data["High"].rolling(window=20, min_periods=5).max()
    
    data.dropna(inplace=True)
    return data


def fit_hmm(features: np.ndarray, n_regimes: int, n_iter: int = 200, n_fits: int = 10) -> tuple:
    """
    Fit Gaussian HMM with multiple random initializations to avoid local optima.
    Returns the best model and decoded states.
    """
    best_score = -np.inf
    best_model = None
    
    for seed in range(n_fits):
        try:
            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=n_iter,
                tol=1e-4,
                random_state=seed * 42,
                init_params="stmc",
                min_covar=1e-3,  # Regularization to prevent singular covariance
                verbose=False,
            )
            model.fit(features)
            score = model.score(features)
            
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue
    
    if best_model is None:
        raise ValueError("No se pudo ajustar el modelo HMM")
    
    states = best_model.predict(features)
    posteriors = best_model.predict_proba(features)
    
    return best_model, states, posteriors, best_score


def sort_regimes_by_return(model, states: np.ndarray, n_regimes: int) -> tuple:
    """
    Reorder regimes so that index 0 = most bearish and index N = most bullish.
    Based on mean return of each regime.
    """
    mean_returns = model.means_[:, 0]  # First feature is log_return
    order = np.argsort(mean_returns)  # Ascending: bear → bull
    
    # Create mapping
    mapping = {old: new for new, old in enumerate(order)}
    sorted_states = np.array([mapping[s] for s in states])
    
    return sorted_states, order


def get_regime_stats(df: pd.DataFrame, states: np.ndarray, n_regimes: int) -> pd.DataFrame:
    """Compute statistics for each regime."""
    stats_list = []
    for i in range(n_regimes):
        mask = states == i
        if mask.sum() == 0:
            continue
        regime_data = df[mask]
        stats_list.append({
            "Régimen": i,
            "Períodos": int(mask.sum()),
            "% Tiempo": f"{100 * mask.sum() / len(states):.1f}%",
            "Retorno Medio": f"{regime_data['log_return'].mean() * 100:.3f}%",
            "Volatilidad": f"{regime_data['log_return'].std() * 100:.3f}%",
            "RSI Medio": f"{regime_data['RSI'].mean():.1f}",
            "Vol. Relativo": f"{regime_data['rel_volume'].mean():.2f}",
        })
    return pd.DataFrame(stats_list)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🔬</span>
        <h2 style="font-family: 'JetBrains Mono', monospace; 
                    background: linear-gradient(90deg, #00d4ff, #7b2ff7);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin: 0.5rem 0 0.2rem;">
            CONFIGURACIÓN
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Asset Selection
    st.markdown("##### 📊 Activo")
    selected_asset = st.selectbox(
        "Seleccionar criptoactivo",
        options=list(CRYPTO_PAIRS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    ticker = CRYPTO_PAIRS[selected_asset]
    
    # Timeframe
    st.markdown("##### ⏱️ Timeframe")
    selected_tf = st.selectbox(
        "Seleccionar timeframe",
        options=list(TIMEFRAMES.keys()),
        index=2,
        label_visibility="collapsed"
    )
    interval = TIMEFRAMES[selected_tf]
    
    # Date Range
    st.markdown("##### 📅 Rango de Fechas")
    
    # Max periods depend on interval
    if interval == "1h":
        default_days = 60
        max_days = 729  # yfinance limit for hourly
    elif interval == "4h":
        default_days = 120
        max_days = 729
    else:
        default_days = 365
        max_days = 3650
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Desde",
            value=datetime.now() - timedelta(days=default_days),
            max_value=datetime.now(),
        )
    with col_d2:
        end_date = st.date_input(
            "Hasta",
            value=datetime.now(),
            max_value=datetime.now(),
        )
    
    st.markdown("---")
    
    # HMM Configuration
    st.markdown("##### 🧠 Modelo HMM")
    
    n_regimes = st.slider(
        "Número de Regímenes",
        min_value=3,
        max_value=7,
        value=4,
        help="Más regímenes = más granularidad, pero riesgo de sobreajuste"
    )
    
    n_fits = st.slider(
        "Inicializaciones (robustez)",
        min_value=5,
        max_value=30,
        value=10,
        help="Más = mejor modelo pero más lento"
    )
    
    st.markdown("---")
    
    # Features toggle
    st.markdown("##### 🔧 Features del Modelo")
    use_rsi = st.checkbox("RSI Normalizado", value=True)
    use_volume = st.checkbox("Volumen Relativo", value=True)
    use_volatility = st.checkbox("Volatilidad", value=True)
    
    st.markdown("---")
    
    # Run button
    run_analysis = st.button(
        "🚀 EJECUTAR ANÁLISIS",
        use_container_width=True,
        type="primary"
    )
    
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0; opacity: 0.5;">
        <small>HMM Regime Analyzer v1.0<br/>
        Powered by hmmlearn + yfinance</small>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 CRYPTO REGIME ANALYZER</h1>
    <p>Detección de Regímenes de Mercado con Hidden Markov Models</p>
</div>
""", unsafe_allow_html=True)

if run_analysis:
    # ─── STEP 1: Download Data ───
    with st.spinner(f"📡 Descargando datos de {selected_asset}..."):
        df_raw = download_data(
            ticker, 
            interval, 
            start=str(start_date), 
            end=str(end_date)
        )
    
    if df_raw.empty or len(df_raw) < 50:
        st.error("⚠️ No hay suficientes datos para el rango seleccionado. Probá con un rango más amplio o un timeframe diferente.")
        st.stop()
    
    # ─── STEP 2: Compute Features ───
    with st.spinner("🔧 Calculando features (retornos, RSI, volumen)..."):
        df = compute_features(df_raw)
    
    if len(df) < 50:
        st.error("⚠️ Datos insuficientes después de calcular indicadores. Necesitás al menos 50 períodos.")
        st.stop()
    
    # Build feature matrix
    feature_cols = ["log_return"]
    if use_rsi:
        feature_cols.append("RSI_norm")
    if use_volume:
        feature_cols.append("rel_volume")
    if use_volatility:
        feature_cols.append("vol_norm")
    
    features = df[feature_cols].values
    
    # ─── STEP 3: Fit HMM ───
    with st.spinner(f"🧠 Entrenando HMM con {n_regimes} regímenes ({n_fits} inicializaciones)..."):
        try:
            model, raw_states, posteriors, log_likelihood = fit_hmm(
                features, n_regimes, n_fits=n_fits
            )
            # Sort regimes: bear → bull
            states, regime_order = sort_regimes_by_return(model, raw_states, n_regimes)
            # Reorder posteriors
            posteriors = posteriors[:, regime_order]
        except Exception as e:
            st.error(f"❌ Error entrenando el modelo: {e}")
            st.stop()
    
    df["regime"] = states
    
    # Get regime names and colors
    regime_names = REGIME_NAMES_MAP.get(n_regimes, [f"Régimen {i}" for i in range(n_regimes)])
    colors = REGIME_COLORS.get(n_regimes, ["#888"] * n_regimes)
    
    # ─── STEP 4: Current Regime Panel ───
    current_regime = states[-1]
    current_confidence = posteriors[-1, current_regime]
    current_name = regime_names[current_regime]
    current_color = colors[current_regime]
    
    # Fetch real-time price from Binance
    rt = get_realtime_price(ticker)
    realtime_price = rt["price"] if rt else None
    yf_price = df['Close'].iloc[-1]
    display_price = realtime_price if realtime_price else yf_price
    price_source = rt["source"] if rt else "yfinance (con delay)"
    
    # Show delay warning if significant difference
    if realtime_price and abs(realtime_price - yf_price) / yf_price > 0.005:
        price_diff_pct = ((realtime_price - yf_price) / yf_price) * 100
        st.markdown(f"""
        <div style="background: rgba(255,170,0,0.1); border: 1px solid rgba(255,170,0,0.3); 
                    border-radius: 8px; padding: 0.5rem 1rem; margin-bottom: 0.5rem;">
            <p style="color: #ffaa00; margin: 0; font-size: 0.8rem;">
                ⚡ <b>Precio en tiempo real:</b> ${realtime_price:,.4f} (Binance) | 
                Última vela cerrada: ${yf_price:,.4f} (yfinance) | 
                Diferencia: {price_diff_pct:+.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if rt:
            chg_color = "#00cc66" if rt["change_pct"] >= 0 else "#ff4444"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Precio Real-Time</div>
                <div class="metric-value" style="color: #c9d1d9;">
                    ${realtime_price:,.4f}
                </div>
                <div style="color: {chg_color}; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
                    {rt['change_pct']:+.2f}% (24h)
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Precio Actual</div>
                <div class="metric-value" style="color: #c9d1d9;">
                    ${yf_price:,.2f}
                </div>
                <div style="color: #8b949e; font-size: 0.65rem;">⚠️ yfinance (delay)</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        ret_24h = df["log_return"].iloc[-1] * 100
        ret_color = "#00cc66" if ret_24h >= 0 else "#ff4444"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Último Retorno</div>
            <div class="metric-value" style="color: {ret_color};">
                {ret_24h:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">RSI Actual</div>
            <div class="metric-value" style="color: {'#ff4444' if df['RSI'].iloc[-1] > 70 else '#00cc66' if df['RSI'].iloc[-1] < 30 else '#ffaa00'};">
                {df['RSI'].iloc[-1]:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Régimen Actual</div>
            <div class="metric-value" style="color: {current_color};">
                {current_name}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        conf_color = "#00cc66" if current_confidence > 0.7 else "#ffaa00" if current_confidence > 0.4 else "#ff4444"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Confianza</div>
            <div class="metric-value" style="color: {conf_color};">
                {current_confidence * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ─── STEP 5: Main Price Chart with Regime Colors ───
    st.markdown("### 📈 Precio + Regímenes Detectados")
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.20, 0.20, 0.15],
        subplot_titles=["", "RSI", "Probabilidades de Régimen", "Volumen Relativo"]
    )
    
    # Price chart colored by regime
    for i in range(n_regimes):
        mask = df["regime"] == i
        if mask.sum() == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=df.index[mask],
                y=df["Close"][mask],
                mode="markers",
                marker=dict(color=colors[i], size=3, opacity=0.8),
                name=regime_names[i],
                legendgroup=f"regime_{i}",
                showlegend=True,
                hovertemplate=f"<b>{regime_names[i]}</b><br>" +
                             "Precio: $%{y:,.2f}<br>" +
                             "Fecha: %{x}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Add price line (subtle)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(color="rgba(201,209,217,0.15)", width=1),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=1
    )
    
    # RSI subplot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            mode="lines",
            line=dict(color="#7b2ff7", width=1.5),
            name="RSI",
            showlegend=False,
        ),
        row=2, col=1
    )
    # RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.4)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,204,102,0.4)", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(139,148,158,0.3)", row=2, col=1)
    
    # Regime probabilities (stacked area)
    def hex_to_rgba(hex_color, alpha=0.4):
        """Convert hex color to rgba string."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    
    for i in range(n_regimes):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=posteriors[:, i],
                mode="lines",
                fill="tonexty" if i > 0 else "tozeroy",
                line=dict(width=0.5, color=colors[i]),
                fillcolor=hex_to_rgba(colors[i], 0.4),
                name=regime_names[i],
                legendgroup=f"regime_{i}",
                showlegend=False,
                hovertemplate=f"{regime_names[i]}: " + "%{y:.1%}<extra></extra>"
            ),
            row=3, col=1
        )
    
    # Volume subplot
    vol_colors = [colors[s] for s in states]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["rel_volume"],
            marker=dict(color=vol_colors, opacity=0.6),
            name="Vol. Relativo",
            showlegend=False,
        ),
        row=4, col=1
    )
    
    # Layout
    fig.update_layout(
        height=900,
        template="plotly_dark",
        paper_bgcolor="rgba(10,10,15,0)",
        plot_bgcolor="rgba(13,17,23,0.8)",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor="rgba(48,54,61,0.6)",
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="x unified",
    )
    
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1, gridcolor="rgba(48,54,61,0.3)")
    fig.update_yaxes(title_text="RSI", row=2, col=1, gridcolor="rgba(48,54,61,0.3)", range=[0, 100])
    fig.update_yaxes(title_text="Prob.", row=3, col=1, gridcolor="rgba(48,54,61,0.3)", range=[0, 1])
    fig.update_yaxes(title_text="Vol. Rel.", row=4, col=1, gridcolor="rgba(48,54,61,0.3)")
    fig.update_xaxes(gridcolor="rgba(48,54,61,0.2)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ─── STEP 6: Regime Probability Panel ───
    st.markdown("### 🎯 Probabilidad Actual del Régimen")
    
    prob_cols = st.columns(n_regimes)
    for i, col in enumerate(prob_cols):
        prob = posteriors[-1, i]
        is_current = (i == current_regime)
        border_style = f"border: 2px solid {colors[i]};" if is_current else f"border: 1px solid rgba(48,54,61,0.6);"
        glow = f"box-shadow: 0 0 15px {colors[i]}40;" if is_current else ""
        
        with col:
            st.markdown(f"""
            <div style="background: rgba(22,27,34,0.9); {border_style} {glow}
                        border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="font-size: 0.8rem; color: {colors[i]}; font-weight: 700;
                            font-family: 'JetBrains Mono', monospace; margin-bottom: 0.5rem;">
                    {regime_names[i]}
                </div>
                <div style="font-size: 2rem; font-weight: 700; 
                            font-family: 'JetBrains Mono', monospace; color: {colors[i]};">
                    {prob * 100:.1f}%
                </div>
                <div style="background: rgba(255,255,255,0.05); border-radius: 10px; 
                            height: 6px; margin-top: 0.5rem; overflow: hidden;">
                    <div style="background: {colors[i]}; height: 100%; width: {prob * 100}%; 
                                border-radius: 10px; transition: width 0.5s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ─── STEP 7: Regime Statistics Table ───
    st.markdown("### 📊 Estadísticas por Régimen")
    
    stats_df = get_regime_stats(df, states, n_regimes)
    
    # Add regime names
    stats_df["Nombre"] = [regime_names[int(r)] for r in stats_df["Régimen"]]
    stats_df = stats_df[["Régimen", "Nombre", "Períodos", "% Tiempo", "Retorno Medio", "Volatilidad", "RSI Medio", "Vol. Relativo"]]
    
    st.dataframe(
        stats_df,
        hide_index=True,
        use_container_width=True,
    )
    
    # ─── STEP 8: Transition Matrix ───
    st.markdown("### 🔄 Matriz de Transición entre Regímenes")
    st.caption("Probabilidad de pasar de un régimen a otro en el siguiente período")
    
    # Reorder transition matrix
    trans_matrix = model.transmat_[regime_order][:, regime_order]
    
    fig_trans = go.Figure(data=go.Heatmap(
        z=trans_matrix * 100,
        x=[regime_names[i] for i in range(n_regimes)],
        y=[regime_names[i] for i in range(n_regimes)],
        colorscale="Viridis",
        text=[[f"{val*100:.1f}%" for val in row] for row in trans_matrix],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        hovertemplate="De: %{y}<br>A: %{x}<br>Prob: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Prob. (%)", ticksuffix="%"),
    ))
    
    fig_trans.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(10,10,15,0)",
        plot_bgcolor="rgba(13,17,23,0.8)",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        xaxis_title="Régimen Destino",
        yaxis_title="Régimen Origen",
        margin=dict(l=120, r=20, t=20, b=80),
    )
    
    st.plotly_chart(fig_trans, use_container_width=True)
    
    # ─── STEP 9: Trading Signal Panel ───
    st.markdown("### 🚦 Señal para Trading")
    
    # Determine signal based on regime
    if current_regime <= n_regimes // 3:
        signal = "🔴 ROJO — No abrir LONGS. Considerar SHORT o cash."
        signal_color = "#ff4444"
        signal_detail = "El mercado está en régimen bajista. Alta probabilidad de continuación de caída."
    elif current_regime >= n_regimes - n_regimes // 3:
        signal = "🟢 VERDE — Condiciones favorables para LONGS."
        signal_color = "#00cc66"
        signal_detail = "El mercado está en régimen alcista. Momentum a favor."
    else:
        signal = "🟡 AMARILLO — Precaución. Mercado en transición."
        signal_color = "#ffaa00"
        signal_detail = "Régimen neutral/mixto. Reducir posiciones o esperar confirmación."
    
    # Confidence qualifier
    if current_confidence < 0.4:
        confidence_note = "⚠️ **Baja confianza** — El modelo no está seguro del régimen actual. Esperar más datos."
    elif current_confidence < 0.7:
        confidence_note = "📊 **Confianza moderada** — Señal válida pero con cautela."
    else:
        confidence_note = "✅ **Alta confianza** — El modelo está muy seguro del régimen actual."
    
    st.markdown(f"""
    <div class="signal-panel" style="border-color: {signal_color};">
        <h3 style="color: {signal_color}; margin: 0 0 0.5rem 0; font-family: 'JetBrains Mono', monospace;">
            {signal}
        </h3>
        <p style="color: #c9d1d9; margin: 0.3rem 0;">{signal_detail}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(confidence_note)
    
    # ─── STEP 9B: Entry Signals Panel ───
    st.markdown("### 🎯 Posibles Ingresos")
    st.caption("Señales basadas en régimen HMM + RSI + EMA20 + Soporte/Resistencia + Transiciones")
    if rt:
        st.caption(f"⚡ Usando precio en tiempo real: ${display_price:,.4f} (Binance)")
    
    price = display_price  # Use real-time price if available
    rsi = df["RSI"].iloc[-1]
    ema20 = df["EMA20"].iloc[-1]
    support = df["support"].iloc[-1]
    resistance = df["resistance"].iloc[-1]
    rel_vol = df["rel_volume"].iloc[-1]
    current_vol = df["volatility"].iloc[-1]
    
    # Detect regime transitions (last 5 periods)
    recent_states = states[-5:] if len(states) >= 5 else states
    regime_changed = len(set(recent_states)) > 1
    prev_regime = states[-2] if len(states) >= 2 else current_regime
    
    # Transition probabilities for next period
    trans_probs = trans_matrix[current_regime]
    prob_up = sum(trans_probs[i] for i in range(n_regimes) if i > current_regime)
    prob_down = sum(trans_probs[i] for i in range(n_regimes) if i < current_regime)
    prob_stay = trans_probs[current_regime]
    
    # ── Generate Signals ──
    signals = []
    
    # === LONG SIGNALS ===
    long_score = 0
    long_reasons = []
    long_warnings = []
    
    # 1. Regime favorable for longs
    if current_regime >= n_regimes // 2:
        long_score += 2
        long_reasons.append(f"Régimen {regime_names[current_regime]} favorable")
    elif current_regime == n_regimes // 2 - 1:
        long_score += 1
        long_reasons.append("Régimen neutral — posible transición")
    
    # 2. RSI oversold (buying opportunity)
    if rsi < 30:
        long_score += 2
        long_reasons.append(f"RSI sobrevendido ({rsi:.1f}) — zona de rebote")
    elif rsi < 45:
        long_score += 1
        long_reasons.append(f"RSI bajo ({rsi:.1f}) — espacio para subir")
    elif rsi > 70:
        long_score -= 2
        long_warnings.append(f"RSI sobrecomprado ({rsi:.1f}) — NO comprar")
    
    # 3. Price vs EMA20
    if price > ema20:
        long_score += 1
        long_reasons.append(f"Precio encima de EMA20 (${ema20:,.2f})")
    else:
        pct_below = ((ema20 - price) / ema20) * 100
        if pct_below > 5:
            long_score += 1
            long_reasons.append(f"Precio {pct_below:.1f}% debajo de EMA20 — posible mean reversion")
        else:
            long_warnings.append(f"Precio debajo de EMA20 (${ema20:,.2f})")
    
    # 4. Near support
    dist_to_support = ((price - support) / price) * 100
    if dist_to_support < 2:
        long_score += 2
        long_reasons.append(f"Cerca de soporte ${support:,.2f} ({dist_to_support:.1f}%)")
    elif dist_to_support < 5:
        long_score += 1
        long_reasons.append(f"Soporte cercano en ${support:,.2f}")
    
    # 5. Regime transitioning upward
    if regime_changed and current_regime > prev_regime:
        long_score += 2
        long_reasons.append(f"Transición alcista: {regime_names[prev_regime]} → {regime_names[current_regime]}")
    
    # 6. Probability of moving to higher regime
    if prob_up > 0.15:
        long_score += 1
        long_reasons.append(f"Prob. transición alcista: {prob_up:.0%}")
    
    # 7. Volume confirmation
    if rel_vol > 0.3:
        long_score += 1
        long_reasons.append(f"Volumen alto ({rel_vol:+.1f}x) — confirma movimiento")
    
    # === SHORT SIGNALS ===
    short_score = 0
    short_reasons = []
    short_warnings = []
    
    # 1. Regime favorable for shorts
    if current_regime <= n_regimes // 3:
        short_score += 2
        short_reasons.append(f"Régimen {regime_names[current_regime]} — tendencia bajista")
    elif current_regime == n_regimes // 3 + 1:
        short_score += 1
        short_reasons.append("Régimen débil — posible continuación bajista")
    
    # 2. RSI overbought (shorting opportunity)
    if rsi > 70:
        short_score += 2
        short_reasons.append(f"RSI sobrecomprado ({rsi:.1f}) — zona de reversión")
    elif rsi > 55:
        short_score += 1
        short_reasons.append(f"RSI elevado ({rsi:.1f}) — presión vendedora probable")
    elif rsi < 30:
        short_score -= 2
        short_warnings.append(f"RSI sobrevendido ({rsi:.1f}) — NO shortear")
    
    # 3. Price vs EMA20
    if price < ema20:
        short_score += 1
        short_reasons.append(f"Precio debajo de EMA20 — tendencia bajista")
    else:
        pct_above = ((price - ema20) / ema20) * 100
        if pct_above > 5:
            short_score += 1
            short_reasons.append(f"Precio {pct_above:.1f}% encima de EMA20 — posible corrección")
    
    # 4. Near resistance
    dist_to_resistance = ((resistance - price) / price) * 100
    if dist_to_resistance < 2:
        short_score += 2
        short_reasons.append(f"Cerca de resistencia ${resistance:,.2f} ({dist_to_resistance:.1f}%)")
    elif dist_to_resistance < 5:
        short_score += 1
        short_reasons.append(f"Resistencia cercana en ${resistance:,.2f}")
    
    # 5. Regime transitioning downward
    if regime_changed and current_regime < prev_regime:
        short_score += 2
        short_reasons.append(f"Transición bajista: {regime_names[prev_regime]} → {regime_names[current_regime]}")
    
    # 6. Probability of moving to lower regime
    if prob_down > 0.15:
        short_score += 1
        short_reasons.append(f"Prob. transición bajista: {prob_down:.0%}")
    
    # 7. Volume confirmation
    if rel_vol > 0.3:
        short_score += 1
        short_reasons.append(f"Volumen alto ({rel_vol:+.1f}x) — confirma movimiento")
    
    # === Calculate entry levels ===
    atr = current_vol * price  # Approximate ATR from volatility
    atr = max(atr, price * 0.01)  # Minimum 1% ATR
    
    # Long entry setup
    long_entry = price
    long_sl = max(support - atr * 0.5, price * 0.97)  # Below support or -3%
    long_tp1 = price + (price - long_sl) * 2  # 1:2 R:R
    long_tp2 = price + (price - long_sl) * 3  # 1:3 R:R
    long_risk_pct = ((long_entry - long_sl) / long_entry) * 100
    long_rr = (long_tp1 - long_entry) / (long_entry - long_sl) if long_entry > long_sl else 0
    
    # Short entry setup
    short_entry = price
    short_sl = min(resistance + atr * 0.5, price * 1.03)  # Above resistance or +3%
    short_tp1 = price - (short_sl - price) * 2  # 1:2 R:R
    short_tp2 = price - (short_sl - price) * 3  # 1:3 R:R
    short_risk_pct = ((short_sl - short_entry) / short_entry) * 100
    short_rr = (short_entry - short_tp1) / (short_sl - short_entry) if short_sl > short_entry else 0
    
    # === Determine signal strength ===
    def signal_strength(score):
        if score >= 5:
            return "FUERTE", "💪"
        elif score >= 3:
            return "MODERADA", "📊"
        elif score >= 1:
            return "DÉBIL", "🔍"
        else:
            return "SIN SEÑAL", "⛔"
    
    long_strength, long_icon = signal_strength(long_score)
    short_strength, short_icon = signal_strength(short_score)
    
    # === Display Signals ===
    col_long, col_short = st.columns(2)
    
    with col_long:
        long_border = "#00cc66" if long_score >= 3 else "#ffaa00" if long_score >= 1 else "#555"
        st.markdown(f"""
        <div style="background: rgba(0,204,102,0.05); border: 2px solid {long_border}; 
                    border-radius: 12px; padding: 1.5rem; min-height: 400px;">
            <h3 style="color: #00cc66; margin: 0 0 0.5rem 0; font-family: 'JetBrains Mono', monospace;">
                {long_icon} LONG — {long_strength}
            </h3>
            <div style="font-size: 2rem; font-weight: 700; color: #00cc66; 
                        font-family: 'JetBrains Mono', monospace;">
                Score: {long_score}/10
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if long_score >= 1:
            st.markdown("**✅ Razones a favor:**")
            for r in long_reasons:
                st.markdown(f"- {r}")
            if long_warnings:
                st.markdown("**⚠️ Advertencias:**")
                for w in long_warnings:
                    st.markdown(f"- {w}")
            
            st.markdown("---")
            st.markdown("**📐 Setup sugerido:**")
            st.markdown(f"""
            | Nivel | Precio | Distancia |
            |-------|--------|-----------|
            | **Entry** | ${long_entry:,.2f} | — |
            | **Stop Loss** | ${long_sl:,.2f} | -{long_risk_pct:.1f}% |
            | **TP1 (1:2)** | ${long_tp1:,.2f} | +{((long_tp1-long_entry)/long_entry)*100:.1f}% |
            | **TP2 (1:3)** | ${long_tp2:,.2f} | +{((long_tp2-long_entry)/long_entry)*100:.1f}% |
            """)
            st.markdown(f"**R:R** = 1:{long_rr:.1f}")
        else:
            st.markdown("*No hay condiciones favorables para LONG en este momento.*")
    
    with col_short:
        short_border = "#ff4444" if short_score >= 3 else "#ffaa00" if short_score >= 1 else "#555"
        st.markdown(f"""
        <div style="background: rgba(255,68,68,0.05); border: 2px solid {short_border}; 
                    border-radius: 12px; padding: 1.5rem; min-height: 400px;">
            <h3 style="color: #ff4444; margin: 0 0 0.5rem 0; font-family: 'JetBrains Mono', monospace;">
                {short_icon} SHORT — {short_strength}
            </h3>
            <div style="font-size: 2rem; font-weight: 700; color: #ff4444; 
                        font-family: 'JetBrains Mono', monospace;">
                Score: {short_score}/10
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if short_score >= 1:
            st.markdown("**✅ Razones a favor:**")
            for r in short_reasons:
                st.markdown(f"- {r}")
            if short_warnings:
                st.markdown("**⚠️ Advertencias:**")
                for w in short_warnings:
                    st.markdown(f"- {w}")
            
            st.markdown("---")
            st.markdown("**📐 Setup sugerido:**")
            st.markdown(f"""
            | Nivel | Precio | Distancia |
            |-------|--------|-----------|
            | **Entry** | ${short_entry:,.2f} | — |
            | **Stop Loss** | ${short_sl:,.2f} | +{short_risk_pct:.1f}% |
            | **TP1 (1:2)** | ${short_tp1:,.2f} | -{((short_entry-short_tp1)/short_entry)*100:.1f}% |
            | **TP2 (1:3)** | ${short_tp2:,.2f} | -{((short_entry-short_tp2)/short_entry)*100:.1f}% |
            """)
            st.markdown(f"**R:R** = 1:{short_rr:.1f}")
        else:
            st.markdown("*No hay condiciones favorables para SHORT en este momento.*")
    
    # === Transition Forecast ===
    st.markdown("### 🔮 Pronóstico de Transición")
    st.caption("Probabilidad de cambio de régimen en el próximo período")
    
    forecast_cols = st.columns(3)
    with forecast_cols[0]:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Prob. Mejora (↑)</div>
            <div class="metric-value" style="color: #00cc66;">{prob_up:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with forecast_cols[1]:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Prob. Mantiene (→)</div>
            <div class="metric-value" style="color: #ffaa00;">{prob_stay:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with forecast_cols[2]:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Prob. Empeora (↓)</div>
            <div class="metric-value" style="color: #ff4444;">{prob_down:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # === Recent Regime Changes (last 10 periods) ===
    st.markdown("### 📜 Cambios de Régimen Recientes")
    
    lookback = min(20, len(states))
    recent_df_data = []
    for j in range(1, lookback):
        idx = len(states) - lookback + j
        if idx > 0 and states[idx] != states[idx - 1]:
            direction = "📈 ALCISTA" if states[idx] > states[idx - 1] else "📉 BAJISTA"
            recent_df_data.append({
                "Fecha": df.index[idx].strftime("%Y-%m-%d %H:%M") if hasattr(df.index[idx], 'strftime') else str(df.index[idx]),
                "Transición": f"{regime_names[states[idx-1]]} → {regime_names[states[idx]]}",
                "Dirección": direction,
                "Precio": f"${df['Close'].iloc[idx]:,.2f}",
            })
    
    if recent_df_data:
        recent_changes_df = pd.DataFrame(recent_df_data[::-1])  # Most recent first
        st.dataframe(recent_changes_df, hide_index=True, use_container_width=True)
    else:
        st.info("Sin cambios de régimen en los últimos 20 períodos — régimen estable.")
    
    # Disclaimer
    st.markdown("""
    <div style="background: rgba(255,170,0,0.1); border: 1px solid rgba(255,170,0,0.3); 
                border-radius: 8px; padding: 0.8rem; margin-top: 1rem;">
        <p style="color: #ffaa00; margin: 0; font-size: 0.8rem;">
            ⚠️ <b>DISCLAIMER:</b> Estas señales son generadas por un modelo estadístico y NO constituyen 
            asesoramiento financiero. Siempre verificá con tu propio análisis técnico y respetá tu 
            protocolo de gestión de riesgo. El modelo puede generar señales incorrectas especialmente 
            en mercados altamente volátiles o durante eventos inesperados.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ─── STEP 10: Model Info ───
    with st.expander("🔍 Información del Modelo"):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Log-Likelihood", f"{log_likelihood:,.1f}")
        with col_m2:
            # AIC approximation
            n_params = n_regimes * (n_regimes - 1) + n_regimes * len(feature_cols) + n_regimes * len(feature_cols) * (len(feature_cols) + 1) // 2
            aic = -2 * log_likelihood + 2 * n_params
            st.metric("AIC", f"{aic:,.1f}")
        with col_m3:
            bic = -2 * log_likelihood + n_params * np.log(len(features))
            st.metric("BIC", f"{bic:,.1f}")
        
        st.markdown(f"""
        **Detalles:**
        - Períodos analizados: **{len(df):,}**
        - Features: **{', '.join(feature_cols)}**
        - Covarianza: **Full** (captura correlaciones entre features)
        - Mejor de **{n_fits}** inicializaciones aleatorias
        - Ticker: **{ticker}** | Timeframe: **{interval}**
        """)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🧬</div>
        <h2 style="font-family: 'Inter', sans-serif; color: #c9d1d9; font-weight: 300;">
            Configura los parámetros en el panel izquierdo
        </h2>
        <p style="color: #8b949e; max-width: 600px; margin: 1rem auto;">
            Esta herramienta utiliza un <b>Modelo Oculto de Markov (HMM)</b> para detectar 
            regímenes de mercado ocultos en criptoactivos. El modelo analiza retornos, RSI, 
            volumen y volatilidad para clasificar cada período en un estado latente.
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;">
            <div class="metric-box" style="min-width: 200px;">
                <div style="font-size: 1.5rem;">📊</div>
                <div class="metric-label" style="margin-top: 0.5rem;">10 Criptoactivos</div>
                <div style="color: #8b949e; font-size: 0.8rem;">BTC, ETH, SOL, DOT, LINK...</div>
            </div>
            <div class="metric-box" style="min-width: 200px;">
                <div style="font-size: 1.5rem;">🧠</div>
                <div class="metric-label" style="margin-top: 0.5rem;">3-7 Regímenes</div>
                <div style="color: #8b949e; font-size: 0.8rem;">Bear → Bull con granularidad</div>
            </div>
            <div class="metric-box" style="min-width: 200px;">
                <div style="font-size: 1.5rem;">⏱️</div>
                <div class="metric-label" style="margin-top: 0.5rem;">3 Timeframes</div>
                <div style="color: #8b949e; font-size: 0.8rem;">1h, 4h, 1d</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
