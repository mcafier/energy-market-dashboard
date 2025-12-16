import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import load_data
from src.features import add_features
from src.model import run_walk_forward_backtest, get_latest_signal

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Energy Quant | Market Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox(
    "Asset Class", 
    ["NG=F", "CL=F"], 
    format_func=lambda x: "Natural Gas (Henry Hub)" if x == "NG=F" else "Crude Oil (WTI)",
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Parameters")
train_window = st.sidebar.slider("Training Window (Days)", 100, 730, 365, help="Initial data required before making the first prediction.")
step_size = st.sidebar.slider("Retraining Frequency (Days)", 7, 90, 21, help="How often the model is retrained to adapt to new market regimes.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Methodology")
st.sidebar.info(
    """
    **Model:** Random Forest Classifier\n
    **Validation:** Walk-Forward (Expanding Window)\n
    **Data:** Yahoo Finance (Price) + Open-Meteo (Temp)
    """
)

# --- MAIN APP LOGIC ---

st.title("⚡ Energy Market Quantitative Monitor")
st.markdown(f"Automated pipeline analyzing **{ticker}** futures using weather data and technical factors.")

# 1. Load Data
@st.cache_data(ttl=3600*12) # Cache data for 12 hours
def get_data(ticker_symbol):
    return load_data(ticker_symbol)

with st.spinner("Ingesting Market & Weather Data..."):
    try:
        raw_df = get_data(ticker)
        df = add_features(raw_df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# 2. Run Backtest
with st.spinner("Running Walk-Forward Simulation..."):
    results = run_walk_forward_backtest(df, train_window_days=train_window, step_days=step_size)

# 3. Get Live Signal
latest_signal = get_latest_signal(df)

# --- DASHBOARD LAYOUT ---

# SECTION A: LIVE PREDICTION 
st.subheader("🔮 AI Prediction for the Next Trading Day")
signal_date = pd.to_datetime(latest_signal['date']).strftime("%A, %d %B %Y")
st.caption(f"Based on market data updated the: **{signal_date}**")

col1, col2, col3 = st.columns(3)

with col1:
    signal_val = latest_signal['signal']
    direction = "LONG (BUY)" if signal_val == 1 else "CASH / SHORT"
    color = "green" if signal_val == 1 else "red"
    
    st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #262730; border-radius: 10px;'>
            <h3 style='color: white; margin:0;'>Signal</h3>
            <h1 style='color: {color}; margin:0;'>{direction}</h1>
            <p style='color: gray;'>Confidence: {latest_signal['confidence']:.1%}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric(
        label="Latest Close Price", 
        value=f"${raw_df['price'].iloc[-1]:.2f}",
        delta=f"{raw_df['price'].pct_change().iloc[-1]:.2%}"
    )

with col3:
    temp_val = raw_df['temperature'].iloc[-1]
    st.metric(
        label="Henry Hub Temperature",
        value=f"{temp_val:.1f}°C"
    )

st.markdown("---")

# SECTION B: PERFORMANCE ANALYSIS
st.subheader("📈 Strategy Performance (Backtest)")

# Calculate Cumulative Returns
results['cum_strategy'] = (1 + results['strategy_returns']).cumprod()
results['cum_market'] = (1 + results['returns']).cumprod()

# Plotly Chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results.index, 
    y=results['cum_strategy'], 
    mode='lines', 
    name='ML Strategy',
    line=dict(color='#00CC96', width=2)
))
fig.add_trace(go.Scatter(
    x=results.index, 
    y=results['cum_market'], 
    mode='lines', 
    name='Market Benchmark (Buy & Hold)',
    line=dict(color='#636EFA', width=2, dash='dash')
))

fig.update_layout(
    title="Equity Curve: Strategy vs Benchmark",
    xaxis_title="Date",
    yaxis_title="Growth of $1 (Cumulative Return)",
    template="plotly_dark",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# SECTION C: UNDER THE HOOD
with st.expander("🔎 View Raw Data & Features"):
    st.write("Recent Data points used for training:")
    st.dataframe(df.tail(10))