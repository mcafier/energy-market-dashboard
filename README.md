![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://energy-market-dashboard.streamlit.app/)

# Energy Market Quantitative Analysis Pipeline

This repository contains an end-to-end quantitative trading framework designed to analyze and predict US Natural Gas Futures (`NG=F`). The pipeline integrates market data with fundamental weather data to identify trading opportunities using Machine Learning.

The project includes a research notebook for data exploration, a modular Python package for strategy execution, and a deployed Streamlit dashboard for real-time monitoring.

## Live Demo

You can test the full functionality immediately in your browser. No API key or installation required. 👉 [Click here to open the Live App](https://energy-market-dashboard.streamlit.app/)

## Project Overview

The core hypothesis of this project is that Natural Gas markets, while efficient regarding price history, are heavily influenced by physical factors. By incorporating temperature data from Henry Hub (the physical delivery point for NYMEX futures), this model aims to capture supply/demand shocks that technical analysis alone might miss.

**Key Capabilities:**
*   **Data Ingestion:** Automated retrieval of financial data (Yahoo Finance) and historical weather data (Open-Meteo API).
*   **Feature Engineering:** Vectorized calculation of technical indicators (RSI, Moving Averages) and fundamental signals (Weather Anomalies).
*   **Modeling:** Random Forest Classifier trained to predict daily price direction.
*   **Validation:** Strict Walk-Forward Validation (Expanding Window) to simulate real-world performance and eliminate look-ahead bias.

## Repository Structure

```text
energy-market-dashboard/
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # ETL pipeline for Market and Weather data
│   ├── features.py       # Vectorized feature engineering logic
│   └── model.py          # Random Forest training and Backtesting engine
├── notebooks/
│   ├── research.ipynb    # Jupyter Notebook for EDA and prototyping
├── app.py                # Streamlit Dashboard entry point
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Installation & Usage

### Prerequisites
* Python 3.10+
* pip

### Setup
1. Clone the repository:

```Bash
git clone https://github.com/mcafier/energy-market-dashboard.git
cd energy-market-dashboard
```

2. Create a virtual environment:
```Bash
conda create -n energy-quant python=3.10 -y
conda activate energy-quant
```
3. Install dependencies:
```Bash
pip install -r requirements.txt
```
### Running the Dashboard
To launch the interactive web interface:
```Bash
streamlit run app.py
```

### Exploratory Analysis
To view the initial data analysis and correlation studies:

```Bash
jupyter notebook notebooks/research.ipynb
```

## Methodology
### 1. Feature Engineering
We transform non-stationary price data into stationary features suitable for Machine Learning:
* **Trend**: Distance from 50-day Moving Average, distance from 20-day Moving Average.
* **Momentum**: Relative Strength Index (RSI) using Wilder's smoothing.
* **Volatility**: 20-day rolling standard deviation of log returns.
* **Fundamental**: Temperature Anomaly (Deviation from 30-day average at Henry Hub).

### 2. Backtesting Strategy
To ensure the reliability of the performance metrics, we do not use a standard Train/Test split. Instead, we use **Walk-Forward Validation**:
* **Initial Training**: 1 year of historical data.
* **Step Size**: The model predicts the next 30 days, then is retrained with the new data included.
* This approach mimics the actual workflow of a systematic trader updating their model periodically.

## Disclaimer
This software is for educational and research purposes only. It does not constitute financial advice.