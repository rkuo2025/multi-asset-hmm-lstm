import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime

def get_data(ticker):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start="2020-01-01", end=datetime.date.today().isoformat())
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data['Returns'] = data['Close'].pct_change()
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    return data.dropna()

def train_and_predict(ticker, lookback=60):
    df = get_data(ticker)
    if df is None: return None
    
    # 1. Market Regimes (HMM)
    hmm_features = df[['Returns', 'Range']].values
    scaler_hmm = StandardScaler()
    hmm_features_scaled = scaler_hmm.fit_transform(hmm_features)
    hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    hmm_model.fit(hmm_features_scaled)
    df['Regime'] = hmm_model.predict(hmm_features_scaled)
    
    # 2. Price Prediction (LSTM)
    regime_dummies = pd.get_dummies(df['Regime'], prefix='Regime').astype(float)
    lstm_data = pd.concat([df[['Close']], regime_dummies], axis=1)
    scaler_lstm = MinMaxScaler()
    scaled_data = scaler_lstm.fit_transform(lstm_data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    # Predict next price
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, -1)
    next_price_scaled = model.predict(last_sequence)
    
    # Inverse scaling
    dummy = np.zeros((1, lstm_data.shape[1]))
    dummy[0, 0] = next_price_scaled[0, 0]
    next_price = scaler_lstm.inverse_transform(dummy)[0, 0]
    
    return next_price, df['Close'].iloc[-1]

def run_portfolio():
    initial_capital = 1_000_000
    vti_weight = 0.50
    others = ["XLP", "GLD", "BND", "REMX"]
    other_weight = (1.0 - vti_weight) / len(others)
    
    tickers = ["VTI"] + others
    weights = {"VTI": vti_weight}
    for t in others: weights[t] = other_weight
    
    print(f"--- Portfolio Allocation ($ {initial_capital:,}) ---")
    results = []
    
    for ticker in tickers:
        try:
            pred, last = train_and_predict(ticker)
            allocation = initial_capital * weights[ticker]
            shares = allocation / last
            
            signal = "BUY/HOLD" if pred > last else "REDUCE/SELL"
            
            results.append({
                "Ticker": ticker,
                "Weight": f"{weights[ticker]*100:.1f}%",
                "Allocation": f"${allocation:,.2f}",
                "Last Price": f"${last:.2f}",
                "Shares": f"{shares:.2f}",
                "Next Price Pred": f"${pred:.2f}",
                "Signal": signal
            })
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    res_df = pd.DataFrame(results)
    print("\n", res_df)
    res_df.to_csv("portfolio_predictions.csv")

if __name__ == "__main__":
    run_portfolio()
