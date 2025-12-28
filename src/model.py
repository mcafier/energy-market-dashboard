import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the Target column for Supervised Learning.
    Target = 1 if the NEXT day's return is positive, else 0.
    """
    df = df.copy()
    # We want to predict if Price(t+1) > Price(t)
    # df['returns'] at index t represents return from t-1 to t.
    # So we need to shift returns backwards by 1 to align t with t+1 return.
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    return df

def run_walk_forward_backtest(df: pd.DataFrame, train_window_days: int = 365, step_days: int = 30) -> pd.DataFrame:
    """
    Simulates a trading strategy using Walk-Forward Validation.
    
    Args:
        df: Dataframe with features.
        train_window_days: Initial size of training data.
        step_days: How often to retrain the model (Re-optimization frequency).
    
    Returns:
        pd.DataFrame: Original data enriched with 'signal' and 'strategy_returns'.
    """
    logging.info("Starting Walk-Forward Backtest simulation...")
    
    df = prepare_targets(df)
    
    # Select Features (Automatically exclude non-numeric or target columns)
    # We exclude 'price' to force model to learn from indicators, not raw levels
    exclude_cols = ['price', 'target', 'returns', 'date', 'volume']
    features = [c for c in df.columns if c not in exclude_cols]
    
    logging.info(f"Training on features: {features}")
    
    # Model Configuration
    # We use a Random Forest.
    # Key Hyperparameters to prevent overfitting:
    # min_samples_split=50: Don't look at patterns that happened less than 50 times.
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
    
    predictions = []
    
    total_rows = len(df)
    # The last row has no target (because we don't know tomorrow's return yet)
    # So we can only backtest up to total_rows - 1
    backtest_limit = total_rows - 1 
    
    # Cursor points to the start of the "Test" period
    cursor = train_window_days 
    
    while cursor < backtest_limit:
        # Define Time Windows
        train_end = cursor
        test_end = min(cursor + step_days, backtest_limit)
        
        # 1. Train
        X_train = df.iloc[:train_end][features]
        y_train = df.iloc[:train_end]['target']
        model.fit(X_train, y_train)
        
        # 2. Predict (the 'step' period)
        X_test = df.iloc[train_end:test_end][features]
        preds = model.predict(X_test)
        
        # 3. Store Results aligned with Dates
        test_dates = df.index[train_end:test_end]
        for date, pred in zip(test_dates, preds):
            predictions.append({'date': date, 'signal': pred})
            
        # 4. Move Window
        cursor += step_days
        
    # Formatting Results
    df_preds = pd.DataFrame(predictions).set_index('date')
    
    # Join predictions with the original data
    results = df.join(df_preds, how='left')
    
    # Calculate Strategy Returns
    # Logic: 
    # If Signal at day T is 1 (Buy), we hold position for day T+1.
    # So Strategy Return at T+1 = Signal(T) * Return(T+1)
    results['simple_returns'] = np.exp(results['returns']) - 1

    results['strategy_returns'] = results['signal'].shift(1) * results['simple_returns']
    
    
    # Clean up (remove the initial training period where we have no signals)
    results.dropna(subset=['signal'], inplace=True)
    
    logging.info(f"Backtest complete. Simulation covers {len(results)} trading days.")
    return results

def get_latest_signal(df: pd.DataFrame) -> dict:
    """
    Retrains the model on the FULL dataset to predict tomorrow's move.
    """
    df = prepare_targets(df)
    
    exclude_cols = ['price', 'target', 'returns', 'date', 'volume']
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Train on everything EXCEPT the last row (which has no target)
    X_train = df.iloc[:-1][features]
    y_train = df.iloc[:-1]['target']
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict using TODAY's data (the last row)
    last_row = df.iloc[[-1]][features]
    prediction = model.predict(last_row)[0]
    
    # Get probability (confidence)
    proba = model.predict_proba(last_row)[0]
    confidence = max(proba)
    
    return {
        "signal": int(prediction),
        "confidence": float(confidence),
        "date": df.index[-1].strftime("%Y-%m-%d")
    }