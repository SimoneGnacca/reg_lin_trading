import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from tradingview_ta import TA_Handler, Interval, Exchange
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === PARAMETRI ===
WINDOW_SIZE = 20
THRESHOLD = 0.002
TEST_SIZE = 0.2
MIN_DATA_POINTS = 100
SYMBOL = "NAS100"
EXCHANGE = "FOREXCOM"
MODEL_FILE = "model.pkl"

# === Scarica dati da TradingView ===
def fetch_data():
    handler = TA_Handler(
        symbol=SYMBOL,
        screener="forex",
        exchange=EXCHANGE,
        interval=Interval.INTERVAL_1_HOUR
    )
    candles = handler.get_analysis().indicators.get("ohlcv", None)
    if not candles:
        raise ValueError("Dati OHLCV non disponibili. Verifica simbolo o exchange.")
    df = pd.DataFrame(candles, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df.set_index('Time', inplace=True)
    return df

# === Feature engineering ===
def create_features(df):
    df['return'] = df['Close'].pct_change()
    df['ma'] = df['Close'].rolling(WINDOW_SIZE).mean()
    df['std'] = df['Close'].rolling(WINDOW_SIZE).std()
    df['upper'] = df['ma'] + df['std']
    df['lower'] = df['ma'] - df['std']
    df['label'] = np.where(df['return'].shift(-1) > THRESHOLD, 1,
                  np.where(df['return'].shift(-1) < -THRESHOLD, -1, 0))
    df = df.dropna()
    return df

# === Allenamento modello ===
def train_model(df):
    X = df[['Close', 'ma', 'std', 'upper', 'lower']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained. Accuracy: {accuracy * 100:.2f}%")
    joblib.dump(model, MODEL_FILE)
    print(f"Modello salvato in {MODEL_FILE}")
    return model

# === Predizione ===
def predict_bias(model, df):
    latest = df[['Close', 'ma', 'std', 'upper', 'lower']].iloc[[-1]]
    prediction = model.predict(latest)
    return prediction[0]

# === MAIN ===
def main():
    try:
        df = fetch_data()
    except Exception as e:
        print(f"Errore nel download dei dati: {e}")
        return

    if len(df) < MIN_DATA_POINTS:
        print("Dati insufficienti per il training.")
        return

    df = create_features(df)

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        print(f"Modello caricato da {MODEL_FILE}")
    else:
        model = train_model(df)

    prediction = predict_bias(model, df)
    bias_map = {-1: 'Bearish', 0: 'Neutral', 1: 'Bullish'}
    print(f"Bias atteso per NAS100: {bias_map[prediction]}")

if __name__ == "__main__":
    main()
