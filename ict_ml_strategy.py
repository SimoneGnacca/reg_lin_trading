
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

# === PARAMETRI MODIFICABILI ===
CSV_FILE = 'nas100_data.csv'        # File storico
WINDOW_SIZE = 20                    # Finestra per feature rolling
THRESHOLD = 0.002                   # Soglia percentuale per bias
TEST_SIZE = 0.2                     # Split train/test
MIN_DATA_POINTS = 100              # Dati minimi richiesti

# === FUNZIONI ===
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

def train_model(df):
    X = df[['Close', 'ma', 'std', 'upper', 'lower']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    return model, X_test, y_test, y_pred

def predict_bias(model, latest_features):
    prediction = model.predict(latest_features)
    return prediction[0]

# === MAIN ===
def main():
    df = pd.read_csv(CSV_FILE)
    if len(df) < MIN_DATA_POINTS:
        print("Dati insufficienti.")
        return

    df = create_features(df)
    model, X_test, y_test, y_pred = train_model(df)
    latest = df[['Close', 'ma', 'std', 'upper', 'lower']].iloc[[-1]]
    prediction = predict_bias(model, latest)

    bias_map = {-1: 'Bearish', 0: 'Neutral', 1: 'Bullish'}
    print(f"Bias atteso: {bias_map[prediction]}")

if __name__ == "__main__":
    main()
