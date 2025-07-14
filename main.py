import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# === PARAMETRI MODIFICABILI ===
CSV_FILE = 'nas100_data.csv'
MODEL_FILE = 'bias_model.xgb'
WINDOW_SIZE = 20
THRESHOLD = 0.002
TEST_SIZE = 0.2
MIN_DATA_POINTS = 100

# === FUNZIONI ===
def create_features(df):
    df['return'] = df['Close'].pct_change()
    df['ma'] = df['Close'].rolling(WINDOW_SIZE).mean()
    df['std'] = df['Close'].rolling(WINDOW_SIZE).std()
    df['upper'] = df['ma'] + df['std']
    df['lower'] = df['ma'] - df['std']
    df['label'] = np.where(df['return'].shift(-1) > THRESHOLD, 1,
                   np.where(df['return'].shift(-1) < -THRESHOLD, -1, 0))
    return df.dropna()

def train_model(df):
    X = df[['Close', 'ma', 'std', 'upper', 'lower']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    return model

def save_model(model, filename):
    model.save_model(filename)

def load_model(filename):
    model = xgb.XGBClassifier()
    model.load_model(filename)
    return model

def predict_bias(model, latest_features):
    return model.predict(latest_features)[0]

# === MAIN ===
def main():
    df = pd.read_csv(CSV_FILE)
    if len(df) < MIN_DATA_POINTS:
        print("Dati insufficienti.")
        return

    df = create_features(df)

    if os.path.exists(MODEL_FILE):
        print("✅ Modello trovato. Caricamento in corso...")
        model = load_model(MODEL_FILE)
    else:
        print("⚠️ Nessun modello trovato. Addestramento in corso...")
        model = train_model(df)
        save_model(model, MODEL_FILE)
        print(f"✅ Modello salvato in {MODEL_FILE}")

    latest = df[['Close', 'ma', 'std', 'upper', 'lower']].iloc[[-1]]
    prediction = predict_bias(model, latest)
    bias_map = {-1: 'Bearish', 0: 'Neutral', 1: 'Bullish'}
    print(f"📈 Bias atteso: {bias_map[prediction]}")

if __name__ == "__main__":
    main()
