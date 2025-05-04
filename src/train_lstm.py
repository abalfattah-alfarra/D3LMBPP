# Updated train_lstm.py with additional baselines and dataset loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Additional baselines
import pmdarima as pm  # ARIMA
from prophet import Prophet  # Prophet

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Directories
MODEL_DIR = 'models'
EVAL_DIR = 'evaluation'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# Load enriched dataset
df = pd.read_csv('last_bitcoin_data_with_sentiment.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Feature engineering (existing + on-chain metrics already in CSV)
# Compute EMA, ATR, OBV if not present
if 'EMA_10' not in df:
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
# Assume 'HashRate' and 'TxVolume' columns exist

# Sentiment score already in file: 'Sentiment_Score'

# Lagged features
lags = list(range(1, 8))
for lag in lags:
    df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

# Rolling stats
if 'Close_rolling_mean_7' not in df:
    df['Close_rolling_mean_7'] = df['Close'].rolling(window=7).mean()
    df['Close_rolling_std_7'] = df['Close'].rolling(window=7).std()

# Drop na
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Select features and target
features = [
    'Close', 'EMA_10', 'EMA_20', 'ATR', 'OBV', 'Sentiment_Score',
] + [f'Close_lag_{lag}' for lag in lags] + [
    'Close_rolling_mean_7', 'Close_rolling_std_7', 'HashRate', 'TxVolume'
]
X = df[features]
y = df['Close']

# Impute, select, scale
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=features)
selector = SelectKBest(score_func=f_regression, k=20)
X = pd.DataFrame(selector.fit_transform(X, y), columns=[features[i] for i in selector.get_support(indices=True)])
X_scaled = StandardScaler().fit_transform(X)

# Create sequences
def create_sequences(X, y, seq_len=7):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y.values)

# Time series CV
tscv = TimeSeriesSplit(n_splits=5)

# Store results
df_results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_seq), 1):
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]

    # LSTM model
    def build_model(input_shape):
        inp = Input(shape=input_shape)
        x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(inp)
        x = Dropout(0.2)(x)
        x = LSTM(32, kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.2)(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(tf.expand_dims(x,1), tf.expand_dims(x,1))
        x = LayerNormalization()(tf.squeeze(attn,1) + x)
        x = Dense(16, activation='relu')(x)
        out = Dense(1)(x)
        model = Model(inp, out)
        model.compile('adam', loss='mse', metrics=['mae'])
        return model

    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(patience=3)],
                        verbose=0)
    y_pred = model.predict(X_test).flatten()
    df_results.append({'Model': 'LSTM', 'Fold': fold,
                       'MAE': mean_absolute_error(y_test,y_pred),
                       'RMSE': np.sqrt(mean_squared_error(y_test,y_pred)),
                       'R2': r2_score(y_test,y_pred)})

    # Linear Regression
gb_x = X_train.reshape(len(X_train), -1)
    lr = LinearRegression().fit(gb_x, y_train)
    y_lr = lr.predict(X_test.reshape(len(X_test), -1))
    df_results.append({'Model': 'Linear', 'Fold': fold,
                       'MAE': mean_absolute_error(y_test,y_lr),
                       'RMSE': np.sqrt(mean_squared_error(y_test,y_lr)),
                       'R2': r2_score(y_test,y_lr)})

    # XGBoost
    xgbr = xgb.XGBRegressor(random_state=42).fit(gb_x, y_train)
    y_xgb = xgbr.predict(X_test.reshape(len(X_test), -1))
    df_results.append({'Model': 'XGBoost', 'Fold': fold,
                       'MAE': mean_absolute_error(y_test,y_xgb),
                       'RMSE': np.sqrt(mean_squared_error(y_test,y_xgb)),
                       'R2': r2_score(y_test,y_xgb)})

    # ARIMA baseline (on 'Close')
    arima = pm.auto_arima(y_train, seasonal=False, suppress_warnings=True)
    y_arima = arima.predict(n_periods=len(y_test))
    df_results.append({'Model': 'ARIMA', 'Fold': fold,
                       'MAE': mean_absolute_error(y_test,y_arima),
                       'RMSE': np.sqrt(mean_squared_error(y_test,y_arima)),
                       'R2': r2_score(y_test,y_arima)})

    # Prophet baseline
    df_prophet = df[['Date','Close']].iloc[train_idx+7].rename(columns={'Date':'ds','Close':'y'})
    m = Prophet().fit(df_prophet)
    future = m.make_future_dataframe(periods=len(y_test), freq='D')
    forecast = m.predict(future)
    y_prophet = forecast['yhat'].values[-len(y_test):]
    df_results.append({'Model': 'Prophet', 'Fold': fold,
                       'MAE': mean_absolute_error(y_test,y_prophet),
                       'RMSE': np.sqrt(mean_squared_error(y_test,y_prophet)),
                       'R2': r2_score(y_test,y_prophet)})

# Save summary
pd.DataFrame(df_results).to_csv(os.path.join(EVAL_DIR,'performance_summary.csv'), index=False)
