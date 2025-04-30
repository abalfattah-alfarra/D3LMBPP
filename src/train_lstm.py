# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# For data preprocessing and feature engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

# For sentiment analysis
from textblob import TextBlob

# For modeling
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Layer, Input, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l2

# For baseline models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# For hyperparameter tuning
import keras_tuner as kt

# Suppress TensorFlow warnings for clarity
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define Paths for Saving Models and Plots
MODEL_DIR = 'D:/bitcoin/models'
EVAL_DIR = 'D:/bitcoin/evaluation'

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# Load Dataset
# Replace 'bitcoin_data.csv' with your actual data file path
df = pd.read_csv('bitcoin_data.csv', parse_dates=['Date'])

# Sort by Date to ensure chronological order
df = df.sort_values('Date').reset_index(drop=True)

# Feature Engineering

# a. Exponential Moving Averages
df['EMA_10'] = df['price'].ewm(span=10, adjust=False).mean()
df['EMA_20'] = df['price'].ewm(span=20, adjust=False).mean()

# b. Average True Range (ATR)
df['High-Low'] = df['high'] - df['low']
df['High-PrevClose'] = abs(df['high'] - df['price'].shift(1))
df['Low-PrevClose'] = abs(df['low'] - df['price'].shift(1))
df['ATR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1).rolling(window=14).mean()

# c. On-Balance Volume (OBV)
df['OBV'] = (np.sign(df['price'].diff()) * df['volume']).fillna(0).cumsum()

# d. Sentiment Analysis on News Headlines
# Ensure you have a 'news_headline' column in your dataset
def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

df['Sentiment_Score'] = df['news_headline'].apply(get_sentiment)

# e. Lagged Features
for lag in range(1, 8):
    df[f'price_lag_{lag}'] = df['price'].shift(lag)
    df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

# f. Rolling Statistics
df['price_rolling_mean_7'] = df['price'].rolling(window=7).mean()
df['price_rolling_std_7'] = df['price'].rolling(window=7).std()

# g. Cyclical Encoding for Month and Day
df['month_sin'] = np.sin(2 * np.pi * df['Month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['Month']/12)
df['day_sin'] = np.sin(2 * np.pi * df['Day']/31)
df['day_cos'] = np.cos(2 * np.pi * df['Day']/31)

# h. Additional Technical Indicators (Optional)
# You can add more technical indicators as needed

# Drop rows with missing values after feature engineering
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Define Features and Target
features = [
    'price', 'EMA_10', 'EMA_20', 'ATR', 'OBV', 'Sentiment_Score',
    'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4',
    'price_lag_5', 'price_lag_6', 'price_lag_7',
    'price_rolling_mean_7', 'price_rolling_std_7',
    'month_sin', 'month_cos', 'day_sin', 'day_cos',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_4',
    'volume_lag_5', 'volume_lag_6', 'volume_lag_7'
]

X = df[features]
y = df['price']

# Handle Missing Values (If Any Remaining)
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature Selection using SelectKBest
selector = SelectKBest(score_func=f_regression, k=20)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sequence Creation for LSTM
sequence_length = 7  # You can experiment with different sequence lengths

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y.values, sequence_length)

# Define Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Initialize Lists to Store Results
lstm_results = []
linear_results = []
xgb_results = []

# Iterate through each fold
for fold, (train_index, test_index) in enumerate(tscv.split(X_seq)):
    print(f'\n--- Fold {fold+1} ---')
    X_train, X_test = X_seq[train_index], X_seq[test_index]
    y_train, y_test = y_seq[train_index], y_seq[test_index]
    
    # Build the Enhanced LSTM Model
    def build_enhanced_lstm_model(input_shape):
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(32, kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.2)(x)
        # Attention Mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = LayerNormalization(epsilon=1e-6)(attention + x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    model = build_enhanced_lstm_model((sequence_length, X_train.shape[2]))
    
    # Define Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    # Train the LSTM Model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Save Training Loss Plot
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold+1} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig(os.path.join(EVAL_DIR, f'LSTM_Fold_{fold+1}_training_loss.png'))
    plt.close()
    
    # Evaluate the LSTM Model
    lstm_mae, lstm_rmse = model.evaluate(X_test, y_test, verbose=0)
    y_pred_lstm = model.predict(X_test).flatten()
    lstm_r2 = r2_score(y_test, y_pred_lstm)
    
    print(f'LSTM - MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}, R²: {lstm_r2:.4f}')
    lstm_results.append({'Fold': fold+1, 'MAE': lstm_mae, 'RMSE': lstm_rmse, 'R2': lstm_r2})
    
    # Save Actual vs Predicted Plot
    plt.figure(figsize=(10,6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred_lstm, label='Predicted')
    plt.title(f'Fold {fold+1} - LSTM Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.savefig(os.path.join(EVAL_DIR, f'actual_vs_predicted_LSTM_Fold_{fold+1}.png'))
    plt.close()
    
    # Save Residuals Distribution Plot
    residuals = y_test - y_pred_lstm
    plt.figure(figsize=(10,6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title(f'Fold {fold+1} - LSTM Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(EVAL_DIR, f'residuals_distribution_LSTM_Fold_{fold+1}.png'))
    plt.close()
    
    # Save the LSTM Model
    model.save(os.path.join(MODEL_DIR, f'best_model_LSTM_Fold_{fold+1}.h5'))
    
    # ----------------------- Baseline Models -----------------------
    
    # a. Linear Regression
    lr = LinearRegression()
    # Reshape data for Linear Regression
    X_train_lr = X_train.reshape(X_train.shape[0], -1)
    X_test_lr = X_test.reshape(X_test.shape[0], -1)
    lr.fit(X_train_lr, y_train)
    y_pred_lr = lr.predict(X_test_lr)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)
    print(f'Linear Regression - MAE: {lr_mae:.4f}, RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}')
    linear_results.append({'Fold': fold+1, 'MAE': lr_mae, 'RMSE': lr_rmse, 'R2': lr_r2})
    
    # Save Actual vs Predicted Plot for Linear Regression
    plt.figure(figsize=(10,6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred_lr, label='Predicted')
    plt.title(f'Fold {fold+1} - Linear Regression Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.savefig(os.path.join(EVAL_DIR, f'actual_vs_predicted_LR_Fold_{fold+1}.png'))
    plt.close()
    
    # Save Residuals Distribution Plot for Linear Regression
    residuals_lr = y_test - y_pred_lr
    plt.figure(figsize=(10,6))
    sns.histplot(residuals_lr, bins=50, kde=True)
    plt.title(f'Fold {fold+1} - Linear Regression Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(EVAL_DIR, f'residuals_distribution_LR_Fold_{fold+1}.png'))
    plt.close()
    
    # b. XGBoost Regressor
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    xgbr.fit(X_train_lr, y_train)
    y_pred_xgbr = xgbr.predict(X_test_lr)
    xgbr_mae = mean_absolute_error(y_test, y_pred_xgbr)
    xgbr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgbr))
    xgbr_r2 = r2_score(y_test, y_pred_xgbr)
    print(f'XGBoost Regressor - MAE: {xgbr_mae:.4f}, RMSE: {xgbr_rmse:.4f}, R²: {xgbr_r2:.4f}')
    xgb_results.append({'Fold': fold+1, 'MAE': xgbr_mae, 'RMSE': xgbr_rmse, 'R2': xgbr_r2})
    
    # Save Actual vs Predicted Plot for XGBoost Regressor
    plt.figure(figsize=(10,6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred_xgbr, label='Predicted')
    plt.title(f'Fold {fold+1} - XGBoost Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.savefig(os.path.join(EVAL_DIR, f'actual_vs_predicted_XGBoost_Fold_{fold+1}.png'))
    plt.close()
    
    # Save Residuals Distribution Plot for XGBoost Regressor
    residuals_xgbr = y_test - y_pred_xgbr
    plt.figure(figsize=(10,6))
    sns.histplot(residuals_xgbr, bins=50, kde=True)
    plt.title(f'Fold {fold+1} - XGBoost Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(EVAL_DIR, f'residuals_distribution_XGBoost_Fold_{fold+1}.png'))
    plt.close()
    
    # Save the XGBoost Model
    xgbr.save_model(os.path.join(MODEL_DIR, f'best_model_XGBoost_Fold_{fold+1}.json'))

# ----------------------- Summary of Results -----------------------

# Convert Results to DataFrames
lstm_df = pd.DataFrame(lstm_results)
linear_df = pd.DataFrame(linear_results)
xgb_df = pd.DataFrame(xgb_results)

# Display Summary
print('\n--- Summary of Results ---')
print('\nLSTM Regressor:')
print(lstm_df.describe())

print('\nLinear Regression:')
print(linear_df.describe())

print('\nXGBoost Regressor:')
print(xgb_df.describe())

# Save Summary to Excel
with pd.ExcelWriter(os.path.join(EVAL_DIR, 'model_performance_summary.xlsx')) as writer:
    lstm_df.describe().to_excel(writer, sheet_name='LSTM')
    linear_df.describe().to_excel(writer, sheet_name='Linear_Regression')
    xgb_df.describe().to_excel(writer, sheet_name='XGBoost_Regressor')

print('\nModel performance summaries saved to Excel.')

# ----------------------- Hyperparameter Tuning (Optional) -----------------------

# Implement Hyperparameter Tuning with Keras Tuner for One Fold (e.g., Fold 1)
# This step can be time-consuming; consider running it separately

"""
# Select Fold 1 data for tuning
fold = 1
train_idx, test_idx = list(tscv.split(X_seq))[fold-1]
X_train_fold, X_test_fold = X_seq[train_idx], X_seq[test_idx]
y_train_fold, y_test_fold = y_seq[train_idx], y_seq[test_idx]

def build_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        return_sequences=True,
        kernel_regularizer=l2(0.001)),
        input_shape=(sequence_length, X_train_fold.shape[2])
    ))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(
        units=hp.Int('units_2', min_value=32, max_value=128, step=32),
        kernel_regularizer=l2(0.001)
    ))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    # Attention Mechanism
    model.add(MultiHeadAttention(num_heads=4, key_dim=64))
    model.add(LayerNormalization(epsilon=1e-6))
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(1))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='mse',
        metrics=['mae']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=10,
    executions_per_trial=3,
    directory='D:/bitcoin/tuner_dir',
    project_name='bitcoin_price_tuning'
)

stop_early = EarlyStopping(monitor='val_mae', patience=10)

tuner.search(
    X_train_fold, y_train_fold,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[stop_early],
    verbose=1
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train_fold, y_train_fold,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[stop_early],
    verbose=1
)

# Evaluate the tuned model
mae, rmse = model.evaluate(X_test_fold, y_test_fold, verbose=0)
print(f'Tuned LSTM - MAE: {mae:.4f}, RMSE: {rmse:.4f}')

# Save the tuned model
model.save(os.path.join(MODEL_DIR, f'best_tuned_model_LSTM_Fold_{fold}.h5'))
"""

# ----------------------- End of Script -----------------------

print("\nAll processes completed successfully!")
