# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 02:34:28 2025

@author: AL
"""

# bitcoin_price_prediction.py

import pandas as pd
import numpy as np
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import keras_tuner as kt

# -------------------------------
# Configuration and Setup
# -------------------------------

OUTPUT_DIR = 'D:/bitcoin/'
FEATURES_DIR = os.path.join(OUTPUT_DIR, 'features')
EVALUATION_DIR = os.path.join(OUTPUT_DIR, 'evaluation')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

for directory in [EVALUATION_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"Using directory: {directory}")

LOG_FILE = os.path.join(EVALUATION_DIR, 'model_training.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# -------------------------------
# Function Definitions
# -------------------------------

def load_data():
    data_file = os.path.join(FEATURES_DIR, 'clean_feature_engineered_data.csv')
    try:
        df = pd.read_csv(data_file, parse_dates=['date'], index_col='date')
        print("Data loaded successfully.")
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Data file not found at {data_file}.")
        logging.error(f"Data file not found at {data_file}.")
        sys.exit()


def feature_selection_rfe(df, target_column, num_features=20):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=num_features)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_].tolist()
    print(f"Top {num_features} features selected based on RFE.")
    logging.info(f"Top {num_features} features selected based on RFE.")
    return selected_features


def preprocess_data(df, feature_columns, target_column, seq_length=30):
    if target_column not in df.columns:
        print(f"Error: '{target_column}' column is missing.")
        logging.error(f"'{target_column}' column is missing.")
        sys.exit()

    df = df.sort_index()
    split = int(0.8 * len(df))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features = feature_scaler.fit_transform(train_df[feature_columns])
    train_target = target_scaler.fit_transform(train_df[[target_column]])
    test_features = feature_scaler.transform(test_df[feature_columns])
    test_target = target_scaler.transform(test_df[[target_column]])

    joblib.dump(feature_scaler, os.path.join(MODELS_DIR, 'feature_scaler.joblib'))
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, 'target_scaler.joblib'))

    def create_sequences(features, target, seq_length):
        X, y = [], []
        for i in range(seq_length, len(features)):
            X.append(features[i-seq_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_features, train_target, seq_length)
    X_test, y_test = create_sequences(test_features, test_target, seq_length)

    print(f"Data preprocessed: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")
    return X_train, y_train, X_test, y_test, target_scaler


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        LSTM(32, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.0005), loss='mean_squared_error')
    print("LSTM model built.")
    return model


def plot_training(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(EVALUATION_DIR, 'training_loss.png'))
    plt.close()


def evaluate(model, X_test, y_test, scaler):
    predictions = scaler.inverse_transform(model.predict(X_test))
    y_actual = scaler.inverse_transform(y_test)
    mae = mean_absolute_error(y_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_actual, predictions))
    r2 = r2_score(y_actual, predictions)

    print(f"Test Results: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

    plt.plot(y_actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.savefig(os.path.join(EVALUATION_DIR, 'actual_vs_predicted.png'))
    plt.close()


if __name__ == "__main__":
    df = load_data()
    target_column = 'Target_Price'
    features = feature_selection_rfe(df, target_column)
    X_train, y_train, X_test, y_test, scaler = preprocess_data(df, features, target_column)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    checkpoint = ModelCheckpoint(os.path.join(MODELS_DIR, 'best_model_LSTM.keras'), save_best_only=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[EarlyStopping(patience=10), checkpoint]
    )

    plot_training(history)
    evaluate(model, X_test, y_test, scaler)
