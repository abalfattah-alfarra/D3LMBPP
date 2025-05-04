# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 02:34:28 2025

@author: ِAL
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
from sklearn.model_selection import TimeSeriesSplit

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2

import keras_tuner as kt

# -------------------------------
# Configuration and Setup
# -------------------------------

# Define the directories
OUTPUT_DIR = 'D:/bitcoin/'
FEATURES_DIR = os.path.join(OUTPUT_DIR, 'features')
EVALUATION_DIR = os.path.join(OUTPUT_DIR, 'evaluation')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# Create evaluation and models directories if they don't exist
for directory in [EVALUATION_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Using existing directory: {directory}")

# Configure logging
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
    """
    Load the cleaned feature-engineered data.
    """
    data_file = os.path.join(FEATURES_DIR, 'clean_feature_engineered_data.csv')
    try:
        df = pd.read_csv(data_file, parse_dates=['date'], index_col='date')
        print("Data loaded successfully.")
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Data file not found at {data_file}. Please ensure the feature engineering step was completed.")
        logging.error(f"Data file not found at {data_file}.")
        sys.exit()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        logging.error(f"An error occurred while loading the data: {e}")
        sys.exit()

def feature_selection_rfe(df, target_column, num_features=20):
    """
    Select top features based on Recursive Feature Elimination (RFE) using Random Forest.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Initialize RFE
    rfe = RFE(estimator=rf, n_features_to_select=num_features)
    rfe.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[rfe.support_].tolist()
    print(f"Top {num_features} features selected based on RFE.")
    logging.info(f"Top {num_features} features selected based on RFE.")
    return selected_features

def preprocess_data(df, feature_columns, target_column, seq_length=30):
    """
    Scale the feature data and create sequences for LSTM.
    """
    # Debug: Print available columns
    print("\nAvailable columns in DataFrame:")
    print(df.columns.tolist())

    if target_column not in df.columns:
        print(f"Error: '{target_column}' column is missing from the DataFrame.")
        logging.error(f"'{target_column}' column is missing from the DataFrame.")
        sys.exit()

    # Ensure the DataFrame is sorted by date
    df = df.sort_index()

    # Split into training and testing sets (80-20 split) before scaling
    split = int(0.8 * len(df))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    # Feature scaling
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scaler on training data
    train_features = feature_scaler.fit_transform(train_df[feature_columns])
    train_target = target_scaler.fit_transform(train_df[[target_column]])

    # Transform testing data
    test_features = feature_scaler.transform(test_df[feature_columns])
    test_target = target_scaler.transform(test_df[[target_column]])

    # Save the scalers for inverse transformation later
    feature_scaler_file = os.path.join(MODELS_DIR, 'feature_scaler_lstm.joblib')
    target_scaler_file = os.path.join(MODELS_DIR, 'target_scaler_lstm.joblib')
    joblib.dump(feature_scaler, feature_scaler_file)
    joblib.dump(target_scaler, target_scaler_file)
    logging.info(f"Feature scaler saved to {feature_scaler_file}.")
    logging.info(f"Target scaler saved to {target_scaler_file}.")

    # Create sequences
    def create_sequences(features, target, seq_length):
        X, y = [], []
        for i in range(seq_length, len(features)):
            X.append(features[i-seq_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_features, train_target, seq_length)
    X_test, y_test = create_sequences(test_features, test_target, seq_length)

    print(f"Data preprocessed: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")
    logging.info(f"Data preprocessed: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")
    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler

def build_enhanced_lstm_model(input_shape):
    """
    Build and compile an enhanced LSTM model with additional layers and regularization.
    """
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))  # Prediction of the next day's price

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print("Enhanced LSTM model built and compiled.")
    logging.info("Enhanced LSTM model built and compiled.")
    return model

def plot_training_history(history):
    """
    Plot the training and validation loss.
    """
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, 'lstm_training_loss.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training loss plot saved to {plot_path}.")
    logging.info(f"Training loss plot saved to {plot_path}.")

def evaluate_model(model, X_test, y_test, target_scaler, model_name):
    """
    Make predictions and evaluate the model.
    """
    predictions = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions)
    y_test_actual = target_scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    r2 = r2_score(y_test_actual, predictions)
    mape = mean_absolute_percentage_error(y_test_actual, predictions)

    print(f"\n{model_name} Performance on Test Set:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    logging.info(f"{model_name} Performance: MAE={mae:.2f}, RMSE={rmse:.2f}, R2_Score={r2:.4f}, MAPE={mape:.2f}%")

    # Plot actual vs predicted
    plt.figure(figsize=(10,6))
    plt.plot(y_test_actual, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f'Actual vs Predicted Prices - {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, f'actual_vs_predicted_{model_name}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Actual vs Predicted plot saved to {plot_path}.")
    logging.info(f"Actual vs Predicted plot saved to {plot_path}.")

    # Residuals plot
    residuals = y_test_actual - predictions
    plt.figure(figsize=(10,6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.tight_layout()
    residuals_plot_path = os.path.join(EVALUATION_DIR, f'residuals_distribution_{model_name}.png')
    plt.savefig(residuals_plot_path)
    plt.close()
    print(f"Residuals distribution plot saved to {residuals_plot_path}.")
    logging.info(f"Residuals distribution plot saved to {residuals_plot_path}.")

def baseline_evaluation(y_test, target_scaler):
    """
    Evaluate a baseline model that predicts the mean of the training data.
    """
    y_test_actual = target_scaler.inverse_transform(y_test)
    mean_prediction = np.mean(y_test_actual)
    predictions = np.full_like(y_test_actual, mean_prediction)

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    r2 = r2_score(y_test_actual, predictions)
    mape = mean_absolute_percentage_error(y_test_actual, predictions)

    print("\nBaseline Performance (Mean Prediction):")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    logging.info(f"Baseline Performance: MAE={mae:.2f}, RMSE={rmse:.2f}, R2_Score={r2:.4f}, MAPE={mape:.2f}%")

    # Plot actual vs baseline
    plt.figure(figsize=(10,6))
    plt.plot(y_test_actual, label='Actual Price')
    plt.plot(predictions, label='Baseline Prediction (Mean)', linestyle='--')
    plt.title('Actual vs Baseline Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, 'actual_vs_baseline.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Actual vs Baseline plot saved to {plot_path}.")
    logging.info(f"Actual vs Baseline plot saved to {plot_path}.")

def residual_analysis(model, X_test, y_test, target_scaler, model_name):
    """
    Perform residual analysis by plotting residuals.
    """
    predictions = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions)
    y_test_actual = target_scaler.inverse_transform(y_test)

    residuals = y_test_actual - predictions
    plt.figure(figsize=(10,6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.tight_layout()
    residuals_plot_path = os.path.join(EVALUATION_DIR, f'residuals_distribution_{model_name}.png')
    plt.savefig(residuals_plot_path)
    plt.close()
    print(f"Residuals distribution plot saved to {residuals_plot_path}.")
    logging.info(f"Residuals distribution plot saved to {residuals_plot_path}.")

def save_model(model, model_name):
    """
    Save the trained model.
    """
    model_file = os.path.join(MODELS_DIR, f'best_model_{model_name}.h5')
    model.save(model_file)
    print(f"{model_name} model saved to {model_file}.")
    logging.info(f"{model_name} model saved to {model_file}.")

def train_random_forest(X_train_flat, y_train, X_test_flat, y_test, target_scaler):
    """
    Train and evaluate a Random Forest Regressor for baseline comparison.
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, y_train.ravel())
    predictions = rf.predict(X_test_flat)
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_actual = target_scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    r2 = r2_score(y_test_actual, predictions)
    mape = mean_absolute_percentage_error(y_test_actual, predictions)

    print(f"\nRandom Forest Performance on Test Set:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    logging.info(f"Random Forest Performance: MAE={mae:.2f}, RMSE={rmse:.2f}, R2_Score={r2:.4f}, MAPE={mape:.2f}%")

    # Plot actual vs predicted
    plt.figure(figsize=(10,6))
    plt.plot(y_test_actual, label='Actual Price')
    plt.plot(predictions, label='Random Forest Prediction')
    plt.title('Actual vs Predicted Prices - Random Forest')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, 'actual_vs_predicted_RandomForest.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Actual vs Predicted plot saved to {plot_path}.")
    logging.info(f"Actual vs Predicted plot saved to {plot_path}.")

def build_model_with_keras_tuner(X_train, y_train, X_val, y_val, input_shape):
    """
    Build and tune the LSTM model using Keras Tuner.
    """
    def model_builder(hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units1', min_value=32, max_value=128, step=32),
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l2(0.001)
        ))
        model.add(Dropout(rate=hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(LSTM(
            units=hp.Int('units2', min_value=16, max_value=64, step=16),
            return_sequences=False,
            kernel_regularizer=l2(0.001)
        ))
        model.add(Dropout(rate=hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(
            units=hp.Int('dense_units', min_value=8, max_value=64, step=8),
            activation='relu'
        ))
        model.add(Dense(1))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        optimizer = Adam(learning_rate=hp_learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='kt_dir',
        project_name='bitcoin_price_prediction_tuning'
    )

    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"Best hyperparameters: {best_hps.values}")

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[stop_early],
        verbose=1
    )

    return model, history

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    # Load data
    df = load_data()

    # Define target column
    target_column = 'Target_Price'

    # Feature selection using RFE
    top_features = feature_selection_rfe(df, target_column, num_features=20)

    # Preprocess data
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler = preprocess_data(df, top_features, target_column, seq_length=30)

    # Further split X_train for validation (e.g., 80-20 split within training)
    validation_split = 0.2
    val_size = int(validation_split * X_train.shape[0])
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # Build the enhanced LSTM model
    model = build_enhanced_lstm_model((X_train.shape[1], X_train.shape[2]))

    # Define Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'best_model_LSTM.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )

    # Train the model with enhanced callbacks
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint, lr_scheduler],
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate the model on the test set
    evaluate_model(model, X_test, y_test, target_scaler, 'LSTM')

    # Save the trained LSTM model
    save_model(model, 'LSTM')

    # Evaluate baseline
    baseline_evaluation(y_test, target_scaler)

    # Flatten the data for Random Forest (as it doesn't handle sequences)
    X_train_flat = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    X_test_flat = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    X_val_flat = X_val.reshape((X_val.shape[0], X_val.shape[1]*X_val.shape[2]))
    y_val_flat = y_val

    # Train and evaluate Random Forest
    train_random_forest(X_train_flat, y_train, X_test_flat, y_test, target_scaler)

    # Example of Hyperparameter Tuning using Keras Tuner (optional)
    # Uncomment the following lines to perform hyperparameter tuning

    # tuned_model, tuned_history = build_model_with_keras_tuner(X_train, y_train, X_val, y_val, (X_train.shape[1], X_train.shape[2]))
    # plot_training_history(tuned_history)
    # evaluate_model(tuned_model, X_test, y_test, target_scaler, 'Tuned_LSTM')
    # save_model(tuned_model, 'Tuned_LSTM')

    # Note: Hyperparameter tuning can be time-consuming. Ensure you have sufficient computational resources before running.
