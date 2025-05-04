# bitcoin_price_regression.py

import pandas as pd
import numpy as np
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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
LOG_FILE = os.path.join(EVALUATION_DIR, 'lstm_model_training.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# -------------------------------
# Function Definitions
# -------------------------------

def load_data(file_path):
    """
    Load the feature-engineered data.
    """
    try:
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path, parse_dates=['date'], dayfirst=True)
        print("Data loaded successfully.")
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Data file not found at {file_path}. Please ensure the feature engineering step was completed.")
        logging.error(f"Data file not found at {file_path}.")
        sys.exit()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        logging.error(f"An error occurred while loading the data: {e}")
        sys.exit()

def preprocess_data_regression(df, feature_columns, target_column, seq_length=7):
    """
    Scale the feature data and create sequences for LSTM (Regression).
    Uses separate scalers for features and target.
    """
    print("\nStarting data preprocessing...")
    logging.info("Starting data preprocessing.")
    
    print("\nAvailable columns in DataFrame:")
    print(df.columns.tolist())

    if target_column not in df.columns:
        print(f"Error: '{target_column}' column is missing from the DataFrame.")
        logging.error(f"'{target_column}' column is missing from the DataFrame.")
        sys.exit()

    # Initialize separate scalers
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    print("Scaling features...")
    logging.info("Scaling features.")
    scaled_features = scaler_features.fit_transform(df[feature_columns])

    print("Scaling target variable...")
    logging.info("Scaling target variable.")
    scaled_target = scaler_target.fit_transform(df[[target_column]])

    # Save the scalers for inverse transformation later
    scaler_features_file = os.path.join(MODELS_DIR, 'scaler_features.joblib')
    scaler_target_file = os.path.join(MODELS_DIR, 'scaler_target.joblib')
    joblib.dump(scaler_features, scaler_features_file)
    joblib.dump(scaler_target, scaler_target_file)
    logging.info(f"Feature scaler saved to {scaler_features_file}.")
    logging.info(f"Target scaler saved to {scaler_target_file}.")

    # Feature Selection: Remove highly correlated features
    print("Performing feature selection based on correlation...")
    correlation_matrix = pd.DataFrame(scaled_features, columns=feature_columns).corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    logging.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    
    feature_columns_reduced = [col for col in feature_columns if col not in to_drop]
    scaled_features_reduced = df[feature_columns_reduced].values
    scaler_features_reduced = MinMaxScaler(feature_range=(0, 1))
    scaled_features_reduced = scaler_features_reduced.fit_transform(scaled_features_reduced)
    
    # Update feature_columns and save reduced scaler
    feature_columns = feature_columns_reduced
    scaler_features_reduced_file = os.path.join(MODELS_DIR, 'scaler_features_reduced.joblib')
    joblib.dump(scaler_features_reduced, scaler_features_reduced_file)
    logging.info(f"Reduced feature scaler saved to {scaler_features_reduced_file}.")

    # Create sequences
    print("Creating sequences for LSTM...")
    logging.info("Creating sequences for LSTM.")
    
    X, y = [], []
    for i in range(seq_length, len(scaled_features_reduced)):
        X.append(scaled_features_reduced[i-seq_length:i])
        y.append(scaled_target[i])
    
    X, y = np.array(X), np.array(y)
    print(f"Data preprocessed: {X.shape} samples, {y.shape} targets.")
    logging.info(f"Data preprocessed: {X.shape} samples, {y.shape} targets.")
    return X, y, scaler_target_file, feature_columns

def build_simpler_lstm_model(input_shape):
    """
    Build and compile a simpler LSTM model for regression.
    """
    print("Building the simpler LSTM model...")
    logging.info("Building the simpler LSTM model.")
    
    model = Sequential()
    model.add(LSTM(units=50, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1))  # Linear activation for regression
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    print("Simpler LSTM model built and compiled.")
    logging.info("Simpler LSTM model built and compiled.")
    return model

def plot_training_history(history, model_name='LSTM_Regressor'):
    """
    Plot the training and validation loss.
    """
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, f'{model_name}_training_loss.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training loss plot saved to {plot_path}.")
    logging.info(f"Training loss plot saved to {plot_path}.")

def evaluate_regression_model(model, X_test, y_test, scaler_target_file, model_name='LSTM_Regressor'):
    """
    Make predictions and evaluate the regression model.
    """
    print("Evaluating the model...")
    logging.info("Evaluating the model.")
    
    # Load the target scaler
    scaler_target = joblib.load(scaler_target_file)
    
    predictions = model.predict(X_test)
    predictions = scaler_target.inverse_transform(predictions)
    y_test_inv = scaler_target.inverse_transform(y_test)
    
    mae = mean_absolute_error(y_test_inv, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
    r2 = r2_score(y_test_inv, predictions)
    
    print(f"\n{model_name} Performance on Test Set:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    logging.info(f"{model_name} Performance: MAE={mae:.2f}, RMSE={rmse:.2f}, R2_Score={r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(14,7))
    plt.plot(y_test_inv, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f'Actual vs Predicted Prices - {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, f'actual_vs_predicted_{model_name}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Actual vs Predicted plot saved to {plot_path}.")
    logging.info(f"Actual vs Predicted plot saved to {plot_path}.")

def save_regression_model(model, model_name='LSTM_Regressor'):
    """
    Save the trained regression LSTM model.
    """
    # Using .keras extension as per previous correction
    model_file = os.path.join(MODELS_DIR, f'best_model_{model_name}.keras')
    model.save(model_file)
    print(f"LSTM regression model saved to {model_file}.")
    logging.info(f"LSTM regression model saved to {model_file}.")

def train_baseline_model(X_train, y_train, X_test, y_test, scaler_target_file):
    """
    Train and evaluate a baseline Linear Regression model.
    """
    print("\nTraining and evaluating the baseline Linear Regression model...")
    logging.info("Training and evaluating the baseline Linear Regression model.")
    
    # Reshape data for Linear Regression
    X_train_lr = X_train.reshape(X_train.shape[0], -1)
    X_test_lr = X_test.reshape(X_test.shape[0], -1)
    y_train_lr = y_train
    y_test_lr = y_test

    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train_lr)

    y_pred_lr = lr_model.predict(X_test_lr)

    # Load the target scaler
    scaler_target = joblib.load(scaler_target_file)
    y_pred_lr_inv = scaler_target.inverse_transform(y_pred_lr)
    y_test_inv = scaler_target.inverse_transform(y_test_lr)

    mae_lr = mean_absolute_error(y_test_inv, y_pred_lr_inv)
    rmse_lr = np.sqrt(mean_squared_error(y_test_inv, y_pred_lr_inv))
    r2_lr = r2_score(y_test_inv, y_pred_lr_inv)

    print(f"Linear Regression Performance:")
    print(f"Mean Absolute Error (MAE): {mae_lr:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_lr:.2f}")
    print(f"R² Score: {r2_lr:.4f}")

    logging.info(f"Linear Regression Performance: MAE={mae_lr:.2f}, RMSE={rmse_lr:.2f}, R2_Score={r2_lr:.4f}")

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    # Load data
    data_file = os.path.join(FEATURES_DIR, 'bitcoin_data_with_sentiment.csv')
    df = load_data(data_file)

    # Define target for regression (e.g., 7 days ahead)
    prediction_horizon = 7  # Number of days ahead to predict
    df['Target_Price'] = df['price'].shift(-prediction_horizon)
    df.dropna(inplace=True)

    # Define feature columns (exclude 'Target_Price', 'date', and any other non-feature columns)
    non_feature_columns = ['Target_Price', 'date']
    feature_columns = [col for col in df.columns if col not in non_feature_columns]

    # Preprocess data
    X, y, scaler_target_file, feature_columns = preprocess_data_regression(df, feature_columns, 'Target_Price', seq_length=7)

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize lists to store metrics
    mae_list = []
    rmse_list = []
    r2_list = []

    # Initialize lists to store baseline metrics
    mae_lr_list = []
    rmse_lr_list = []
    r2_lr_list = []

    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"\nStarting Fold {fold}...")
        logging.info(f"Starting Fold {fold}.")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"Fold {fold} - Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
        logging.info(f"Fold {fold} - Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

        # Build the simpler LSTM model
        model = build_simpler_lstm_model((X_train.shape[1], X_train.shape[2]))

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f'best_model_regression_fold_{fold}.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )

        # Plot training history
        plot_training_history(history, model_name=f'LSTM_Regressor_Fold_{fold}')

        # Evaluate the model
        evaluate_regression_model(model, X_test, y_test, scaler_target_file, model_name=f'LSTM_Regressor_Fold_{fold}')

        # Save the model (already saved via ModelCheckpoint)
        save_regression_model(model, model_name=f'LSTM_Regressor_Fold_{fold}')

        # Train and evaluate the baseline model
        train_baseline_model(X_train, y_train, X_test, y_test, scaler_target_file)

        fold += 1

    print("\nRegression script execution completed successfully.")
    logging.info("Regression script execution completed successfully.")
