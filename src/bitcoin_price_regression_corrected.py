# bitcoin_price_regression_revised_improved.py

import pandas as pd
import numpy as np
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

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

def remove_highly_correlated_features(df, feature_columns, threshold=0.95, min_features=10):
    """
    Remove features that are highly correlated above the given threshold.
    Ensures that at least 'min_features' remain after dropping.
    """
    print("Removing highly correlated features...")
    logging.info("Removing highly correlated features.")
    
    # Compute the correlation matrix
    correlation_matrix = df[feature_columns].corr().abs()
    
    # Select the upper triangle of the correlation matrix
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Adjust the list to drop only as many as possible without going below 'min_features'
    max_features_to_drop = len(feature_columns) - min_features
    if len(to_drop) > max_features_to_drop:
        to_drop = to_drop[:max_features_to_drop]
        print(f"Adjusted the number of features to drop to {len(to_drop)} to maintain at least {min_features} features.")
        logging.info(f"Adjusted the number of features to drop to {len(to_drop)} to maintain at least {min_features} features.")
    
    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    logging.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    
    # Drop the features
    feature_columns_reduced = [col for col in feature_columns if col not in to_drop]
    df_reduced = df[feature_columns_reduced]
    
    print(f"Number of features before dropping: {len(feature_columns)}")
    print(f"Number of features after dropping: {len(feature_columns_reduced)}")
    logging.info(f"Number of features before dropping: {len(feature_columns)}")
    logging.info(f"Number of features after dropping: {len(feature_columns_reduced)}")
    
    return df_reduced, feature_columns_reduced

def perform_feature_selection(X_train, y_train, feature_names, k=10):
    """
    Perform feature selection using SelectKBest with mutual information.
    Returns the selected feature indices and names.
    """
    print("Performing feature selection using SelectKBest...")
    logging.info("Performing feature selection using SelectKBest.")
    
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    selector.fit(X_train, y_train)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"Selected Features: {selected_features}")
    logging.info(f"Selected Features: {selected_features}")
    
    X_train_selected = selector.transform(X_train)
    
    # Save the selector for future use
    selector_file = os.path.join(MODELS_DIR, f'selectkbest_fold.joblib')
    joblib.dump(selector, selector_file)
    logging.info(f"SelectKBest transformer saved to {selector_file}.")
    
    return X_train_selected, selected_features

def create_sequences(X, y, seq_length=7):
    """
    Create sequences for LSTM input.
    """
    print(f"Creating sequences with sequence length = {seq_length}...")
    logging.info(f"Creating sequences with sequence length = {seq_length}.")
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    print(f"Created sequences: {X_seq.shape} samples, {y_seq.shape} targets.")
    logging.info(f"Created sequences: {X_seq.shape} samples, {y_seq.shape} targets.")
    return X_seq, y_seq

def build_optimized_lstm(input_shape):
    """
    Build and compile an optimized LSTM model with Bidirectional layers.
    """
    print("Building the optimized LSTM model...")
    logging.info("Building the optimized LSTM model.")
    
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(LSTM(units=32, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))  # Linear activation for regression
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    print("Optimized LSTM model built and compiled.")
    logging.info("Optimized LSTM model built and compiled.")
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

def plot_residuals(y_true, y_pred, model_name):
    """
    Plot residuals (actual - predicted) to analyze model errors.
    """
    residuals = y_true.flatten() - y_pred.flatten()
    plt.figure(figsize=(10,6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title(f'Residuals Distribution for {model_name}')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, f'residuals_distribution_{model_name}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Residuals distribution plot saved to {plot_path}.")
    logging.info(f"Residuals distribution plot saved to {plot_path}.")

def evaluate_regression_model(model, X_test, y_test, scaler_target_file, model_name='LSTM_Regressor'):
    """
    Make predictions and evaluate the regression model.
    """
    print("Evaluating the model...")
    logging.info("Evaluating the model.")
    
    if scaler_target_file:
        # Load the target scaler
        scaler_target = joblib.load(scaler_target_file)
        
        predictions = model.predict(X_test)
        predictions = scaler_target.inverse_transform(predictions)
        y_test_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    else:
        # If no scaler is provided, use the raw predictions
        predictions = model.predict(X_test)
        y_test_inv = y_test.reshape(-1, 1)
    
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
    
    # Plot residuals
    plot_residuals(y_test_inv, predictions, model_name)

def save_regression_model(model, model_name='LSTM_Regressor'):
    """
    Save the trained regression LSTM model.
    """
    # Using .keras extension as per previous correction
    model_file = os.path.join(MODELS_DIR, f'best_model_{model_name}.keras')
    model.save(model_file)
    print(f"{model_name} model saved to {model_file}.")
    logging.info(f"{model_name} model saved to {model_file}.")

def train_baseline_model(X_train, y_train, X_test, y_test, scaler_target_file=None):
    """
    Train and evaluate a baseline Linear Regression and XGBoost model.
    """
    print("\nTraining and evaluating the baseline Linear Regression model...")
    logging.info("Training and evaluating the baseline Linear Regression model.")
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    
    if scaler_target_file:
        # Load the target scaler
        scaler_target = joblib.load(scaler_target_file)
        y_pred_lr_inv = scaler_target.inverse_transform(y_pred_lr.reshape(-1, 1))
        y_test_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    else:
        # If no scaler is provided, use the raw predictions
        y_pred_lr_inv = y_pred_lr
        y_test_inv = y_test.reshape(-1, 1)
    
    mae_lr = mean_absolute_error(y_test_inv, y_pred_lr_inv)
    rmse_lr = np.sqrt(mean_squared_error(y_test_inv, y_pred_lr_inv))
    r2_lr = r2_score(y_test_inv, y_pred_lr_inv)
    
    print(f"Linear Regression Performance:")
    print(f"Mean Absolute Error (MAE): {mae_lr:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_lr:.2f}")
    print(f"R² Score: {r2_lr:.4f}")
    
    logging.info(f"Linear Regression Performance: MAE={mae_lr:.2f}, RMSE={rmse_lr:.2f}, R2_Score={r2_lr:.4f}")
    
    # Plot actual vs predicted for Linear Regression
    plt.figure(figsize=(14,7))
    plt.plot(y_test_inv, label='Actual Price')
    plt.plot(y_pred_lr_inv, label='Predicted Price (Linear Regression)')
    plt.title('Actual vs Predicted Prices - Linear Regression')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path_lr = os.path.join(EVALUATION_DIR, 'actual_vs_predicted_Linear_Regression.png')
    plt.savefig(plot_path_lr)
    plt.close()
    print(f"Actual vs Predicted plot for Linear Regression saved to {plot_path_lr}.")
    logging.info(f"Actual vs Predicted plot for Linear Regression saved to {plot_path_lr}.")
    
    # Residuals for Linear Regression
    plot_residuals(y_test_inv, y_pred_lr_inv, 'Linear_Regression')
    
    # XGBoost Regression
    print("\nTraining and evaluating the baseline XGBoost Regressor model...")
    logging.info("Training and evaluating the baseline XGBoost Regressor model.")
    
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    model_xgb.fit(X_train, y_train.flatten())
    
    y_pred_xgb = model_xgb.predict(X_test)
    
    if scaler_target_file:
        y_pred_xgb_inv = scaler_target.inverse_transform(y_pred_xgb.reshape(-1, 1))
    else:
        y_pred_xgb_inv = y_pred_xgb
    
    mae_xgb = mean_absolute_error(y_test_inv, y_pred_xgb_inv)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_inv, y_pred_xgb_inv))
    r2_xgb = r2_score(y_test_inv, y_pred_xgb_inv)
    
    print(f"XGBoost Regressor Performance:")
    print(f"Mean Absolute Error (MAE): {mae_xgb:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_xgb:.2f}")
    print(f"R² Score: {r2_xgb:.4f}")
    
    logging.info(f"XGBoost Regressor Performance: MAE={mae_xgb:.2f}, RMSE={rmse_xgb:.2f}, R2_Score={r2_xgb:.4f}")
    
    # Plot actual vs predicted for XGBoost
    plt.figure(figsize=(14,7))
    plt.plot(y_test_inv, label='Actual Price')
    plt.plot(y_pred_xgb_inv, label='Predicted Price (XGBoost)')
    plt.title('Actual vs Predicted Prices - XGBoost Regressor')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path_xgb = os.path.join(EVALUATION_DIR, 'actual_vs_predicted_XGBoost.png')
    plt.savefig(plot_path_xgb)
    plt.close()
    print(f"Actual vs Predicted plot for XGBoost Regressor saved to {plot_path_xgb}.")
    logging.info(f"Actual vs Predicted plot for XGBoost Regressor saved to {plot_path_xgb}.")
    
    # Residuals for XGBoost
    plot_residuals(y_test_inv, y_pred_xgb_inv, 'XGBoost_Regressor')

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

    # Scale the target variable
    scaler_target = MinMaxScaler()
    df['Target_Price'] = scaler_target.fit_transform(df[['Target_Price']])
    # Save the scaler for inverse transformation
    joblib.dump(scaler_target, os.path.join(MODELS_DIR, 'scaler_target.joblib'))

    # Define feature columns (exclude 'Target_Price', 'date', and any other non-feature columns)
    non_feature_columns = ['Target_Price', 'date']
    feature_columns = [col for col in df.columns if col not in non_feature_columns]

    # Remove highly correlated features
    df_reduced, feature_columns_reduced = remove_highly_correlated_features(df, feature_columns, threshold=0.95, min_features=10)

    # Separate features and target
    X = df_reduced.values
    y = df['Target_Price'].values

    # Apply Feature Selection (SelectKBest) **Before** Creating Sequences
    # This ensures that the same features are used across all timesteps within a sequence
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"\nStarting Fold {fold}...")
        logging.info(f"Starting Fold {fold}.")

        X_train, X_test = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        print(f"Fold {fold} - Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
        logging.info(f"Fold {fold} - Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

        # Feature Selection on Training Data
        # Determine k (number of features to select)
        k = min(20, X_train.shape[1])  # Select up to 20 features or the available number

        X_train_selected, selected_features = perform_feature_selection(
            X_train, y_train_fold, feature_columns_reduced, k=k
        )

        # Ensure that RFE selected at least some features
        selected_num_features = len(selected_features)
        if selected_num_features == 0:
            print(f"Fold {fold}: No features selected. Skipping this fold.")
            logging.warning(f"Fold {fold}: No features selected. Skipping this fold.")
            fold +=1
            continue

        print(f"Number of features selected: {selected_num_features}")
        logging.info(f"Number of features selected: {selected_num_features}")

        # Select the same features from the test set
        selected_feature_indices = [feature_columns_reduced.index(f) for f in selected_features]
        X_test_selected = X_test[:, selected_feature_indices]

        # Create sequences for training and testing data
        # Adjust sequence length based on data size
        seq_length = 7  # Reduced from 14 to 7 to increase training samples

        # Create sequences separately to maintain temporal integrity
        X_train_seq, y_train_seq = create_sequences(X_train_selected, y_train_fold, seq_length=seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test_selected, y_test_fold, seq_length=seq_length)

        print(f"Fold {fold} - Training sequences: {X_train_seq.shape}, Testing sequences: {X_test_seq.shape}")
        logging.info(f"Fold {fold} - Training sequences: {X_train_seq.shape}, Testing sequences: {X_test_seq.shape}")

        # Check if there are enough sequences
        if X_train_seq.shape[0] < 10 or X_test_seq.shape[0] < 10:
            print(f"Fold {fold}: Not enough sequences. Skipping this fold.")
            logging.warning(f"Fold {fold}: Not enough sequences. Skipping this fold.")
            fold +=1
            continue

        # Scale the features within the sequences
        scaler_features = StandardScaler()
        X_train_seq = X_train_seq.reshape(-1, X_train_seq.shape[2])
        X_test_seq = X_test_seq.reshape(-1, X_test_seq.shape[2])
        X_train_seq = scaler_features.fit_transform(X_train_seq)
        X_test_seq = scaler_features.transform(X_test_seq)
        X_train_seq = X_train_seq.reshape(-1, seq_length, selected_num_features)
        X_test_seq = X_test_seq.reshape(-1, seq_length, selected_num_features)

        # Build the optimized LSTM model
        model = build_optimized_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))

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
            X_train_seq, y_train_seq,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )

        # Plot training history
        plot_training_history(history, model_name=f'LSTM_Regressor_Fold_{fold}')

        # Evaluate the model
        evaluate_regression_model(model, X_test_seq, y_test_seq, scaler_target_file=os.path.join(MODELS_DIR, 'scaler_target.joblib'), model_name=f'LSTM_Regressor_Fold_{fold}')

        # Save the model (already saved via ModelCheckpoint)
        save_regression_model(model, model_name=f'LSTM_Regressor_Fold_{fold}')

        # Train and evaluate the baseline models
        # Reshape for baseline models
        X_train_lr = X_train_seq.reshape(X_train_seq.shape[0], -1)
        X_test_lr = X_test_seq.reshape(X_test_seq.shape[0], -1)
        
        train_baseline_model(
            X_train_lr, y_train_seq, 
            X_test_lr, y_test_seq, scaler_target_file=os.path.join(MODELS_DIR, 'scaler_target.joblib')
        )

        fold += 1

    print("\nRegression script execution completed successfully.")
    logging.info("Regression script execution completed successfully.")
