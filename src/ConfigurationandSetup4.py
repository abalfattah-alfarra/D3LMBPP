import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -------------------------------
# Configuration and Setup
# -------------------------------

# Define the output directories
OUTPUT_DIR = 'D:/bitcoin/'
FEATURES_DIR = os.path.join(OUTPUT_DIR, 'features')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# Create directories if they don't exist
for directory in [FEATURES_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Using existing directory: {directory}")

# Configure logging
LOG_FILE = os.path.join(OUTPUT_DIR, 'model_development.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# -------------------------------
# Function Definitions
# -------------------------------

def define_target_variable(df):
    """
    Define the target variable for prediction.
    """
    print("\n--- Defining Target Variable ---")
    df['Target_Price'] = df['price'].shift(-1)  # Next day's price
    df.dropna(inplace=True)  # Drop the last row with NaN target
    print("Target variable 'Target_Price' defined successfully.")
    logging.info("Target variable 'Target_Price' defined successfully.")
    return df

def split_data(df, test_size=0.2):
    """
    Split the dataset into training and testing sets.
    """
    print("\n--- Splitting Data into Training and Testing Sets ---")
    
    X = df.drop(['Target_Price'], axis=1)
    y = df['Target_Price']
    
    split_index = int(len(df) * (1 - test_size))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"Data split into {int((1 - test_size)*100)}% training and {int(test_size*100)}% testing.")
    logging.info(f"Data split into {int((1 - test_size)*100)}% training and {int(test_size*100)}% testing.")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """
    Train multiple machine learning models.
    """
    print("\n--- Training Machine Learning Models ---")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"{name} trained successfully.")
            logging.info(f"{name} trained successfully.")
        except Exception as e:
            print(f"An error occurred while training {name}: {e}")
            logging.error(f"An error occurred while training {name}: {e}")
    
    print("\nAll models trained successfully.")
    logging.info("All models trained successfully.")
    return trained_models

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate the performance of trained models on the test set.
    """
    print("\n--- Evaluating Model Performance ---")
    
    evaluation_results = {}
    
    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        try:
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            evaluation_results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2_Score': r2
            }
            
            print(f"{name} Performance:")
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            print(f"R² Score: {r2:.4f}")
            
            logging.info(f"{name} Performance: MAE={mae:.2f}, RMSE={rmse:.2f}, R2_Score={r2:.4f}")
        except Exception as e:
            print(f"An error occurred while evaluating {name}: {e}")
            logging.error(f"An error occurred while evaluating {name}: {e}")
    
    print("\nModel evaluation completed successfully.")
    logging.info("Model evaluation completed successfully.")
    return evaluation_results

def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning on Random Forest using Grid Search.
    """
    print("\n--- Hyperparameter Tuning for Random Forest ---")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=2
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        print(f"Best parameters for Random Forest: {grid_search.best_params_}")
        logging.info(f"Best parameters for Random Forest: {grid_search.best_params_}")
        return best_rf
    except Exception as e:
        print(f"An error occurred during hyperparameter tuning: {e}")
        logging.error(f"An error occurred during hyperparameter tuning: {e}")
        return None

def select_and_save_best_model(evaluation_results, trained_models):
    """
    Select the best model based on RMSE and save it.
    """
    print("\n--- Selecting the Best Model ---")
    
    # Convert evaluation_results to DataFrame for easier comparison
    eval_df = pd.DataFrame(evaluation_results).T
    best_model_name = eval_df['RMSE'].idxmin()
    best_model = trained_models[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with RMSE: {eval_df.loc[best_model_name, 'RMSE']:.2f}")
    logging.info(f"Best Model: {best_model_name} with RMSE: {eval_df.loc[best_model_name, 'RMSE']:.2f}")
    
    # Save the best model
    model_file = os.path.join(MODELS_DIR, f'best_model_{best_model_name.replace(" ", "_")}.joblib')
    try:
        joblib.dump(best_model, model_file)
        print(f"Best model '{best_model_name}' saved successfully to {model_file}.")
        logging.info(f"Best model '{best_model_name}' saved successfully to {model_file}.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        logging.error(f"An error occurred while saving the model: {e}")
    
    return best_model_name, best_model

def model_development():
    """
    Perform model development including training, evaluation, and selection.
    """
    print("\n=== Step 5.2: Model Development ===")
    
    # Define the path to the feature-engineered CSV file
    engineered_csv = os.path.join(FEATURES_DIR, 'feature_engineered_data.csv')
    
    # Load the feature-engineered data
    try:
        df = pd.read_csv(engineered_csv, parse_dates=['date'], index_col='date')
        print("\nFirst 5 rows of the feature-engineered dataset:")
        print(df.head())
        logging.info("Feature-engineered data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Feature-engineered data file not found at {engineered_csv}.")
        print(f"Feature-engineered data file not found at {engineered_csv}. Please ensure the feature engineering step was successful.")
        return
    except Exception as err:
        logging.error(f"An error occurred while loading the feature-engineered data: {err}")
        print(f"An error occurred while loading the feature-engineered data: {err}")
        return
    
    # Define Target Variable
    df = define_target_variable(df)
    
    # Split Data
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    
    # Train Models
    trained_models = train_models(X_train, y_train)
    
    # Evaluate Models
    evaluation_results = evaluate_models(trained_models, X_test, y_test)
    
    # Hyperparameter Tuning for Random Forest (Optional)
    print("\nWould you like to perform hyperparameter tuning for Random Forest? (yes/no)")
    choice = input().strip().lower()
    if choice == 'yes':
        best_rf = hyperparameter_tuning(X_train, y_train)
        if best_rf:
            # Retrain Random Forest with best parameters
            print("\nRetraining Random Forest with best parameters...")
            try:
                best_rf.fit(X_train, y_train)
                trained_models['Random Forest Tuned'] = best_rf
                print("Random Forest retrained successfully with best parameters.")
                logging.info("Random Forest retrained successfully with best parameters.")
            except Exception as e:
                print(f"An error occurred while retraining Random Forest: {e}")
                logging.error(f"An error occurred while retraining Random Forest: {e}")
        
            # Re-evaluate models
            evaluation_results = evaluate_models(trained_models, X_test, y_test)
    else:
        print("\nSkipping hyperparameter tuning.")
        logging.info("Hyperparameter tuning skipped.")
    
    # Select and Save Best Model
    best_model_name, best_model = select_and_save_best_model(evaluation_results, trained_models)
    
    print("\n=== Step 5.2: Model Development Completed ===\n")
    print("You can find the best model in the 'models' directory within the output directory.")
    print("Logs are saved in 'model_development.log'.")
    
    logging.info("Model development completed successfully.")
    
    return best_model_name, best_model

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    best_model_name, best_model = model_development()
