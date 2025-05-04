import pandas as pd
import numpy as np
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# -------------------------------
# Configuration and Setup
# -------------------------------

# Define the directories
OUTPUT_DIR = 'D:/bitcoin/'
FEATURES_DIR = os.path.join(OUTPUT_DIR, 'features')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
EVALUATION_DIR = os.path.join(OUTPUT_DIR, 'evaluation')

# Create evaluation directory if it doesn't exist
if not os.path.exists(EVALUATION_DIR):
    os.makedirs(EVALUATION_DIR)
    print(f"Created evaluation directory: {EVALUATION_DIR}")
else:
    print(f"Using existing evaluation directory: {EVALUATION_DIR}")

# Configure logging
LOG_FILE = os.path.join(EVALUATION_DIR, 'model_evaluation.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# -------------------------------
# Function Definitions
# -------------------------------

def list_saved_models():
    """
    List all saved models in the models directory.
    """
    print("\n--- Listing Saved Models ---")
    models = []
    for file in os.listdir(MODELS_DIR):
        if file.endswith('.joblib'):
            models.append(file)
    if not models:
        print("No models found in the models directory.")
        logging.warning("No models found in the models directory.")
    else:
        print("Available Models:")
        for idx, model in enumerate(models, 1):
            print(f"{idx}. {model}")
    return models

def load_model(model_filename):
    """
    Load a saved model from the models directory.
    """
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}.")
        logging.error(f"Model file not found at {model_path}.")
        return None
    try:
        model = joblib.load(model_path)
        print(f"Loaded model '{model_filename}' successfully.")
        logging.info(f"Loaded model '{model_filename}' successfully.")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        logging.error(f"An error occurred while loading the model: {e}")
        return None

def load_data():
    """
    Load the feature-engineered data from the features directory.
    """
    engineered_csv = os.path.join(FEATURES_DIR, 'feature_engineered_data.csv')
    try:
        df = pd.read_csv(engineered_csv, parse_dates=['date'], index_col='date')
        logging.info("Loaded feature-engineered data successfully.")
        print("Loaded feature-engineered data successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"Feature-engineered data file not found at {engineered_csv}.")
        print(f"Feature-engineered data file not found at {engineered_csv}. Please ensure the feature engineering step was successful.")
        return None
    except Exception as err:
        logging.error(f"An error occurred while loading the feature-engineered data: {err}")
        print(f"An error occurred while loading the feature-engineered data: {err}")
        return None

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

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def plot_feature_importance(model, X_train, model_name):
    """
    Plot feature importances for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_names = X_train.columns
        feature_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=feature_imp[:20], y=feature_imp.index[:20])
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plot_path = os.path.join(EVALUATION_DIR, f'feature_importance_{model_name.replace(" ", "_")}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to {plot_path}.")
        logging.info(f"Feature importance plot saved to {plot_path}.")
    else:
        print(f"{model_name} does not have feature_importances_ attribute.")
        logging.warning(f"{model_name} does not have feature_importances_ attribute.")

def plot_residuals(y_true, y_pred, model_name):
    """
    Plot residuals to evaluate model performance.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residual Plot - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, f'residual_plot_{model_name.replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Residual plot saved to {plot_path}.")
    logging.info(f"Residual plot saved to {plot_path}.")

def perform_cross_validation(model, X_train, y_train, cv=5):
    """
    Perform cross-validation and return the scores.
    """
    print("\n--- Performing Cross-Validation ---")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    print(f"Cross-Validation RMSE Scores: {rmse_scores}")
    print(f"Mean CV RMSE: {rmse_scores.mean():.2f}")
    print(f"Standard Deviation of CV RMSE: {rmse_scores.std():.2f}")
    
    logging.info(f"Cross-Validation RMSE Scores: {rmse_scores}")
    logging.info(f"Mean CV RMSE: {rmse_scores.mean():.2f}")
    logging.info(f"Standard Deviation of CV RMSE: {rmse_scores.std():.2f}")
    
    # Plotting CV Scores
    plt.figure(figsize=(8,6))
    sns.boxplot(y=rmse_scores)
    plt.title('Cross-Validation RMSE Scores')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plot_path = os.path.join(EVALUATION_DIR, 'cross_validation_rmse.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Cross-validation RMSE boxplot saved to {plot_path}.")
    logging.info(f"Cross-validation RMSE boxplot saved to {plot_path}.")
    
    return rmse_scores

def document_evaluation(evaluation_results, cross_val_scores, model_name):
    """
    Document evaluation results into CSV files.
    """
    # Save evaluation metrics
    evaluation_df = pd.DataFrame(evaluation_results).T
    evaluation_df.to_csv(os.path.join(EVALUATION_DIR, 'model_evaluation_metrics.csv'))
    logging.info("Model evaluation metrics saved to 'model_evaluation_metrics.csv'.")
    
    # Save cross-validation scores
    cv_df = pd.DataFrame({
        'Model': [model_name],
        'CV_RMSE_Mean': [cross_val_scores.mean()],
        'CV_RMSE_STD': [cross_val_scores.std()]
    })
    cv_df.to_csv(os.path.join(EVALUATION_DIR, 'cross_validation_scores.csv'), index=False)
    logging.info("Cross-validation scores saved to 'cross_validation_scores.csv'.")
    print("\nEvaluation metrics documented successfully.")

def model_evaluation():
    """
    Perform comprehensive model evaluation and selection.
    """
    print("\n=== Step 5.3: Model Evaluation and Selection ===")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Define target variable
    df = define_target_variable(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    
    # List all saved models
    models = list_saved_models()
    if not models:
        print("No models available for evaluation. Please ensure models are trained and saved correctly.")
        return
    
    # Prompt user to select a model
    print("\nSelect the model you want to evaluate:")
    for idx, model_file in enumerate(models, 1):
        print(f"{idx}. {model_file}")
    choice = input("Enter the number corresponding to the model: ").strip()
    
    try:
        choice = int(choice)
        if choice < 1 or choice > len(models):
            raise ValueError
        selected_model_file = models[choice - 1]
    except ValueError:
        print("Invalid input. Please enter a valid number corresponding to the model.")
        logging.error("Invalid model selection input.")
        return
    
    # Load the selected model
    model = load_model(selected_model_file)
    if model is None:
        return
    
    # Make predictions on the test set
    print(f"\n--- Making Predictions with {selected_model_file} ---")
    try:
        y_pred = model.predict(X_test)
        print("Predictions made successfully.")
        logging.info(f"Predictions made with model '{selected_model_file}'.")
    except Exception as e:
        print(f"An error occurred while making predictions: {e}")
        logging.error(f"An error occurred while making predictions: {e}")
        return
    
    # Calculate evaluation metrics
    print("\n--- Calculating Evaluation Metrics ---")
    try:
        mae, rmse, r2 = calculate_metrics(y_test, y_pred)
        print(f"{selected_model_file} Performance on Test Set:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        logging.info(f"{selected_model_file} Performance on Test Set: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    except Exception as e:
        print(f"An error occurred while calculating metrics: {e}")
        logging.error(f"An error occurred while calculating metrics: {e}")
        return
    
    # Plot Feature Importance
    plot_feature_importance(model, X_train, selected_model_file)
    
    # Plot Residuals
    plot_residuals(y_test, y_pred, selected_model_file)
    
    # Perform Cross-Validation
    cross_val_scores = perform_cross_validation(model, X_train, y_train, cv=5)
    
    # Document Evaluation Results
    evaluation_results = {
        selected_model_file: {
            'MAE': mae,
            'RMSE': rmse,
            'R2_Score': r2
        }
    }
    document_evaluation(evaluation_results, cross_val_scores, selected_model_file)
    
    print("\n=== Step 5.3: Model Evaluation and Selection Completed ===\n")
    print("You can find the evaluation results and plots in the 'evaluation' directory within the output directory.")
    print("Logs are saved in 'model_evaluation.log'.")
    logging.info("Model evaluation and selection completed successfully.")

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    model_evaluation()
