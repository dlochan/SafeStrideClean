import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def make_ridge(alpha: float = 1.0) -> Pipeline:
    """
    Create a Ridge regression pipeline with standard scaling.

    Parameters:
    - alpha: float : Regularization strength.

    Returns:
    - Pipeline : A scikit-learn pipeline with StandardScaler and Ridge.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])


def make_random_forest(n_estimators: int = 200, max_depth: int = None, random_state: int = 42) -> RandomForestRegressor:
    """
    Create a Random Forest Regressor.

    Parameters:
    - n_estimators: int : Number of trees in the forest.
    - max_depth: int : Maximum depth of the tree.
    - random_state: int : Random seed for reproducibility.

    Returns:
    - RandomForestRegressor : A configured RandomForestRegressor.
    """
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)


def train_model(model, X_train, y_train):
    """
    Train the model with the provided training data.

    Parameters:
    - model: : The model to train.
    - X_train: : Training features.
    - y_train: : Training target.

    Returns:
    - : Fitted model.
    """
    return model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test, time_s=None, outdir: str = "out") -> dict:
    """
    Evaluate the model and save evaluation plots and metrics.

    Parameters:
    - model: : Trained model.
    - X_test: : Test features.
    - y_test: : Test target.
    - time_s: : Optional time series for plotting.
    - outdir: str : Directory to save outputs.

    Returns:
    - dict : Evaluation metrics.
    """
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    rmse_percent_bw = (rmse / np.mean(y_test)) * 100
    pearson_r = np.corrcoef(y_test, predictions)[0, 1]

    # Save plots
    if time_s is not None:
        plt.figure()
        plt.plot(time_s, y_test, label='True')
        plt.plot(time_s, predictions, label='Predicted')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Predicted vs True Values')
        plt.savefig(f'{outdir}/pred_vs_truth.png')
        plt.close()
    else:
        plt.figure()
        plt.scatter(y_test, predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predicted vs True Values')
        plt.savefig(f'{outdir}/pred_vs_truth.png')
        plt.close()

    residuals = y_test - predictions
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title('Residuals Histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig(f'{outdir}/residuals.png')
    plt.close()

    # Write metrics
    metrics = {
        'rmse_percent_bw': rmse_percent_bw,
        'pearson_r': pearson_r
    }
    with open(f'{outdir}/metrics.json', 'w') as f:
        json.dump(metrics, f)

    return metrics


def save_model(model, path: str):
    """
    Save the model to a file using joblib.

    Parameters:
    - model: : The model to save.
    - path: str : Path to save the model.
    """
    joblib.dump(model, path)


def load_model(path: str):
    """
    Load a model from a file using joblib.

    Parameters:
    - path: str : Path to the model file.

    Returns:
    - : Loaded model.
    """
    return joblib.load(path)
    
def make_model(kind="rf"):
    if kind == "rf":
        return make_random_forest()  # your existing RF setup
    elif kind == "ridge":
        return make_ridge()
    elif kind == "hgb":
        hgb = HistGradientBoostingRegressor(
            max_depth=None,
            max_iter=800,
            learning_rate=0.05,
            l2_regularization=1e-3,
            random_state=42
        )
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", hgb)
        ])
    else:
        raise ValueError(f"Unknown model kind: {kind}")
