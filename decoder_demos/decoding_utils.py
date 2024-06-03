import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

TRAIN_TEST = (0.8, 0.2)

def zscore_data(data): 
    """
    Z-scores the input data.

    Parameters:
    data (np.ndarray): The input data.

    Returns:
    np.ndarray: The z-scored data.
    """
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    s[s==0] = 1
    return (data - m) / s

def generate_lagged_matrix(input_matrix: np.ndarray, lag: int):
    """
    Generate a lagged version of an input matrix.

    Parameters:
    input_matrix (np.ndarray): The input matrix.
    lag (int): The number of lags to consider.

    Returns:
    np.ndarray: The lagged matrix.
    """
    # Initialize the lagged matrix
    lagged_matrix = np.zeros((input_matrix.shape[0] - lag, input_matrix.shape[1] * (lag + 1)))

    # Fill the lagged matrix
    for i in range(lag + 1):
        lagged_matrix[:, i*input_matrix.shape[1]:(i+1)*input_matrix.shape[1]] = input_matrix[lag-i : (-i if i != 0 else None)]

    return lagged_matrix


def apply_neural_behavioral_lag(neural_matrix: np.ndarray, behavioral_matrix: np.ndarray, lag: int):
    """
    Apply a lag to the neural matrix and the behavioral matrix.

    Parameters:
    neural_matrix (np.ndarray): The neural matrix. (Time x Channels)
    behavioral_matrix (np.ndarray): The behavioral matrix. (Time x Dimensions)
    lag (int): The number of bins to consider.

    Returns:
    np.ndarray: The lagged neural matrix.
    np.ndarray: The lagged behavioral matrix.
    """

    if lag != 0:
        # Apply the lag to the neural matrix
        neural_matrix = neural_matrix[:-lag, :]

        # Apply the lag to the behavioral matrix
        behavioral_matrix = behavioral_matrix[lag:, :]

    return neural_matrix, behavioral_matrix


def fit_and_eval_decoder(
    train_rates: np.ndarray,
    train_behavior: np.ndarray,
    eval_rates: np.ndarray,
    eval_behavior: np.ndarray,
    grid_search: bool=True,
    param_grid: np.ndarray=np.logspace(-5, 5, 20),
    return_preds: bool=False
):
    """Fits ridge regression on train data passed
    in and evaluates on eval data

    Parameters
    ----------
    train_rates :
        2d array time x units.
    train_behavior :
        2d array time x output dims.
    eval_rates :
        2d array time x units
    eval_behavior :
        2d array time x output dims
    grid_search :
        Whether to perform a cross-validated grid search to find
        the best regularization hyperparameters.

    Returns
    -------
    float
        Uniform average R2 score on eval data
    """
    if np.any(np.isnan(train_behavior)):
        train_rates = train_rates[~np.isnan(train_behavior)[:, 0]]
        train_behavior = train_behavior[~np.isnan(train_behavior)[:, 0]]
    if np.any(np.isnan(eval_behavior)):
        eval_rates = eval_rates[~np.isnan(eval_behavior)[:, 0]]
        eval_behavior = eval_behavior[~np.isnan(eval_behavior)[:, 0]]
    assert not np.any(np.isnan(train_rates)) and not np.any(np.isnan(eval_rates)), \
        "fit_and_eval_decoder: NaNs found in rate predictions within required trial times"

    if grid_search:
        decoder = GridSearchCV(Ridge(), {"alpha": param_grid})
    else:
        decoder = Ridge(alpha=1e-2)
    decoder.fit(train_rates, train_behavior)
    if return_preds:
        return decoder.score(eval_rates, eval_behavior), decoder, decoder.predict(eval_rates)
    else:
        return decoder.score(eval_rates, eval_behavior), decoder
