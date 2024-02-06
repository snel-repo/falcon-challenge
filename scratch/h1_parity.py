#%%
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

def fit_and_eval_decoder(
    train_rates,
    train_behavior,
    eval_rates,
    eval_behavior,
    grid_search=True,
):
    """Fits ridge regression on train data passed
    in and evaluates on eval data

    Parameters
    ----------
    train_rates : np.ndarray
        2d array with 1st dimension being samples (time) and
        2nd dimension being input variables (units).
        Used to train regressor
    train_behavior : np.ndarray
        2d array with 1st dimension being samples (time) and
        2nd dimension being output variables (channels).
        Used to train regressor
    eval_rates : np.ndarray
        2d array with same dimension ordering as train_rates.
        Used to evaluate regressor
    eval_behavior : np.ndarray
        2d array with same dimension ordering as train_behavior.
        Used to evaluate regressor
    grid_search : bool
        Whether to perform a cross-validated grid search to find
        the best regularization hyperparameters.

    Returns
    -------
    float
        R2 score on eval data
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
        decoder = GridSearchCV(Ridge(), {"alpha": np.logspace(-1, 6, 21)})
    else:
        # Create a ridge regression model with feature scaling
        # decoder = make_pipeline(StandardScaler(), Ridge(alpha=1000)) # irrelevant, features are normalized
        decoder = Ridge(alpha=1000, solver='svd')
    decoder.fit(train_rates, train_behavior)
    return decoder.score(eval_rates, eval_behavior), decoder

test = loadmat('stability_debug.mat', simplify_cells=True)
train_x = test['NeuralSig']
test_x = test['NeuralSig_val']
train_y = test['KinematicSig']
test_y = test['KinematicSig_val']
score, decoder = fit_and_eval_decoder(train_x, train_y, test_x, test_y)
# score, decoder = fit_and_eval_decoder(train_x, train_y, test_x, test_y, grid_search=False)
pred_y = decoder.predict(test_x)
print(score)

#%%
# Explicit solve - something isn't making sense
lambda_ = 1000
X_with_intercept = np.hstack([np.ones((train_x.shape[0], 1)), train_x])

# Compute the ridge regression coefficients using the normal equation
lambda_identity = lambda_ * np.identity(X_with_intercept.shape[1])
coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept + lambda_identity) @ X_with_intercept.T @ train_y

# Make prediction
test_x_with_intercept = np.hstack([np.ones((test_x.shape[0], 1)), test_x])
pred_y = test_x_with_intercept @ coefficients
#%%
SSR = np.sum((test_y - pred_y)**2, 0)
SST = np.sum((test_y - test_y.mean(axis=0))**2, 0)
print((SSR / SST).mean())
#%%
# ! Parity line
# We have parity. The underlying diff is that R2 computed by corrcoef ** 2 doesn't seem to match
# any twisting of the below R2.

# Not to be confused with predictions of individual R2
flat_r2 = 1 - np.sum((test_y - pred_y)**2) / np.sum((test_y - test_y.mean(axis=0))**2)
print(flat_r2)
from scipy.stats import pearsonr
matlab_parity_corr_coef = pearsonr(pred_y.flatten(), test_y.flatten())[0]
print(matlab_parity_corr_coef**2)

# Plot dynamic range of each dimension to make sense of disparity from flattening
import matplotlib.pyplot as plt
f, ax = plt.subplots(test_y.shape[1], figsize=(5, 15))
for i in range(test_y.shape[1]):
    ax[i].plot(pred_y[:, i], label='pred')
    ax[i].plot(test_y[:, i], label='true')
    ax[i].legend()
    # Note per-dim r2
    r2 = 1 - np.sum((test_y[:, i] - pred_y[:, i])**2) / np.sum((test_y[:, i] - test_y[:, i].mean())**2)
    ax[i].set_title(f'R2: {r2:.2f}')
f.tight_layout()

# ! Conclusion - we don't have very good decoding with OLE - even given the full Pitt processing pipeline
# ! Don't expect to see much better
# ? Can we maintain this mediocre perf without Pitt preprocessing?
#%%
# Now test iOLE, which achieves a higher r
data = loadmat('stability_debug_iOLE.mat', simplify_cells=True)
baselines = data['baselines']
weights = data['weights']
#%%
print(weights.shape)
print(train_x.shape)
#%%
print(baselines.shape)
pred = np.dot(test_x - baselines, weights)
r2 = 1 - np.sum((test_y - pred)**2) / np.sum((test_y - test_y.mean(axis=0))**2)
print(r2) # 0.104