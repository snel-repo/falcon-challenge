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

#%%
# Archive - Pitt iOLE
# Inverse OLE - strict translation of matlab
from scipy import stats

# def ControlSpaceToDomainCells(K):

K_train = train_y  # Time x Dims
F_train = train_x  # Time x Channels
K_val = test_y
F_val = test_x
NUM_DOMAINS = 6
IgnoreIdx = np.zeros(30, dtype=bool)
IgnoreIdx[7:] = 1
# Kt = ControlSpaceToDomainCells(K_train)
# Kv = ControlSpaceToDomainCells(K_val)
# K = ControlSpaceToDomainCells(np.vstack([KinematicSig, KinematicSig_val]))
# I = ControlSpaceToDomainCells(IgnoreIdx)

invSigma = np.zeros((F_train.shape[1], F_train.shape[1], NUM_DOMAINS))
import numpy as np
from scipy import stats

# Assuming F_train is your neural signals matrix (Time x Channels)
# and K_train is your kinematic signals matrix (Time x Kinematic Dimensions)
num_kin_dims = K_train.shape[1]
num_neural_units = F_train.shape[1]
# Initialize the invSigma matrix
invSigma = np.zeros((num_neural_units, num_neural_units, num_kin_dims))

# Loop over each neural unit
for i in range(num_neural_units):
    # Loop over each kinematic dimension
    for k in range(num_kin_dims):
        # Add a column of ones to K_train for the intercept
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(K_train[:, k], F_train[:, i])

        # Calculate residuals
        residuals = F_train[:, i] - (intercept + slope * K_train[:, k])
        # Calculate variance of residuals and store its inverse
        Sigma = np.var(residuals)
        invSigma[i, i, k] = 1 / Sigma if Sigma != 0 else 0
# for i in range(train_x.shape[1]):
#     for d in range(NUM_DOMAINS):
#         _, _, _, _, residuals = stats.linregress(np.hstack([np.ones((Kt[d].shape[0], 1)), Kt[d][:, ~I[d]]]), F_train[:, i].astype(np.float32))
#         Sigma = np.var(residuals)
#         invSigma[i, i, d] = 1 / Sigma

lambda_ = 10.
lambda1 = 10. # skip cross val, no major impact in Pitt data and complex to implement

# lambda_ = np.concatenate(([0], np.logspace(-3, 6, 100)))
# lambda1 = np.concatenate(([0], np.logspace(-3, 6, 100)))
# idx1 = range(len(lambda1))
# idx = range(len(lambda_))
# metric = np.full((max(idx), max(idx1)), np.inf)

import numpy as np

# Assuming KinematicSig is the kinematic signals matrix (Time x Kinematic Dimensions)
# NeuralSig is the neural signals matrix (Time x Channels)
# lambda1 and lambda are regularization parameters
# IgnoreIdx is a boolean array indicating which kinematic dimensions to ignore

KinematicSig = K_train
NeuralSig = F_train
# Initial ridge regression
X = np.hstack([np.ones((KinematicSig.shape[0], 1)), KinematicSig])
W = np.linalg.pinv(X.T @ X + lambda1 * np.eye(KinematicSig.shape[1] + 1)) @ X.T @ NeuralSig
baselines = W[0, :]
W1 = W[1:, :]

# Adjusting for ignored indices
W1_ = np.zeros((W1.shape[1], len(IgnoreIdx)))
W1_[:, ~IgnoreIdx] = W1.T

# Ridge regression per kinematic dimension
num_kin_dims = KinematicSig.shape[1]
final_weights = []

for k in range(num_kin_dims):
    if not IgnoreIdx[k]:
        w1invSigma = W1_[:, k].T @ invSigma[:, :, k]
        Wt = np.linalg.pinv(w1invSigma @ W1_[:, k] + lambda_ * np.eye(W1_[:, k].shape[0])) @ w1invSigma
        final_weights.append(Wt.T)

# Concatenate the weights for each dimension
final_weights = np.stack(final_weights, axis=1)

from sklearn.base import BaseEstimator, RegressorMixin

class CustomPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, weights, baselines):
        self.weights = weights
        self.baselines = baselines

    def predict(self, X):
        return (X - self.baselines) @ self.weights

# Assuming final_weights and baselines are the weights and baselines computed earlier
predictor = CustomPredictor(final_weights, baselines)

# test the predictor
pred_y = predictor.predict(test_x)
print(pred_y.shape)
from sklearn.metrics import r2_score
r2 = r2_score(test_y, pred_y)
# r2 = r2_score(test_y, pred_y, multioutput='raw_values')
print(f'iOLE: {r2}')

print(train_x.shape, test_x.shape)
"""
    Parity notes:

    # PrepData
    Pitt:
    Full Prep: Original timepoint is 12922 * 2 = 25.8440
    Their split prep:
    - 12922, and 2290 (with nans in between)
        - Their val is 3 random trials (floor 20% of 18 trials = 3)

    Our dimensions:
    - Train - 20592 x C
    - Val - 5144 x C
        - Roughly 25.736s

    # OK, so extracting the relevant fields
    - Pitt:
        - 2006 non-nan bins = 4.012s
        - Pitt then further kills non-calibration (defined by TaskState) and non-moving samples, which brings to about just 800 bins of activity

    - Ours:
        -



    After data loading:
    Pitt dimensions:
    - Train - 4507 x C=175
    - Val - 815 x C
    # ! After some processing, we've lost some data, down to 4465 timepoints
    # ! And val is 861 - 5326 in total, vs 5322 before

    # ---

    - # ! How is their split computed? - Make parity on their side
    - # ! Why do they have so few timepoints?
    - Get parity on channels!
    - Looks like their crossval uses lambda=1e6
    - Looks like inverse OLE is substantial other regularization?

    Other high level notes
    - [x] Pitt uses NEURAL signal AND KIN signal zscore (does nothing useful)
    - [x] Final scores computed on train sets
        - When switching to val set, reported score is 0.2-0.3 (vs 0.4-0.5).
"""


print(pred_y.shape)
print(test_y.shape)
from sklearn.metrics import r2_score

decoder = Ridge(alpha=200.0)
# decoder = Ridge(alpha=20000.0)
decoder.fit(train_x, train_y)

# compute r2
# print(test_x.shape, test_y.shape)
is_nan_y = np.isnan(test_y).any(axis=1)
test_x = test_x[~is_nan_y]
test_y = test_y[~is_nan_y]

# pred_y = predictor.predict(test_x)
# train_pred_y = predictor.predict(train_x)

pred_y = decoder.predict(test_x)
train_pred_y = decoder.predict(train_x)
r2 = r2_score(test_y, pred_y, multioutput='raw_values')
train_r2 = r2_score(train_y, train_pred_y, multioutput='raw_values')
print(f"Test : {r2}")
print(f"Train: {train_r2}")

palette = sns.color_palette(n_colors=kin.shape[1])
f, axes = plt.subplots(kin.shape[1], figsize=(12, 10))
# Plot true vs predictions
for idx, (true, pred) in enumerate(zip(test_y.T, pred_y.T)):
    axes[idx].plot(true, label=f'{idx}', color=palette[idx])
    axes[idx].plot(pred, linestyle='--', color=palette[idx])
