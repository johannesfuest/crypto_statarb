import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def analyze_pairs_spreads_var(spreads_list, interval, var_lags=None, apply_shrinkage=True):
    """
    Joint modeling using Vector Autoregression (VAR) for all spreads together.
    
    Parameters:
    -----------
    spreads_list : list of pd.Series
        List of spread series (can contain negative values)
    interval : int
        Forecast horizon (periods ahead)
    var_lags : int or None
        Number of lags for VAR (None = automatic selection via AIC)
    apply_shrinkage : bool
        Whether to apply Ledoit-Wolf shrinkage to forecast covariance
    
    Returns:
    --------
    tuple : (mean_changes, cov_matrix, var_model)
        - mean_changes: Expected spread changes at horizon
        - cov_matrix: Covariance matrix of spread changes (possibly shrunk)
        - var_model: Fitted VAR model for diagnostics
    """
    # Combine spreads into DataFrame
    spreads_df = pd.DataFrame({
        f'spread_{i}': spread 
        for i, spread in enumerate(spreads_list)
    }).dropna()
    
    # Fit VAR model
    var_model = VAR(spreads_df)
    
    # Select lag order if not specified
    if var_lags is None:
        # Use information criteria to select lags
        lag_selection = var_model.select_order(maxlags=min(10, len(spreads_df) // 10))
        var_lags = lag_selection.aic
    
    # Fit with selected lags
    var_fit = var_model.fit(var_lags)
    
    # Generate forecast
    forecast = var_fit.forecast(spreads_df.values[-var_lags:], steps=interval)
    
    # Current values
    current_values = spreads_df.iloc[-1].values
    
    # Expected changes (from current to forecast horizon)
    predictions = forecast[-1, :]  # Last step of forecast
    mean_changes = predictions - current_values
    
    # Get forecast error covariance matrices for each step
    forecast_cov_steps = var_fit.forecast_cov(steps=interval)
    cov_matrix = np.sum(forecast_cov_steps, axis=0)
    
    # Apply shrinkage if requested
    if apply_shrinkage:
        # Use residuals for shrinkage target estimation
        residuals = var_fit.resid
        
        # Apply Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        # Fit on residuals to get shrinkage parameter
        _, shrinkage_constant = lw.fit(residuals).covariance_, lw.shrinkage_
        
        # Shrink toward diagonal (conservative for optimization)
        diagonal_target = np.diag(np.diag(cov_matrix))
        cov_matrix = (1 - shrinkage_constant) * cov_matrix + shrinkage_constant * diagonal_target
    
    return mean_changes, cov_matrix, var_fit


def maximize_sharpe_ratio(mean_changes, cov_matrix, risk_free_rate=0.0):
    """
    Portfolio optimization to maximize Sharpe ratio.
    
    Parameters:
    -----------
    mean_changes : np.array
        Expected spread changes from VAR
    cov_matrix : np.array
        Covariance matrix of spread changes
    risk_free_rate : float
        Risk-free rate for the period (default 0 for spread trading)
    
    Returns:
    --------
    dict : Optimization results containing:
        - weights: Optimal portfolio weights
        - sharpe: Maximum Sharpe ratio achieved
        - expected_change: Portfolio expected change
        - volatility: Portfolio volatility
    """
    n_assets = len(mean_changes)
    
    # Ensure positive definite covariance matrix
    # Add small regularization if needed
    min_eigenvalue = np.min(np.linalg.eigvals(cov_matrix))
    if min_eigenvalue < 1e-8:
        cov_matrix = cov_matrix + (1e-8 - min_eigenvalue) * np.eye(n_assets)
    
    # Objective function: negative Sharpe ratio (for minimization)
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, mean_changes) - risk_free_rate
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Handle edge cases
        if portfolio_std < 1e-8:
            return 1e8  # Large penalty for zero volatility
        
        return -portfolio_return / portfolio_std
    
    # Constraints: weights sum to 1 (fully invested)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Bounds: allow long and short positions
    # You might want to adjust these based on your constraints
    bounds = [(-2, 2) for _ in range(n_assets)]
    
    # Initial guess: equal weights
    w0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        negative_sharpe,
        w0,
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")
    
    # Extract results
    optimal_weights = result.x
    portfolio_return = np.dot(optimal_weights, mean_changes)
    portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
    portfolio_std = np.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    
    return {
        'weights': optimal_weights,
        'sharpe': sharpe_ratio,
        'expected_change': portfolio_return,
        'volatility': portfolio_std,
        'success': result.success
    }