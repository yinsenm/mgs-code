"""
Name     : lib_cov_func.py
Author   : William Smyth/Philip Ernst/Yinsen Miao
Contact : drwss.academy@gmail.com/yinsenm@gmail.com
Time     : 01/09/2025
Desc     : covariance/co-movement stastistic functions
"""


import pandas as pd
import numpy as np
from numpy import linalg as LA

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def is_psd_def(matrix, tol: float = 1e-10):
    """
    :param matrix: covariance matrix of p x p
    :return: true if positive semi definite (PSD)
    """
    values = np.asarray(matrix, dtype=float)
    values = (values + values.T) / 2.0
    return np.all(np.linalg.eigvalsh(values) >= -tol)


def correlation_from_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    """
    :param covariance: covariance matrix as input
    :return: correlation matrix
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    return correlation


def ledoit(df_rets: pd.DataFrame) -> pd.DataFrame:
    """
    compute Ledoit covariance Statistics
    :param df_rets: assets return matrix of dimension n x p
    :return: Ledoit covariance matrix of p x p
    """
    symbols = df_rets.columns
    x = df_rets.values.copy()
    t, n = x.shape
    _mean = np.tile(x.mean(axis=0), (t, 1))

    # de-mean the returns
    x -= _mean

    # compute sample covariance matrix
    sample = (1 / t) * x.transpose() @ x

    # compute the prior
    _var = np.diag(sample)
    sqrt_var = np.sqrt(_var).reshape((-1, n))
    rBar = (np.sum(sample / (np.tile(sqrt_var, (n, 1)).T * np.tile(sqrt_var, (n, 1)))) - n) / (n * (n - 1))
    prior = rBar * (np.tile(sqrt_var, (n, 1)).T * np.tile(sqrt_var, (n, 1)))
    prior[np.diag_indices_from(prior)] = _var.tolist()

    # compute shrinkage parameters and constant
    # what we call pi-hat
    y = x ** 2
    phiMat = y.T @ y / t - 2 * (x.T @ x) * sample / t + sample ** 2
    phi = np.sum(phiMat)

    # what we call rho-hat
    term1 = (x ** 3).T @ x / t
    help = (x.T @ x) / t
    helpDiag = np.diag(help).reshape((n, 1))
    term2 = np.tile(helpDiag, (1, n)) * sample
    term3 = help * np.tile(_var.reshape(n, 1), (1, n))
    term4 = np.tile(_var.reshape(n, 1), (1, n)) * sample
    thetaMat = term1 - term2 - term3 + term4
    thetaMat[np.diag_indices_from(thetaMat)] = np.zeros(n)
    rho = np.sum(np.diag(phiMat)) + rBar * np.sum(((1 / sqrt_var.T).dot(sqrt_var)) * thetaMat)

    # what we call gamma-hat
    gamma = LA.norm(sample - prior, 'fro') ** 2

    # compute shrinkage costant
    kappa = (phi - rho) / gamma
    shrinkage = max(0, min(1, kappa / t))

    # compute the estimator
    covMat = shrinkage * prior + (1 - shrinkage) * sample
    return pd.DataFrame(covMat, index=symbols, columns=symbols)

def gerber_cov_stat1(df_rets: pd.DataFrame, threshold: float = 0.5, center_method: str="zero") -> pd.DataFrame:
    """
    compute Gerber covariance Statistics 1
    :param df_rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    symbols = df_rets.columns
    rets = df_rets.values
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    cov_mat = np.zeros((p, p))  # store covariance matrix
    cor_mat = np.zeros((p, p))  # store correlation matrix

    assert 1 > threshold > 0, "threshold shall between 0 and 1"

    if center_method == "mean":
        rets = rets - np.mean(rets, axis=0)
    elif center_method == "median":
        rets = rets - np.median(rets, axis=0)

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                elif abs(rets[k, i]) < threshold * sd_vec[i] and abs(rets[k, j]) < threshold * sd_vec[j]:
                    nn += 1

            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (n - nn)
            cor_mat[j, i] = cor_mat[i, j]
            cov_mat[i, j] = cor_mat[i, j] * sd_vec[i] * sd_vec[j]
            cov_mat[j, i] = cov_mat[i, j]
    return pd.DataFrame(cov_mat, index=symbols, columns=symbols)


def _coerce_signal_frames(
        signal_input: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...],
) -> tuple[list[pd.DataFrame], bool]:
    if isinstance(signal_input, pd.DataFrame):
        return [signal_input], False
    if isinstance(signal_input, (list, tuple)) and signal_input:
        if not all(isinstance(frame, pd.DataFrame) for frame in signal_input):
            raise TypeError("All signal inputs must be pandas DataFrames.")
        base = signal_input[0]
        for idx, frame in enumerate(signal_input[1:], start=1):
            if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
                raise ValueError(
                    f"Signal DataFrame at position {idx} must have the same index and columns as the first signal."
                )
        return list(signal_input), True
    raise TypeError("df_rets must be a pandas DataFrame or a non-empty sequence of DataFrames.")


def _normalize_signal_weights(n_signals: int, signal_weights) -> np.ndarray:
    if signal_weights is None:
        return np.full(n_signals, 1.0 / n_signals, dtype=float)

    alpha = np.asarray(signal_weights, dtype=float)
    if alpha.ndim != 1 or alpha.size != n_signals:
        raise ValueError("signal_weights must be one-dimensional and match the number of signal families.")
    if np.any(~np.isfinite(alpha)) or np.any(alpha < 0):
        raise ValueError("signal_weights must be finite and non-negative.")

    alpha_sum = alpha.sum()
    if alpha_sum <= 0:
        raise ValueError("signal_weights must sum to a positive value.")

    return alpha / alpha_sum


def _coerce_volatility_vector(volatility_source, template: pd.DataFrame) -> np.ndarray:
    if volatility_source is None:
        return template.to_numpy(dtype=float).std(axis=0, ddof=0)
    if isinstance(volatility_source, pd.DataFrame):
        if not volatility_source.columns.equals(template.columns):
            raise ValueError("volatility_source must have the same asset columns as the signal input.")
        return volatility_source.to_numpy(dtype=float).std(axis=0, ddof=0)
    if isinstance(volatility_source, pd.Series):
        if not volatility_source.index.equals(template.columns):
            raise ValueError("volatility_source series must be indexed by the signal asset columns.")
        return volatility_source.to_numpy(dtype=float)

    vol_array = np.asarray(volatility_source, dtype=float)
    if vol_array.ndim == 1:
        if vol_array.shape[0] != len(template.columns):
            raise ValueError("volatility_source vector length must match the number of asset columns.")
        return vol_array
    if vol_array.ndim == 2:
        if vol_array.shape[1] != len(template.columns):
            raise ValueError("volatility_source matrix column count must match the number of asset columns.")
        return vol_array.std(axis=0, ddof=0)
    raise ValueError("volatility_source must be a 1D volatility vector or a 2D raw-return matrix.")


def _rank_center_signal_frames(signal_frames: list[pd.DataFrame]) -> list[np.ndarray]:
    return [(frame.rank(pct=True) - 0.5).to_numpy(dtype=float) for frame in signal_frames]


def _time_decay_weights(n_periods: int, half_life: float | None) -> np.ndarray:
    if half_life is None:
        return np.ones(n_periods, dtype=float)

    ages = (n_periods - 1) - np.arange(n_periods)
    return np.power(2.0, -ages / half_life)


def modified_gerber_stat(
        df_rets: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...],
        gamma: float = 1.0,
        n: float = 2.0,
        get_corr: bool = False,
        half_life: float | None = None,
        signal_weights=None,
        volatility_source=None,
) -> pd.DataFrame:
    """
    Article-faithful Modified Gerber Statistic.

    Each signal family is transformed into centered empirical percentiles, the
    paper's continuous MGS weight is applied, and the resulting correlation-like
    matrix is given a unit diagonal and scaled to covariance with population
    volatilities (``ddof=0``) from raw returns.

    Parameters
    ----------
    df_rets
        Either one signal DataFrame or a list/tuple of aligned signal DataFrames.
        Rows are time, columns are assets. For the paper's return-only case, pass
        a single raw-return DataFrame. For the multi-signal extension, pass
        multiple aligned signal families with the same index and columns.
    gamma
        Positive curvature parameter in the paper's weighting function.
    n
        Positive penalty parameter controlling how quickly the weight decreases
        when the magnitudes of two centered percentile scores diverge.
    get_corr
        If True, return the correlation-like MGS matrix. If False, scale that
        matrix into covariance using population volatilities (``ddof=0``).
    half_life
        Optional positive half-life in periods. If provided, observation weights
        are multiplied by ``2^{-age / half_life}``.
    signal_weights
        Optional non-negative weights for multiple signal families. If omitted,
        all signal families are weighted equally.
    volatility_source
        Optional source used only for covariance scaling when ``get_corr=False``.
        This can be a raw-return DataFrame, aligned volatility DataFrame, Series
        of per-asset volatilities, or a NumPy-compatible array. If omitted, the
        first input signal frame is used.

    Returns
    -------
    pd.DataFrame
        A square DataFrame indexed and columned by asset names containing either
        the MGS correlation-like matrix or the corresponding covariance matrix.

    Examples
    --------
    Single signal:
    >>> df_cov = modified_gerber_stat(df_rets, gamma=1.0, n=3.0)
    >>> df_corr = modified_gerber_stat(df_rets, gamma=1.0, n=3.0, get_corr=True)

    Multiple aligned signal families:
    >>> signal_1 = df_rets
    >>> signal_2 = df_rets.rolling(3, min_periods=1).mean()
    >>> signal_3 = df_rets.rolling(6, min_periods=1).std().fillna(0.0)
    >>> df_multi = modified_gerber_stat(
    ...     [signal_1, signal_2, signal_3],
    ...     gamma=0.8,
    ...     n=2.0,
    ...     half_life=6.0,
    ...     signal_weights=[0.5, 0.3, 0.2],
    ...     volatility_source=df_rets,
    ... )
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if half_life is not None and half_life <= 0:
        raise ValueError("half_life must be positive.")

    signal_frames, _ = _coerce_signal_frames(df_rets)
    ranked_signals = _rank_center_signal_frames(signal_frames)
    alpha = _normalize_signal_weights(len(signal_frames), signal_weights)

    symbols = signal_frames[0].columns
    n_assets = len(symbols)
    cor_mat = np.zeros((n_assets, n_assets), dtype=float)
    time_weights = _time_decay_weights(len(signal_frames[0]), half_life)

    for weight, signal_mat in zip(alpha, ranked_signals):
        signal_corr = np.zeros((n_assets, n_assets), dtype=float)

        for i in range(n_assets):
            x = signal_mat[:, i]
            abs_x = np.abs(x)
            # The weighted ratio defines only distinct-pair co-movement.
            # Self-association is imposed as one below.
            for j in range(i):
                y = signal_mat[:, j]
                base_weights = np.power(
                    np.sqrt((1.0 + abs_x) * (1.0 + np.abs(y))) /
                    (1.0 + np.power(np.abs(abs_x - np.abs(y)), n)),
                    gamma,
                )
                weighted = base_weights * time_weights
                denominator = weighted.sum()
                corr_ij = 0.0 if denominator == 0 else (np.sign(x * y) * weighted).sum() / denominator
                signal_corr[i, j] = corr_ij
                signal_corr[j, i] = corr_ij

        np.fill_diagonal(signal_corr, 1.0)
        cor_mat += weight * signal_corr

    np.fill_diagonal(cor_mat, 1.0)
    df_corr = pd.DataFrame(cor_mat, index=symbols, columns=symbols)
    if get_corr:
        return df_corr

    stds = _coerce_volatility_vector(volatility_source, signal_frames[0])
    cov_mat = cor_mat * np.outer(stds, stds)
    np.fill_diagonal(cov_mat, stds * stds)
    return pd.DataFrame(cov_mat, index=symbols, columns=symbols)
