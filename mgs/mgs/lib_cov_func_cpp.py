import numpy as np
import pandas as pd

try:
    from .cgerber import c_gerber_cov_stat1, c_modified_gerber_stat
except ImportError as exc:
    raise ImportError(
        "Could not import the compiled `cgerber` extension. "
        "Build it first with `./cpp/run_compile.sh` from the `mgs` folder."
    ) from exc


def _coerce_signal_frames(signal_input):
    if isinstance(signal_input, pd.DataFrame):
        return [signal_input], False
    if isinstance(signal_input, (list, tuple)):
        if not signal_input:
            raise ValueError("signal_input must contain at least one DataFrame.")
        if not all(isinstance(frame, pd.DataFrame) for frame in signal_input):
            raise TypeError("All signal inputs must be pandas DataFrames.")
        base = signal_input[0]
        for idx, frame in enumerate(signal_input[1:], start=1):
            if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
                raise ValueError(
                    f"Signal DataFrame at position {idx} must have the same index and columns as the first signal."
                )
        return list(signal_input), True
    raise TypeError("signal_input must be a pandas DataFrame or a non-empty sequence of DataFrames.")


def _coerce_signal_weights(signal_weights):
    if signal_weights is None:
        return None
    weights = np.asarray(signal_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("signal_weights must be one-dimensional.")
    return weights


def _coerce_volatility_vector(volatility_source, template: pd.DataFrame):
    if volatility_source is None:
        return template.to_numpy(dtype=np.float64).std(axis=0, ddof=0)
    if isinstance(volatility_source, pd.DataFrame):
        if not volatility_source.columns.equals(template.columns):
            raise ValueError("volatility_source must have the same asset columns as the signal input.")
        return volatility_source.to_numpy(dtype=np.float64).std(axis=0, ddof=0)
    if isinstance(volatility_source, pd.Series):
        if not volatility_source.index.equals(template.columns):
            raise ValueError("volatility_source series must be indexed by the signal asset columns.")
        return volatility_source.to_numpy(dtype=np.float64)

    vol_array = np.asarray(volatility_source, dtype=np.float64)
    if vol_array.ndim == 1:
        if vol_array.shape[0] != len(template.columns):
            raise ValueError("volatility_source vector length must match the number of asset columns.")
        return vol_array
    if vol_array.ndim == 2:
        if vol_array.shape[1] != len(template.columns):
            raise ValueError("volatility_source matrix column count must match the number of asset columns.")
        return vol_array.std(axis=0, ddof=0)
    raise ValueError("volatility_source must be a 1D volatility vector or a 2D raw-return matrix.")


def _rank_center_signal_frames(signal_frames):
    return [(frame.rank(pct=True) - 0.5).to_numpy(dtype=np.float64) for frame in signal_frames]


def gerber_cov_stat1(
    df_rets: pd.DataFrame,
    threshold: float = 0.5,
    center_method: str = "zero",
    get_corr: bool = False,
) -> pd.DataFrame:
    symbols = df_rets.columns
    res_mat = c_gerber_cov_stat1(
        df_rets.to_numpy(dtype=np.float64),
        threshold,
        center_method,
        get_corr,
    )
    return pd.DataFrame(res_mat, index=symbols, columns=symbols)


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
    Article-faithful Modified Gerber Statistic backed by the C++ kernel.

    The weighted ratio is evaluated for distinct asset pairs, self-correlation
    is fixed at one, and covariance scaling uses raw-return population
    volatilities (``ddof=0``).
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if half_life is not None and half_life <= 0:
        raise ValueError("half_life must be positive.")

    signal_frames, is_multi = _coerce_signal_frames(df_rets)
    symbols = signal_frames[0].columns
    signal_payload = _rank_center_signal_frames(signal_frames)
    signal_input = signal_payload if is_multi else signal_payload[0]
    weights = _coerce_signal_weights(signal_weights)
    vol_source = None if get_corr else _coerce_volatility_vector(
        signal_frames[0] if volatility_source is None else volatility_source,
        signal_frames[0],
    )
    res_mat = c_modified_gerber_stat(
        signal_input,
        gamma,
        n,
        get_corr,
        half_life,
        weights,
        vol_source,
    )
    return pd.DataFrame(res_mat, index=symbols, columns=symbols)
