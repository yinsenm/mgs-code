"""Covariance estimators compared in the 10-asset SSRN experiment."""

from .lib_cov_func import ledoit


def format_tag(value: float) -> str:
    return f"{float(value):.1f}".replace(".", "_")


def get_paper_methods(cov_impl: str, mgs_n_values: list[float], mgs_gamma: float = 1.0):
    if cov_impl == "cpp":
        from .lib_cov_func_cpp import gerber_cov_stat1, modified_gerber_stat
    elif cov_impl == "python":
        from .lib_cov_func import gerber_cov_stat1, modified_gerber_stat
    else:
        raise ValueError(f"Unknown covariance implementation: {cov_impl}")

    methods = {
        "HC": {"cov_func": lambda returns: returns.cov(), "cov_params": {}},
        "SM": {"cov_func": ledoit, "cov_params": {}},
        "GS1-ts=0_5": {"cov_func": gerber_cov_stat1,
                       "cov_params": {"threshold": 0.5, "center_method": "zero"}},
    }
    for n_value in sorted({float(value) for value in mgs_n_values}):
        name = f"modified_gerber_stat-g={format_tag(mgs_gamma)}-n={format_tag(n_value)}-h=inf"
        methods[name] = {
            "cov_func": modified_gerber_stat,
            "cov_params": {"gamma": float(mgs_gamma), "n": n_value},
        }
    return methods
