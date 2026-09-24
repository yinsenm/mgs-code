import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
from cvxpy import psd_wrap

class PortfolioStrategyCVX:
    def __init__(
            self,
            strategy: str,
            min_weight: float = None,
            max_weight: float = None,
            txn_cost: float = None,
            use_cash: bool=False,
            solver: str = "GUROBI",
            verbose: bool=False
    ):
        """
        :param strategy: name of the strategy
        :param min_weight: constraint on minimum weight on asset
        :param max_weight: constraint on maximum weight on asset
        :param txn_cost: cost of transaction fee and slippage in bps or 0.01%
        :param use_cash: use cash or not, if set to true, then can be not fully invested.

        """
        assert strategy in ["ew", "mvo_tgt_vol", "mvo_tgt_vols", "mvo_tgt_ret", "mvo_tgt_rets", "mvp", "mr", "mdp", "ivw", "rp", "benchmark"], \
            "strategy must be one of ew, mvo_tgt_vol, mvo_tgt_vols, mvo_tgt_ret, mvp, mr, mdp, ivw, rp and benchmark strategies."
        assert min_weight is not None and  0 <= min_weight < 1, "min_weight in [0, 1)"
        assert max_weight is not None and  0 < max_weight <= 1, "max_weight in (0, 1]"
        assert min_weight is not None and max_weight is not None and min_weight < max_weight, "min_weight < max_weight"
        assert txn_cost >= 0, "transaction penalty must be larger or equal to 0"
        self.strategy = strategy
        self.min_weight = min_weight
        self.max_weight = max_weight
        if txn_cost is not None:
            self.txn_cost = txn_cost / 10000
        else:
            self.txn_cost = 0

        self.prev_weights = None  # previous weights of portfolio
        self.use_cash = use_cash
        self.solver = solver
        self.symbols = None
        self.n_asset = 0
        self.verbose = verbose

    def _ensure_solution(self, problem, weights, strategy_label: str):
        if weights.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError(
                f"Solver {self.solver} failed to find a solution for {strategy_label}. "
                f"Status: {problem.status}"
            )

    @staticmethod
    def _portfolio_variance(weights: np.ndarray, covariance_matrix: pd.DataFrame) -> float:
        cov_values = np.asarray(covariance_matrix.values, dtype=float)
        return float(weights.T @ cov_values @ weights)

    def _ensure_target_volatility(
            self,
            weights: np.ndarray,
            covariance_matrix: pd.DataFrame,
            target_volatility: float,
            strategy_label: str,
    ) -> None:
        realized_variance = self._portfolio_variance(weights, covariance_matrix)
        target_variance = float(target_volatility) ** 2.0
        # Conic solver tolerances vary slightly by host. A 2e-6 absolute
        # variance allowance retains the paper's borderline monthly solves
        # (for example, 0.00040104 against 0.00040000 in December 2012)
        # without accepting a materially over-target portfolio.
        tolerance = max(2e-6, 1e-3 * target_variance)
        if realized_variance > target_variance + tolerance:
            raise ValueError(
                f"Solver {self.solver} returned an infeasible {strategy_label} solution: "
                f"variance {realized_variance:.8f} exceeds target {target_variance:.8f}."
            )


    def get_weights(
            self,
            expected_returns: pd.Series=None,
            covariance_matrix: pd.DataFrame=None,
            target_returns: [float, list]=None,
            target_volatilities: [float, list] = None,
            index_name: str=None
    ) -> dict:

        if covariance_matrix is not None:
            assert covariance_matrix.shape[0] == covariance_matrix.shape[1], f"cov dim must be square"
            assert np.allclose(covariance_matrix, covariance_matrix.T), "covariance_matrix must be symmetric"
            self.symbols = covariance_matrix.columns

        if expected_returns is not None:
            self.symbols = expected_returns.index

        if covariance_matrix is not None and expected_returns is not None:
            assert (expected_returns.index == covariance_matrix.columns).all().item(), f"ticker list {self.symbols} must match for expected_returns and covariance_matrix"

        if target_returns is not None and isinstance(target_returns, list):
            for target_return in target_returns:
                assert target_return > 0, "target return must larger than 0"
        elif target_returns is not None and isinstance(target_returns, float):
            assert target_returns > 0, "target return must larger than 0"

        if target_volatilities is not None and isinstance(target_volatilities, list):
            for target_volatility in target_volatilities:
                assert target_volatility > 0, "target volatility must larger than 0"
        elif target_volatilities is not None and isinstance(target_volatilities, float):
            assert target_volatilities > 0, "target volatility must larger than 0"

        if index_name is not None:
            assert index_name in self.symbols, "index name must be in the symbol"

        self.n_asset = len(self.symbols)

        if self.strategy == "ew":
            return self._ew_func()
        elif self.strategy == "ivw":
            return self._ivw_func(covariance_matrix)
        elif self.strategy == "mvo_tgt_ret":  # mean variance optimization
            return self._mvo_tgt_ret_func(expected_returns, covariance_matrix, target_returns)
        elif self.strategy == "mvo_tgt_rets":  # mean variance optimization
            return self._mvo_tgt_rets_func(expected_returns, covariance_matrix, target_returns)
        elif self.strategy == "mvo_tgt_vol":  # mean variance optimization
            return self._mvo_tgt_vol_func(expected_returns, covariance_matrix, target_volatilities)
        elif self.strategy == "mvo_tgt_vols":  # mean variance optimization
            return self._mvo_tgt_vols_func(expected_returns, covariance_matrix, target_volatilities)
        elif self.strategy == "mr":   # maximize return
            return self._mr_func(expected_returns)
        elif self.strategy == "mvp":  # minimize variance
            return self._mvp_func(covariance_matrix)
        elif self.strategy == "mdp":
            return self._mdp_func(covariance_matrix)
        elif self.strategy == "rp":
            return self._rp_func(covariance_matrix)
        elif self.strategy == "benchmark":
            return self._benchmark_func(index_name)

    @staticmethod
    def get_port_vol(wgts: np.ndarray, cov_mat: np.ndarray) -> float:
        return np.sqrt(np.dot(wgts.T, np.dot(cov_mat, wgts)))

    @staticmethod
    def get_port_mu(wgts: np.ndarray, mu_vec: np.ndarray) -> float:
        return np.dot(wgts, mu_vec)

    @staticmethod
    def get_port_div_ratio(wgts: np.ndarray, cov_mat: np.ndarray) -> float:
        # weighted average volatility
        w_vol = np.dot(np.sqrt(np.diag(cov_mat)), wgts.T)
        # portfolio volatility
        p_vol = np.sqrt(np.dot(wgts.T, np.dot(cov_mat, wgts)))
        return w_vol / (p_vol + 1e-8)

    def get_port_txn_cost(self, wgts: np.ndarray, prev_wgts: np.ndarray):
        return self.txn_cost * np.abs(wgts - prev_wgts).sum()

    def get_marginal_risk_contribution(self, wgts: np.ndarray, cov_mat: np.ndarray) -> np.ndarray:
        # Function to calculate asset contribution to total risk
        sigma = self.get_port_vol(wgts, cov_mat)
        return np.multiply(np.dot(cov_mat, wgts), wgts.T) / (sigma ** 2)

    def get_rp_obj(self, wgts: np.ndarray, cov_mat: np.ndarray) -> float:
        sigma = self.get_port_vol(wgts, cov_mat)
        x = wgts / sigma
        return (np.dot(x.T, np.dot(cov_mat, x)) / 2.) - np.sum(np.log(x + 1e-10)) / cov_mat.shape[0]

    def _benchmark_func(self, index_name) -> dict:
        weights = {s: 0. for s in self.symbols}
        weights[index_name] = 1.   # allocate everything to the benchmark asset
        return weights

    def _ew_func(self) -> dict:
        return {s: 1 / self.n_asset for s in self.symbols}

    def _ivw_func(self, covariance_matrix: pd.DataFrame) -> dict:
        inv_vars = 1.0 / np.sqrt(covariance_matrix.values.diagonal())
        return {
            s: v for s, v in zip(self.symbols, inv_vars / inv_vars.sum())
        }

    def _mvp_func(self, covariance_matrix: pd.DataFrame) -> dict:
        w = cp.Variable(self.n_asset)
        # define constraints
        if self.use_cash:
            constraints = [cp.sum(w) <= 1]
        else:
            constraints = [cp.sum(w) == 1]

        if self.min_weight is not None:
            constraints += [w >= self.min_weight]

        if self.max_weight is not None:
            constraints += [w <= self.max_weight]

        # compute risk
        risk = cp.quad_form(w, psd_wrap(covariance_matrix.values))

        if self.prev_weights is not None:
            objective = cp.Minimize(risk + self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Minimize(risk)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver, verbose=self.verbose)

        self._ensure_solution(problem, w, "MVP")

        return {k: v for k, v in zip(self.symbols, w.value.tolist())}

    def _mr_func(self, expected_returns) -> dict:
        w = cp.Variable(self.n_asset)
        ret = expected_returns.values @ w
        # define constraints
        if self.use_cash:
            constraints = [cp.sum(w) <= 1]
        else:
            constraints = [cp.sum(w) == 1]

        if self.min_weight is not None:
            constraints += [w >= self.min_weight]

        if self.max_weight is not None:
            constraints += [w <= self.max_weight]

        if self.prev_weights is not None:
            objective = cp.Maximize(ret - self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Maximize(ret)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        self._ensure_solution(problem, w, "MR")

        return {k: v for k, v in zip(self.symbols, w.value.tolist())}

    def _mvo_tgt_vol_func(self, expected_returns, covariance_matrix, target_volatility) -> dict:
        w = cp.Variable(self.n_asset)
        ret = expected_returns.values @ w
        # define constraints
        if self.use_cash:
            constraints = [cp.sum(w) <= 1]
        else:
            constraints = [cp.sum(w) == 1]

        if self.min_weight is not None:
            constraints += [w >= self.min_weight]

        if self.max_weight is not None:
            constraints += [w <= self.max_weight]

        # compute risk
        risk = cp.quad_form(w, psd_wrap(covariance_matrix.values))
        constraints += [risk <= (target_volatility ** 2)]

        if self.prev_weights is not None:
            objective = cp.Maximize(ret - self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Maximize(ret)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        self._ensure_solution(problem, w, "MVO Target Vol")
        self._ensure_target_volatility(
            np.asarray(w.value, dtype=float),
            covariance_matrix,
            target_volatility,
            "MVO Target Vol",
        )

        return {k: v for k, v in zip(self.symbols, w.value.tolist())}

    def _mvo_tgt_ret_func(self, expected_returns, covariance_matrix, target_return) -> dict:
        w = cp.Variable(self.n_asset)
        ret = expected_returns.values @ w
        # define constraints
        if self.use_cash:
            constraints = [cp.sum(w) <= 1]
        else:
            constraints = [cp.sum(w) == 1]

        if self.min_weight is not None:
            constraints += [w >= self.min_weight]

        if self.max_weight is not None:
            constraints += [w <= self.max_weight]

        # compute risk
        risk = cp.quad_form(w, psd_wrap(covariance_matrix.values))
        constraints += [ret >= target_return]

        if self.prev_weights is not None:
            objective = cp.Minimize(risk + self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Minimize(risk)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver, verbose=self.verbose)

        self._ensure_solution(problem, w, "MVO Target Return")

        return {k: v for k, v in zip(self.symbols, w.value.tolist())}

    def _mvo_tgt_rets_func(self, expected_returns, covariance_matrix, target_returns: list) -> dict:
        w = cp.Variable(self.n_asset)
        tgt_ret = cp.Parameter(nonneg=True)
        ret = expected_returns.values @ w
        # define constraints
        if self.use_cash:
            constraints = [cp.sum(w) <= 1]
        else:
            constraints = [cp.sum(w) == 1]

        if self.min_weight is not None:
            constraints += [w >= self.min_weight]

        if self.max_weight is not None:
            constraints += [w <= self.max_weight]

        # compute risk
        risk = cp.quad_form(w, psd_wrap(covariance_matrix.values))
        constraints += [ret >= tgt_ret]

        if self.prev_weights is not None:
            objective = cp.Minimize(risk + self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Minimize(risk)

        problem = cp.Problem(objective, constraints)
        res_dict = {}
        for target_return in target_returns:
            tgt_ret.value = target_return
            problem.solve(solver=self.solver, verbose=self.verbose)
            if w.value is not None and problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                res_dict[target_return] = {k: v for k, v in zip(self.symbols, w.value.tolist())}
        return res_dict

    def _mvo_tgt_vols_func(self, expected_returns, covariance_matrix, target_volatilities: list) -> dict:
        w = cp.Variable(self.n_asset)
        tgt_var = cp.Parameter(nonneg=True)
        ret = expected_returns.values @ w
        # define constraints
        if self.use_cash:
            constraints = [cp.sum(w) <= 1]
        else:
            constraints = [cp.sum(w) == 1]

        if self.min_weight is not None:
            constraints += [w >= self.min_weight]

        if self.max_weight is not None:
            constraints += [w <= self.max_weight]

        # compute risk
        risk = cp.quad_form(w, psd_wrap(covariance_matrix.values))
        constraints += [risk <= tgt_var]

        if self.prev_weights is not None:
            objective = cp.Minimize(-ret + self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Minimize(-ret)

        problem = cp.Problem(objective, constraints)
        res_dict = {}
        for target_volatility in target_volatilities:
            tgt_var.value = target_volatility ** 2.0
            problem.solve(solver=self.solver, verbose=self.verbose)
            if w.value is not None and problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                weights = np.asarray(w.value, dtype=float)
                self._ensure_target_volatility(
                    weights,
                    covariance_matrix,
                    target_volatility,
                    "MVO Target Vols",
                )
                res_dict[target_volatility] = {k: v for k, v in zip(self.symbols, weights.tolist())}
        return res_dict

    def _mdp_func(self, covariance_matrix: pd.DataFrame) -> dict:

        if self.prev_weights is not None:
            cost = lambda wgts: -self.get_port_div_ratio(wgts, covariance_matrix.values) + self.get_port_txn_cost(wgts, self.prev_weights)
        else:
            cost = lambda wgts: -self.get_port_div_ratio(wgts, covariance_matrix.values)

        if self.use_cash:
            constraints = [
                {"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)}
            ]
        else:
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
            ]
        init_weights = np.array(self.n_asset * [1. / self.n_asset])
        opt = minimize(
            cost,
            x0=init_weights,
            bounds=tuple((self.min_weight, self.max_weight) for k in range(self.n_asset)),
            constraints=constraints, method='SLSQP'
        )
        if not opt.success:
            raise RuntimeError(f"MDP optimization failed: {opt.message}")
        return {k: v for k, v in zip(self.symbols, opt.x.tolist())}

    def _rp_func(self, covariance_matrix: pd.DataFrame) -> dict:
        w = cp.Variable(self.n_asset)

        # compute risk
        risk = cp.quad_form(w, psd_wrap(covariance_matrix.values))

        # define constraints
        constraints = [
            cp.sum(cp.log(w)) >= 1.0 / self.n_asset,
        ]

        if self.min_weight is not None:
            constraints += [w >= self.min_weight]

        if self.prev_weights is not None:
            objective = cp.Minimize(risk + self.txn_cost * cp.sum(cp.abs(w - self.prev_weights)))
        else:
            objective = cp.Minimize(risk)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        if w.value is None:
            raise ValueError(f"Solver {self.solver} failed to find a solution for Risk Parity.")

        w_val = w.value / np.sum(w.value)
        return {k: v for k, v in zip(self.symbols, w_val.tolist())}

    def mvo_bs_tgt_vol(
            self,
            covariance_matrix: pd.DataFrame,
            w_min: pd.Series,
            w_max: pd.Series,
            target_volatility: float,
            tol: float=1e-6, max_iter: int=100
    ):
        """
            - w_min: Portfolio weight corresponding to the minimum bound.
            - w_max: Portfolio weight corresponding to the maximum bound
            - tol: Tolerance for convergence.
            - max_iter: Maximum number of iterations.
            :return: optimal weight as convex combination of w_min and w_max
        """
        assert (w_min.index == w_max.index).all().item() and (w_min.index == covariance_matrix.columns).all().item(), \
            "asset symbol universe must equal"
        assert target_volatility > 0, "target_volatility must larger than 0"
        min_vol = self.get_port_vol(w_min.values, covariance_matrix.values)
        max_vol = self.get_port_vol(w_max.values, covariance_matrix.values)
        assert min_vol <= target_volatility <= max_vol, f"target_volatility must be within bound {min_vol} and {max_vol}"

        alpha_min, alpha_max = 0, 1
        for iteration in range(max_iter):
            alpha_mid = (alpha_min + alpha_max) / 2
            w_mid = (1 - alpha_mid) * w_min + alpha_mid * w_max
            vol_mid = self.get_port_vol(w_mid.values, covariance_matrix.values)
            # print(iteration, alpha_mid, vol_mid)
            # Check if the volatility is close to the target
            if abs(vol_mid - target_volatility) < tol:
                return w_mid

            if vol_mid > target_volatility:
                alpha_max = alpha_mid
            else:
                alpha_min = alpha_mid

    def mvo_bs_tgt_ret(
            self,
            expected_returns: pd.Series,
            w_min: pd.Series,
            w_max: pd.Series,
            target_return: float,
            tol: float=1e-6, max_iter: int=100
    ):
        """
            - w_min: Portfolio weight corresponding to the minimum bound.
            - w_max: Portfolio weight corresponding to the maximum bound
            - tol: Tolerance for convergence.
            - max_iter: Maximum number of iterations.
            :return: optimal weight as convex combination of w_min and w_max
        """
        assert (w_min.index == w_max.index).all().item() and (w_min.index == expected_returns.index).all().item(), \
            "asset symbol universe must equal"
        assert target_return > 0, "target_return must larger than 0"

        min_ret = self.get_port_mu(w_min.values, expected_returns.values)
        max_ret = self.get_port_mu(w_max.values, expected_returns.values)
        assert min_ret <= target_return <= max_ret, f"target_return must be within bound {min_ret} and {max_ret}"

        alpha_min, alpha_max = 0., 1.
        for iteration in range(max_iter):
            # Compute midpoint alpha
            alpha_mid = (alpha_min + alpha_max) / 2
            w_mid = (1 - alpha_mid) * w_min + alpha_mid * w_max
            ret_mid = self.get_port_mu(w_mid.values, expected_returns.values)
            # print(iteration, alpha_mid, ret_mid)
            if abs(ret_mid - target_return) < tol:
                return w_mid
            if ret_mid > target_return:
                alpha_max = alpha_mid
            else:
                alpha_min = alpha_mid
