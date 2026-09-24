"""
The function calculates the portfolio value after rebalancing,
accounting for transaction costs that are proportional to the dollar value traded for each asset.

V_{t+1} / V{t+} = 1 - cost
where cost is the drag on portfolio value

"""
def bisection_method(f, a, b, tol=1e-5, max_iter=100):
    """
    Bisection method for finding the root of a function.

    Parameters:
    f: function - The function for which to find the root.
    a: float - The lower bound of the interval.
    b: float - The upper bound of the interval.
    tol: float - The tolerance for convergence.
    max_iter: int - The maximum number of iterations.

    Returns:
    float - The approximate root of the function.
    """

    if f(a) * f(b) >= 0:
        raise ValueError("The function must have different signs at a and b.")

    for i in range(max_iter):
        c = (a + b) / 2  # Midpoint
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c  # Root found

        if f(c) * f(a) < 0:
            b = c  # Root is in the left subinterval
        else:
            a = c  # Root is in the right subinterval

    raise ValueError("Maximum number of iterations reached without convergence.")


class TransCost:
    def __init__(self, c: float):
        self.c = c / 10000. # cost of transaction cost in bps
        self.tickers = None

    def get_init_cost(self, new_weights: list, old_weights: list) -> float:
        """
        :param new_weights: A dict of asset weights representing the asset allocation after rebalancing
        :param old_weights: A dict of asset weights representing the asset allocation before rebalancing
        :return: V_{t+1} / V_{t+}
        """
        # Calculate e_i for each asset, the sign of the weight change
        e = [(-1 if n_w > o_w else +1) for o_w, n_w in zip(old_weights, new_weights)]

        return (
            (1 - self.c * sum(w * ei for w, ei in zip(old_weights, e))) /
            (1 - self.c * sum(w * ei for w, ei in zip(new_weights, e)))
        )

    def cost_func(self, new_weights: list, old_weights: list, cost: float):
        e = [(-1 if n_w > o_w else +1) for o_w, n_w in zip(old_weights, new_weights)]
        return 1 - self.c * sum((old_weights[i] - new_weights[i] * cost) * e[i] for i in range(len(new_weights))) - cost

    def get_cost(self, new_weights: dict, old_weights: dict):
        """
        :param new_weights: A dict of asset weights representing the asset allocation after rebalancing
        :param old_weights: A dict of asset weights representing the asset allocation before rebalancing
        :return: 1 - V_{t+1} / V_{t+}
        """
        self.tickers = sorted(set(new_weights.keys()).union(old_weights.keys()))
        # create numpy array of weight for
        np_new_weights = [new_weights.get(ticker, 0) for ticker in self.tickers]
        np_old_weights = [old_weights.get(ticker, 0) for ticker in self.tickers]

        # normalize the weights to that they sum up to 1 or weight equal 0
        if sum(np_new_weights) != 0:
            np_new_weights = [wgt / sum(np_new_weights) for wgt in np_new_weights]
        if sum(np_old_weights) != 0:
            np_old_weights = [wgt / sum(np_old_weights) for wgt in np_old_weights]

        init_cost = self.get_init_cost(np_new_weights, np_old_weights)
        check = self.cost_func(np_new_weights, np_old_weights, cost=init_cost)
        if abs(check) < 1e-10:
            return 1 - init_cost  # if approximation is good enough
        else: # else return bisection result
            return 1 - bisection_method(lambda c: self.cost_func(np_new_weights, np_old_weights, c), 0, 1)


if __name__ == "__main__":
    tc = TransCost(c=10)
    new_weights = {"SPY": 0.4, "QQQ": 0.3, "LBUSTRUU": 0.3}
    old_weights = {"SPY": 0, "QQQ": 0, "LBUSTRUU": 0}

    cost = tc.get_cost(new_weights=new_weights, old_weights=old_weights)
    print(cost)
