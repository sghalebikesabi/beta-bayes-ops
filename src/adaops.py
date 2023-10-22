### ORIGINAL CODE HERE https://github.com/yuxiangw/optimal_dp_linear_regression/blob/master/code/adaops.m

import numpy as np
from scipy.linalg import svd, cholesky
from scipy.optimize import minimize


def adaops(X, y, opts):
    epsilon = opts["eps"]
    delta = opts["delta"]

    # bound of variables
    BY = 1
    BX = 1

    n, d = X.shape
    XTy = X.T.dot(y)
    XTX = X.T.dot(X) + np.eye(d)

    # eps/4 for releasing eigenvalue lamb_min
    # eps/4 for releasing local Lipschitz constant for the chosen lamb
    # eps/2 for doing OPS in the end.

    # lamb_min + lamb  >  (1+log(2/\delta))BX^2/ [2(eps/2)] otherwise  even
    # if \gamma = 0, we won't get the privacy level.
    # so we set a bar at 2 times that much.  (1+log(2/\delta))BX^2/ [2(eps/2)]
    # then when that's the case, it doesn't make sense to take gamma to be
    # very small you know as it won't matter... choose gamma such that

    # DP release the smallest eigenvalue
    S = svd(XTX)[1]
    logsod = np.log(6 / delta)

    lamb_min = (
        S[-1]
        + np.random.randn() * BX**2 * np.sqrt(logsod) / (epsilon / 4)
        - logsod / (epsilon / 4)
    )
    lamb_min = max(lamb_min, 0)

    # how much to allocate for the part that does not depend on \gamma
    alpha = 1 / 2
    lamb_target = BX**2 * (1 + np.log(6 / delta)) / 2 / (epsilon / 2 * alpha)
    lamb_target = max(lamb_target - lamb_min, 0)

    # solve the quadratic equation to get epsilonbar, when we take
    # gamma  =  (lamb+lamb_min) * epsilonbar^2 /L^2/logsod

    # solve the quadratic equation w.r.t. the epsilonbar.
    a = 1 / 2 * logsod
    b = 1
    c = -epsilon / 2 * (1 - alpha)
    epsilonbar = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    varrho = 0.05
    C1 = C1_fun(epsilonbar, delta / 3, varrho, d)
    C2 = C2_fun(epsilon / 4, delta / 3)

    def fun_obj(t):
        return (
            C1
            * BX**4
            * (1 + BX**2 / (t + lamb_min)) ** (2 * C2)
            / (t + lamb_min)
            + t
        )

    def grad_obj(t):
        return (
            1
            - (
                2
                * BX**6
                * C1
                * C2
                * (BX**2 / (lamb_min + t) + 1) ** (2 * C2 - 1)
            )
            / (lamb_min + t) ** 3
            - (BX**4 * C1 * (BX**2 / (lamb_min + t) + 1) ** (2 * C2))
            / (lamb_min + t) ** 2
        )

    options = {"disp": False}
    result = minimize(
        fun_obj,
        2 * C2,
        bounds=[(-1, -lamb_target)],
        jac=grad_obj,
        method="SLSQP",
        options=options,
    )
    lamb = result.x[0]

    # solve
    if lamb < 0:
        print("AdaOPS Optimization error!")

    H = XTX + lamb * np.eye(d)

    # now get an estimate of the magnitude of theta
    R = cholesky(H)  # do Cholesky decomposition H = R'*R
    theta = np.linalg.solve(R.T, np.linalg.solve(R, XTy))

    thetanorm = np.linalg.norm(theta)
    sigma = np.log(1 + BX**2 / (lamb + lamb_min)) / (epsilon / 4)
    Delta = (
        np.log(BY + BX * thetanorm)
        + np.random.randn() * sigma * np.sqrt(logsod)
        + sigma * logsod
    )

    # estimate \|hat(theta)\| and the Lipschitz constant.
    L = BX * min(
        np.exp(Delta),
        BY + BX * np.sqrt(2 * np.sqrt(n * BY) / (lamb + lamb_min)),
    )

    h = lamb + lamb_min

    # This ensures that the variance is at most as big as the intrinsic
    # variance.

    # solve the quadratic equation to get the actual epsilontilde
    a = L**2 / 2 / (h + BX**2)
    b = np.sqrt(L**2 * logsod / h)
    c = BX**2 * (1 + logsod) / 2 / h - epsilon / 2
    sqrtgamma = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # calibrate gamma
    # sqrtgamma = np.sqrt(lamb_min + lamb) * epsilontilde / np.sqrt(logsod) / L

    # output the OPS sample
    thetahat = theta + np.linalg.solve(R.T, np.random.randn(d) / sqrtgamma)

    return thetahat
