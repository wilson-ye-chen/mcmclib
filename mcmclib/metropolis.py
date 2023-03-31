import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

def mala(fp, fg, x0, h, c, n):
    """
    Sample from a target distribution using the Metropolis-adjusted Langevin
    algorithm.

    Args:
    fp - handle to the log-density function of the target.
    fg - handle to the gradient function of the log target.
    x0 - vector of the starting values of the Markov chain.
    h  - step-size parameter.
    c  - preconditioning matrix.
    n  - number of MCMC iterations.

    Returns:
    x  - matrix of generated points.
    g  - matrix of gradients of the log target at X.
    p  - vector of log-density values of the target at X.
    a  - binary vector indicating whether a move is accepted.
    """

    # Initialise the chain
    d = len(x0)
    x = np.empty((n, d))
    g = np.empty((n, d))
    p = np.empty(n)
    a = np.zeros(n, dtype=bool)
    x[0] = x0
    g[0] = fg(x0)
    p[0] = fp(x0)

    # For each MCMC iteration
    for i in tqdm(range(1, n)):
        # Langevin proposal
        hh = h ** 2
        mx = x[i - 1] + hh / 2 * np.dot(c, g[i - 1])
        s = hh * c
        y = np.random.multivariate_normal(mx, s)

        # Log acceptance probability
        py = fp(y)
        gy = fg(y)
        my = y + hh / 2 * np.dot(c, gy)
        qx = multivariate_normal.logpdf(x[i - 1], my, s)
        qy = multivariate_normal.logpdf(y, mx, s)
        acc_pr = (py + qx) - (p[i - 1] + qy)

        # Accept with probability acc_pr
        if acc_pr >= 0 or np.log(np.random.uniform()) < acc_pr:
            x[i] = y
            g[i] = gy
            p[i] = py
            a[i] = True
        else:
            x[i] = x[i - 1]
            g[i] = g[i - 1]
            p[i] = p[i - 1]

    return (x, g, p, a)
