import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.stats.correlation_tools import cov_nearest
from functools import wraps
import sys

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def use_progress_bar(func):
    @wraps(func)
    def wrapper(*args, use_progress_bar=True, **kwargs):
        if use_progress_bar:
            with tqdm(total=100) as pbar:
                # Call the decorated function and update the progress bar
                result = func(*args, **kwargs)
                pbar.update(100)
        else:
            # Call the decorated function without using the progress bar
            result = func(*args, **kwargs)
        return result
    return wrapper

@use_progress_bar
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

@use_progress_bar
def mala_adapt(fp, fg, x0, h0, c0, alpha, epoch):
    """
    Sample from a target distribution using an adaptive version of the
    Metropolis-adjusted Langevin algorithm.

    Args:
    fp    - handle to the log-density function of the target.
    fg    - handle to the gradient function of the log target.
    x0    - vector of the starting values of the Markov chain.
    h0    - initial step-size parameter.
    c0    - initial preconditioning matrix.
    alpha - adaptive schedule.
    epoch - length of each tuning epoch.

    Returns:
    h     - tuned step-size.
    c     - tuned preconditioning matrix.
    x     - list of matrices of generated points.
    g     - list of matrices of gradients of the log target at X.
    p     - list of vectors of log-density values of the target at X.
    a     - list of binary vectors indicating whether a move is accepted.
    """

    n_ep = len(epoch)
    x = n_ep * [None]
    g = n_ep * [None]
    p = n_ep * [None]
    a = n_ep * [None]

    # First epoch
    h = h0
    c = c0
    x[0], g[0], p[0], a[0] = mala(fp, fg, x0, h, c, epoch[0])

    for i in range(1, n_ep):
        # Adapt preconditioning matrix
        c = alpha[i] * c + (1 - alpha[i]) * np.cov(x[i - 1].T)
        c = cov_nearest(c)

        # Tune step-size
        ar = np.mean(a[i - 1])
        h = h * np.exp(ar - 0.57)

        # Next epoch
        x0_new = x[i - 1][-1]
        x[i], g[i], p[i], a[i] = mala(fp, fg, x0_new, h, c, epoch[0])

    return (h, c, x, g, p, a)
