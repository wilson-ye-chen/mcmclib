import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.stats.correlation_tools import cov_nearest
from tqdm import tqdm

def mala(fp, fg, x0, h, c, n, pb=True):
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
    pb - a progress bar is shown if set to True (default).

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
    for i in tqdm(range(1, n), disable=(not pb)):
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

def mala_adapt(fp, fg, x0, h0, c0, alpha, epoch, pb=True):
    """
    Sample from a target distribution using an adaptive version of the
    Metropolis-adjusted Langevin algorithm.

    Args:
    fp    - handle to the log-density function of the target.
    fg    - handle to the gradient function of the log target.
    x0    - vector of the starting values of the Markov chain.
    h0    - initial step-size parameter.
    c0    - initial preconditioning matrix.
    alpha - vector of learning rates for preconditioning matrix.
    epoch - vector of tuning epoch lengths.
    pb    - a progress bar is shown if set to True (default).

    Returns:
    x     - list of matrices of generated points.
    g     - list of matrices of gradients of the log target at x.
    p     - list of vectors of log-density values of the target at x.
    a     - list of binary vectors indicating whether a move is accepted.
    h     - tuned step-size.
    c     - tuned preconditioning matrix.
    """

    n_ep = len(epoch)
    x = n_ep * [None]
    g = n_ep * [None]
    p = n_ep * [None]
    a = n_ep * [None]

    # First epoch
    h = h0
    c = c0
    x[0], g[0], p[0], a[0] = mala(fp, fg, x0, h, c, epoch[0], False)

    for i in tqdm(range(1, n_ep), disable=(not pb)):
        # Adapt preconditioning matrix
        c = alpha[i - 1] * c + (1 - alpha[i - 1]) * np.cov(x[i - 1].T)
        c = cov_nearest(c)

        # Tune step-size
        ar = np.mean(a[i - 1])
        h = h * np.exp(ar - 0.57)

        # Next epoch
        x0_new = x[i - 1][-1]
        x[i], g[i], p[i], a[i] = mala(fp, fg, x0_new, h, c, epoch[i], False)

    return (x, g, p, a, h, c)

def rwm(fp, x0, h, c, n, pb=True):
    """
    Sample from a target distribution using the random-walk Metropolis
    algorithm.

    Args:
    fp - handle to the log-density function of the target.
    x0 - vector of the starting values of the Markov chain.
    h  - step-size parameter.
    c  - preconditioning matrix.
    n  - number of MCMC iterations.
    pb - a progress bar is shown if set to True (default).

    Returns:
    x  - matrix of generated points.
    p  - vector of log-density values of the target at X.
    a  - binary vector indicating whether a move is accepted.
    """

    # Initialise the chain
    d = len(x0)
    x = np.empty((n, d))
    p = np.empty(n)
    a = np.zeros(n, dtype=bool)
    x[0] = x0
    p[0] = fp(x0)

    # For each MCMC iteration
    for i in tqdm(range(1, n), disable=(not pb)):
        # Random walk proposal
        s = h ** 2 * c
        y = np.random.multivariate_normal(x[i - 1], s)

        # Log acceptance probability
        py = fp(y)
        acc_pr = py - p[i - 1]

        # Accept with probability acc_pr
        if acc_pr >= 0 or np.log(np.random.uniform()) < acc_pr:
            x[i] = y
            p[i] = py
            a[i] = True
        else:
            x[i] = x[i - 1]
            p[i] = p[i - 1]

    return (x, p, a)

def rwm_adapt(fp, x0, h0, c0, alpha, epoch, pb=True):
    """
    Sample from a target distribution using an adaptive version of the
    random-walk Metropolis algorithm.

    Args:
    fp    - handle to the log-density function of the target.
    x0    - vector of the starting values of the Markov chain.
    h0    - initial step-size parameter.
    c0    - initial preconditioning matrix.
    alpha - vector of learning rates for preconditioning matrix.
    epoch - vector of tuning epoch lengths.
    pb    - a progress bar is shown if set to True (default).

    Returns:
    x     - list of matrices of generated points.
    p     - list of vectors of log-density values of the target at x.
    a     - list of binary vectors indicating whether a move is accepted.
    h     - tuned step-size.
    c     - tuned preconditioning matrix.
    """

    n_ep = len(epoch)
    x = n_ep * [None]
    p = n_ep * [None]
    a = n_ep * [None]

    # First epoch
    h = h0
    c = c0
    x[0], p[0], a[0] = rwm(fp, x0, h, c, epoch[0], False)

    for i in tqdm(range(1, n_ep), disable=(not pb)):
        # Adapt preconditioning matrix
        c = alpha[i - 1] * c + (1 - alpha[i - 1]) * np.cov(x[i - 1].T)
        c = cov_nearest(c)

        # Tune step-size
        ar = np.mean(a[i - 1])
        h = h * np.exp(ar - 0.234)

        # Next epoch
        x0_new = x[i - 1][-1]
        x[i], p[i], a[i] = rwm(fp, x0_new, h, c, epoch[i], False)

    return (x, p, a, h, c)
