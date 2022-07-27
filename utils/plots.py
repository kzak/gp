from pdb import set_trace

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D


def plot_gp1d(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * jnp.sqrt(jnp.abs(jnp.diag(cov)))
    # uncertainty = 1.96 * jnp.sqrt(jnp.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label="Mean")

    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls="--", label=f"Sample {i+1}")

    if X_train is not None:
        plt.plot(X_train, Y_train, "rx")

    # plt.legend()
