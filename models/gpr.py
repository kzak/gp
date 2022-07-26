import jax.numpy as jnp
from jax import grad, jit, vmap


def jitter(n, eps=1e-6):
    return eps * jnp.eye(n)


def rbf_kn(X1, X2, params):
    """
    Args:
        X1: (n, d) matrix
        X2: (m, d) matrix
        params: [tau, sigma], tau, sigma > 0
    Returns:
        (n, m) matrix
        tau * exp(- 1/(2 * sigma^2) | x_i - x_j |^2 )
    """
    tau, sigma = params

    d = jnp.sum(X1**2, 1).reshape(-1, 1) + jnp.sum(X2**2, 1) - 2 * jnp.dot(X1, X2.T)
    return tau * jnp.exp(-(0.5 * sigma**2) * d)
