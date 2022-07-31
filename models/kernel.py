import jax.numpy as jnp


class RBFKernel:
    def __init__(self, tau=1.0, sgm=1.0, eta=0.1):
        self.fn = rbf_kn
        self.ps = jnp.array([tau, sgm, eta])

    def nm(self, X1, X2):
        return self.fn(X1, X2, self.ps[0:2])

    def nn(self, X):
        return self.nm(X, X) + self.ps[2] ** 2 * jnp.eye(len(X))

    def mm(self, X):
        return self.nm(X, X) + jitter(len(X))


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
    tau, sgm = params

    d = jnp.sum(X1**2, 1).reshape(-1, 1) + jnp.sum(X2**2, 1) - 2 * jnp.dot(X1, X2.T)
    return tau * jnp.exp(-(0.5 / sgm**2) * d)


def jitter(n, eps=1e-6):
    return eps * jnp.eye(n)


def softplus(X):
    return jnp.log(1 + jnp.exp(X))


def softplus_inv(X):
    return jnp.log(jnp.exp(X) - 1)
