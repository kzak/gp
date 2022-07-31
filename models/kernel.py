import jax.numpy as jnp


class Kernel:
    def __init__(self, kernel_fn, kernel_params):
        self.fn = kernel_fn
        self.ps = jnp.array(kernel_params)

    def __call__(self, X1, X2, jittering=False):
        K = self.fn(X1, X2, self.ps)

        if jittering:
            K = K + jitter(len(X1))

        return K


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
