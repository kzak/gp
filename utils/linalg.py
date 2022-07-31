import jax
import jax.numpy as jnp
from jax.numpy.linalg import cholesky, inv, slogdet
from jax.scipy.linalg import solve_triangular


def K_inv_y(K, y):
    """
    Stable caclulation of K^{-1} y
    K^{-1} y = (L L.T)^{-1} y
    Args:
        K : (n, n) positive semi-definite matrix
        y : (n, 1) matrix / vector
    Returns:
        (n, 1) matrix / vector of K^{-1} y
    """

    L = cholesky(K)

    # s1 = L_inv y
    s1 = solve_triangular(L, y, lower=True)
    # s2 = L_inv.T (L_inv y)
    s2 = solve_triangular(L.T, s1, lower=False)
    return s2


def cholesky_logdet(L):
    """
    Args:
        L : Cholesky decomposition of a gram matrix K
    Returns:
        log determinant of K
    """

    # det(K) = det(L L.T) = det(L) det(L.T) = det(L) ** 2
    # logdet(K) = 2 * det(L) = 2 * \sum log(L_ii)

    return 2 * jnp.sum(jnp.log(jnp.diagonal(L)))
