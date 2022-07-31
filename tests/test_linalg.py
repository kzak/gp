import jax
import jax.numpy as jnp
import pytest
from jax.numpy.linalg import cholesky, inv, slogdet
from jax.scipy.linalg import cho_solve
from models.kernel import *
from utils.linalg import *


@pytest.fixture(scope="module")
def X():
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, shape=(10, 2))
    return X


@pytest.fixture(scope="module")
def y(X):
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, shape=(len(X),))
    return jnp.sin(jnp.linalg.norm(X, axis=1)) + noise


def test_cho_solve(X, y):
    K = rbf_kn(X, X, [1.0, 1.0])

    K_inv_y_naive = jnp.dot(inv(K), y)
    K_inv_y_stable = cho_solve((cholesky(K), True), y)

    assert jnp.allclose(K_inv_y_naive, K_inv_y_stable, atol=1.0e-5)


def test_cholesky_logdet(X):
    K = rbf_kn(X, X, [1.0, 1.0])
    L = cholesky(K)

    logdet_naive = slogdet(K)[1]
    logdet_stable = cholesky_logdet(L)

    assert jnp.abs(logdet_naive - logdet_stable) < 1.0e-3


def test_cho_cov(X):
    Xn = X
    Xm = jax.random.normal(jax.random.PRNGKey(0), shape=(2, 2))

    Knn = rbf_kn(Xn, Xn, [1.0, 1.0])
    Knm = rbf_kn(Xn, Xm, [1.0, 1.0])
    Kmm = rbf_kn(Xm, Xm, [1.0, 1.0])

    L = cholesky(Knn)
    K_inv_Knm = cho_solve((L, True), Knm)

    cov_naive = Kmm - Knm.T @ inv(Knn) @ Knm
    cov_stable = Kmm - Knm.T.dot(K_inv_Knm)

    assert jnp.allclose(cov_naive, cov_stable, atol=1.0e-5)
