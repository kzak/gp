import jax
import jax.numpy as jnp
import pytest
from jax.numpy.linalg import cholesky, inv, slogdet
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


def test_K_inv_y(X, y):
    K = rbf_kn(X, X, [1.0, 1.0])

    K_inv_y_naive = jnp.dot(inv(K), y)
    K_inv_y_stable = K_inv_y(K, y)

    assert jnp.allclose(K_inv_y_naive, K_inv_y_stable, atol=1.0e-5)


def test_logdet(X):
    K = rbf_kn(X, X, [1.0, 1.0])
    L = cholesky(K)

    logdet_naive = slogdet(K)[1]
    logdet_stable = cholesky_logdet(L)

    assert jnp.abs(logdet_naive - logdet_stable) < 1.0e-3
