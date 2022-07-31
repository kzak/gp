import jax
import jax.numpy as jnp
import pytest
from models.kernel import *


@pytest.fixture(scope="module")
def X():
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, shape=(10, 2))
    return X


def test_rbf_kn(X):
    N, _ = X.shape

    Kx = rbf_kn(X, X, [1.0, 1.0])

    assert (N, N) == Kx.shape
    assert jnp.allclose(jnp.ones(N), jnp.diag(Kx))
