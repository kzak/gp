import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.numpy.linalg import inv
from models.kernel import *
from utils.linalg import *


@pytest.fixture(scope="module")
def X():
    X = np.random.normal(0, 1, size=(10, 2))
    return X


@pytest.fixture(scope="module")
def y(X):
    noise = jax.random.normal(jax.random.PRNGKey(0), shape=(len(X),))
    return jnp.sin(jnp.linalg.norm(X, axis=1)) + noise


def test_K_inv_y(X, y):
    K = rbf_kn(X, X, [1.0, 1.0])

    K_inv_y_naive = jnp.dot(inv(K), y)
    K_inv_y_stable = K_inv_y(K, y)

    assert jnp.allclose(K_inv_y_naive, K_inv_y_stable, atol=1.0e-5)
