import jax.numpy as jnp
import numpy as np
import pytest
from models.kernel import *


@pytest.fixture(scope="module")
def X():
    X = np.random.normal(0, 1, size=(10, 2))
    return X


def test_rbf_kn(X):
    N, _ = X.shape

    Kx = rbf_kn(X, X, [1.0, 1.0])

    assert (N, N) == Kx.shape
    np.testing.assert_array_almost_equal(jnp.ones(N), jnp.diag(Kx))
