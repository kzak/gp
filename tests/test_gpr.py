import jax.numpy as jnp
import numpy as np
import pytest
from models.gpr import *
from models.kernel import *


@pytest.fixture(scope="module")
def X():
    X = np.random.normal(0, 1, size=(10, 2))
    return jnp.array(X)


@pytest.fixture(scope="module")
def y(X):
    return jnp.sin(X) + np.random.normal(0, 1, size=X.shape)


def test_gpr(X, y):
    m = GPRegressor(RBFKernel(1.0, 1.0, 0.1))
    m.fit(X, y)
    m.optimize()

    X_test = np.random.normal(0, 1, size=(1, 2))
    mu, cov = m.predict(X_test)

    assert len(X_test) == len(mu)
    assert (len(X_test), len(X_test)) == cov.shape
