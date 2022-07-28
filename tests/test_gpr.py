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
    m = GPRegressor(Kernel(rbf_kn, [1.0, 1.0]))
    m.fit(X, y)
    m.optimize()

    X_test = np.random.normal(0, 1, size=(1, 2))
    mu, cov = m.predict(X_test)

    assert len(X_test) == len(mu)
    assert (len(X_test), len(X_test)) == cov.shape


def test_nloglik(X, y):
    kernel_params = jnp.array([1.0, 1.0])
    gpr = GPRegressor(Kernel(rbf_kn, kernel_params)).fit(X, y)

    nll_fn_stable = gpr.loss_fn()
    nll_fn_naive = gpr.loss_fn(naive=True)

    nll_stable = nll_fn_stable(kernel_params)
    nll_naive = nll_fn_naive(kernel_params)

    assert jnp.abs(nll_naive - nll_stable) < 1.0e-3
