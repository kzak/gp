from pdb import set_trace

import jax.numpy as jnp
import jaxopt
from jax import grad, jit, vmap
from jax.numpy.linalg import cholesky, inv, slogdet
from jax.scipy.linalg import solve_triangular
from utils.linalg import cholesky_K_inv_y, cholesky_logdet

from models.kernel import *


class GPRegressor:
    def __init__(self, kernel):
        self.k = kernel

    def fit(self, X, y, sgm_y=1.0e-6):
        """
        Args:
            X: (n, d) matrix
            y: (n, 1) matrix
            sgm_y: observation noise
        Returns:
            this instance
        """
        self.X = X
        self.y = y
        self.sgm_y = sgm_y
        self.N, self.D = X.shape

        self.Knn = self.k(X, X) + jitter(self.N, eps=sgm_y)
        self.Knn_inv = inv(self.Knn)

        return self

    def predict(self, X_test):
        """
        Args:
            X_test: (m, d) matrix
        Returns:
            mu: mean function over X_test
            cov: covariance over X_test
        """
        y = self.y
        Knn_inv = self.Knn_inv
        Kmm = self.k(X_test, X_test)
        Knm = self.k(self.X, X_test)

        mu = Knm.T.dot(Knn_inv).dot(y)
        cov = Kmm - Knm.T.dot(Knn_inv).dot(Knm)

        return mu, cov

    def optimize(self, n_maxiter=1000, verbose=False):
        """
        Optimize kernel parameters with given data {X, y}
        """
        solver = jaxopt.ScipyMinimize(
            fun=self.loss_fn(),
            method="L-BFGS-B",
            options={"maxiter": n_maxiter, "disp": verbose},
        )

        params_pack = self.pack_params(self.k.ps)
        res = solver.run(init_params=params_pack)
        self.res = res

        kernel_params = self.unpack_params(res.params)

        self.k.ps = kernel_params
        self.Knn = self.k(self.X, self.X) + jitter(self.N, self.sgm_y)
        self.Knn_inv = inv(self.Knn)

        return self

    def pack_params(self, params):
        return jnp.log(params)

    def unpack_params(self, params):
        return jnp.exp(params)

    def loss_fn(self):
        """
        Negative Log Loss function
        -0.5 y.T @ K_inv @ y - 0.5 log|K| - 0.5*N log(2*pi)
        """

        def nll_fn(params):
            tau, sgm = self.unpack_params(params)

            K = self.k.fn(self.X, self.X, [tau, sgm]) + jitter(len(self.X), self.sgm_y)
            L = cholesky(K)

            K_inv_y = cholesky_K_inv_y(L, self.y)

            nll = 0
            nll += 0.5 * cholesky_logdet(L)
            nll += 0.5 * self.y.T.dot(K_inv_y)  # y.T L_inv L_inv y
            nll += 0.5 * self.N * jnp.log(2 * jnp.pi)

            return nll[0][0]  # nll is (1, 1) matrix

        return nll_fn
