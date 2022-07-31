from pdb import set_trace

import jax.numpy as jnp
import jaxopt
from jax import grad, jit, vmap
from jax.numpy.linalg import cholesky, inv, slogdet
from jax.scipy.linalg import cho_solve, solve_triangular
from utils.linalg import cholesky_K_inv_y, cholesky_logdet

from models.kernel import *


class GPRegressor:
    def __init__(self, kernel):
        self.k = kernel

    def fit(self, X, y):
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
        self.N, self.D = X.shape

        return self

    def predict(self, X_test, naive=True):
        """
        Args:
            X_test: (m, d) matrix
        Returns:
            mu: mean function over X_test
            cov: covariance over X_test
        """
        y = self.y
        Knn = self.k.nn(self.X)
        L = cholesky(Knn)

        K_inv_y = cho_solve((L, True), y)
        # Knn_inv = inv(Knn)

        Kmm = self.k.nm(X_test, X_test) + jitter(len(X_test))
        Knm = self.k.nm(self.X, X_test)

        mu = Knm.T.dot(K_inv_y)
        # mu = Knm.T.dot(Knn_inv).dot(y)

        K_inv_Knm = cho_solve((L, True), Knm)
        cov = Kmm - Knm.T.dot(K_inv_Knm)
        # cov = Kmm - Knm.T.dot(Knn_inv).dot(Knm)

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

        self.k.ps = self.unpack_params(res.params)

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
            self.k.ps = self.unpack_params(params)
            K = self.k.nn(self.X)
            L = cholesky(K)

            K_inv_y = cho_solve((L, True), self.y)

            nll = 0
            nll += 0.5 * cholesky_logdet(L)
            nll += 0.5 * self.y.T.dot(K_inv_y)  # y.T L_inv L_inv y
            nll += 0.5 * self.N * jnp.log(2 * jnp.pi)

            return nll[0][0]  # nll is (1, 1) matrix

        return nll_fn
