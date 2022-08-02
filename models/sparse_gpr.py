from pdb import set_trace

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt
import numpy as np
from jax import jit, random, value_and_grad
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import cho_solve
from scipy.optimize import minimize
from utils.linalg import cholesky_logdet

from models.kernel import *


class SGPRegressor:
    def __init__(self, kernel):
        self.k = kernel

    def fit(self, X, y, n_inducing=10, sgm_y=0.1):
        """
        Args:
            X: (n, d) matrix
            y: (n, 1) matrix
        Returns:
            this instance
        """
        self.X = X
        self.y = y
        self.sgm_y = sgm_y
        self.N, self.D = X.shape
        self.n_inducing = n_inducing

        self.inducing_points = random.choice(
            random.PRNGKey(0), self.X, (n_inducing,), replace=False
        )

        return self

    def optimize(self, X_m, n_maxiter=1000, verbose=False):
        """
        Optimize kernel parameters with given data {X, y}
        """
        solver = jaxopt.ScipyMinimize(
            fun=self.loss_fn(),
            method="L-BFGS-B",
            options={"maxiter": n_maxiter, "disp": verbose},
        )

        params_pack = self.pack_params(jnp.array([1.0, 1.0]), self.inducing_points)
        res = solver.run(init_params=params_pack)
        self.res = res

        self.k.ps, self.inducing_points = self.unpack_params(res.params)

        return self

    def pack_params(self, kernel_params, inducing_inputs):
        return jnp.hstack([softplus_inv(kernel_params), inducing_inputs.ravel()])

    def unpack_params(self, params):
        kernel_params, inducing_inputs = jnp.split(params, [2])
        return softplus(kernel_params), inducing_inputs.reshape(self.n_inducing, self.D)

    def loss_fn(self):
        def nlb(params):
            sigma_y = self.sgm_y
            kernel_params, X_m = self.unpack_params(params)
            K_mm = self.k.fn(X_m, X_m, kernel_params) + jitter(X_m.shape[0])
            K_mn = self.k.fn(X_m, self.X, kernel_params)

            L = jnp.linalg.cholesky(K_mm)  # m x m
            A = jsp.linalg.solve_triangular(L, K_mn, lower=True) / sigma_y  # m x n
            AAT = A @ A.T  # m x m
            B = jnp.eye(X_m.shape[0]) + AAT  # m x m
            LB = jnp.linalg.cholesky(B)  # m x m
            c = (
                jsp.linalg.solve_triangular(LB, A.dot(self.y), lower=True) / sigma_y
            )  # m x 1

            K_diag = jnp.full(shape=self.N, fill_value=kernel_params[0] ** 2)

            lb = -self.N / 2 * jnp.log(2 * jnp.pi)
            lb -= jnp.sum(jnp.log(jnp.diag(LB)))
            lb -= self.N / 2 * jnp.log(sigma_y**2)
            lb -= 0.5 / sigma_y**2 * self.y.T.dot(self.y)
            lb += 0.5 * c.T.dot(c)
            lb -= 0.5 / sigma_y**2 * jnp.sum(K_diag)
            lb += 0.5 * jnp.trace(AAT)

            return -lb[0, 0]

        return nlb

    def phi_opt(self):
        """Optimize mu_m and A_m using Equations (11) and (12)."""
        sigma_y = self.sgm_y
        precision = 1.0 / sigma_y**2
        X_m = self.inducing_points

        K_mm = self.k.fn(X_m, X_m, self.k.ps[0:2]) + jitter(X_m.shape[0])
        K_mm_inv = jnp.linalg.inv(K_mm)
        K_nm = self.k.fn(self.X, X_m, self.k.ps[0:2])
        K_mn = K_nm.T

        Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)

        mu_m = precision * (K_mm @ Sigma @ K_mn).dot(self.y)
        A_m = K_mm @ Sigma @ K_mm

        return mu_m, A_m, K_mm_inv

    def predict(self, X_test):
        """p(f*|y) = q(f*) = \int p(f*|f) phi(fm) dfm"""
        mu_m, A_m, K_mm_inv = self.phi_opt()

        K_ss = self.k.fn(X_test, X_test, self.k.ps[0:2])
        K_sm = self.k.fn(X_test, self.inducing_points, self.k.ps[0:2])
        K_ms = K_sm.T

        f_q = (K_sm @ K_mm_inv).dot(mu_m)
        f_q_cov = (
            K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
        )

        return f_q, f_q_cov
