import jax
import jax.numpy as jnp
import numpy as np

from jax import grad
from typing import Callable
from typing import Tuple

from functions.function_classes import polynomial_and_trigonometric_function, black_scholes, black_scholes_delta


class FunctionGenerator:
    def __init__(self, n_dim: int):
        self.n_dim = n_dim

    def generate_trigonometric_polynomial_data(
            self, n_samples: int, key: jax.random.PRNGKey, polynomial_degree: int, alpha: float, frequency: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate data from a polynomial modulated with a trigonometric function.

        :param key:
        :param polynomial_degree: Degree of polynomial.
        :param alpha:
        :param frequency: Frequency of the trigonometric function
        :return:
        """
        x = jax.random.uniform(
            key,
            shape=(n_samples, self.n_dim),
            minval=-1,
            maxval=1,
        )

        vectorized_polynomial_trigonometric = jax.vmap(
            polynomial_and_trigonometric_function,
            in_axes=(0, None, None, None, None,),
        )
        gradient_vectorized_polynomial_trigonometric = jax.vmap(
            jax.jacrev(polynomial_and_trigonometric_function, argnums=0),
            in_axes=(0, None, None, None, None,),
        )

        random_coefficients = self._compute_coefficients_polynomial(key, polynomial_degree)
        y = vectorized_polynomial_trigonometric(
            x,
            alpha,
            polynomial_degree,
            random_coefficients,
            frequency,
        )
        dydx = gradient_vectorized_polynomial_trigonometric(
            x,
            alpha,
            polynomial_degree,
            random_coefficients,
            frequency,
        )

        return np.asarray(x), np.asarray(y.reshape(-1, 1)), np.asarray(dydx)

    def _compute_coefficients_polynomial(self, key: jax.random.PRNGKey, polynomial_degree: int):
        """
        Compute random coefficients for a polynomial of degree polynomial_degree.

        The main idea is to shrink coefficients as the degree increases in order to avoid
        functions with one single extremum.

        :param key:
        :param polynomial_degree:
        :return:
        """
        random_coefficients = jax.random.uniform(
            key, shape=(self.n_dim, polynomial_degree + 1),
            minval=-1,
            maxval=1,
        ) * jnp.array([0.9 ** i for i in range(polynomial_degree + 1)])
        return random_coefficients

    def generate_black_scholes_dataset(self, m: int):
        lower = 10
        upper = 200
        sigma = 0.2
        K = 110
        T = 2
        x = np.linspace(lower, upper, m)
        y = black_scholes(x, K, sigma, T)
        y = np.maximum(0, y)
        dydx = black_scholes_delta(x, K, sigma, T)

        return x.reshape(-1, 1), y.reshape(-1, 1), dydx.reshape(-1, 1)

    def generate_bachelier_dataset(self, m: int):
        def genCorrel(n):
            np.random.seed(8192)
            randoms = np.random.uniform(low=-1., high=1., size=(2 * n, n))
            cov = randoms.T @ randoms
            invvols = np.diag(1. / np.sqrt(np.diagonal(cov)))
            return np.linalg.multi_dot([invvols, cov, invvols])

        def genWeights(n):
            np.random.seed(8192)
            w = np.random.uniform(size=n)
            return w / w.sum()

        def genVols(n, correl, weights, bkt_vol=20):
            np.random.seed(8192)
            vols = np.random.uniform(size=n)
            weighted_vols = (weights * vols).reshape(-1, 1)
            v = np.sqrt(np.linalg.multi_dot([weighted_vols.T, correl, weighted_vols]).reshape(1))
            vols = vols * bkt_vol / v
            return vols

        n_dim = 15
        bkt_vol = 20
        K = 110
        T = 3
        lower = 10
        upper = 200
        correl = genCorrel(n_dim)
        weights = genWeights(n_dim)
        vols = genVols(n_dim, correl, weights, bkt_vol)
        np.random.seed(8192)
        s0 = np.random.uniform(low=lower, high=upper, size=(m, n_dim))
        b0 = np.dot(s0, weights)
        np.random.seed(8192)
        wT = np.random.normal(size=(m, n_dim))
        vT = np.diag(vols) * np.sqrt(T)
        cov = np.linalg.multi_dot([vT, correl, vT])
        chol = np.linalg.cholesky(cov)
        sT = s0 + wT @ chol.T
        bT = np.dot(sT, weights)
        pay = np.maximum(0, bT - K)
        deriv = np.where(bT > K, 1, 0).reshape(-1, 1) * weights.reshape(1, -1)
        return s0.reshape(-1, n_dim), pay.reshape(-1, 1), deriv.reshape(-1, n_dim)

    def generate_step_function_data(
            self, n_samples: int, key: jax.random.PRNGKey, n_steps: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        step_point_key, step_value_key = jax.random.split(key)
        x = jax.random.uniform(key, shape=(n_samples, self.n_dim))
        y = jnp.zeros(n_samples)
        dydx = jnp.array([jnp.zeros(self.n_dim) for _ in range(self.n_samples)])
        step_points = jnp.sort(jax.random.uniform(step_point_key, shape=(n_steps,)))
        step_values = jnp.sort(jax.random.uniform(step_value_key, shape=(n_steps,)))
        for i, point in enumerate(x):
            # return the value of the step function at the first index larger than the current point
            y = y.at[i].set(
                sum([step_values[jnp.argmax(step_points > dim)] for dim in point])
            )

        return x.reshape(-1, self.n_dim), y.reshape(-1, 1), dydx.reshape(-1, self.n_dim)

    def generate_half_step_half_continuous(self, n_samples: int, key: jax.random.PRNGKey, polynomial_degree: int,
                                           alpha: float, frequency: float) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = jax.random.uniform(key, shape=(n_samples, self.n_dim))
        fold_points = jax.random.uniform(key, shape=(self.n_dim,))
        x = jnp.where(x < fold_points, 0, x)
        y = jnp.zeros(n_samples)
        random_coefficients = self._compute_coefficients_polynomial(key, polynomial_degree)
        dydx = jnp.array([jnp.zeros(self.n_dim) for _ in range(n_samples)])
        gradient = grad(polynomial_and_trigonometric_function)

        for i, point in enumerate(x):
            fx = polynomial_and_trigonometric_function(
                point, alpha, polynomial_degree, random_coefficients, frequency
            )
            y = y.at[i].set(fx)
            dydx = dydx.at[i].set(gradient(point, alpha, polynomial_degree, random_coefficients, frequency))

        return x.reshape(-1, self.n_dim), y.reshape(-1, 1), dydx.reshape(-1, self.n_dim)

    def generate_n_datasets(
            self,
            n_datasets: int,
            function_type: Callable[
                [jax.random.PRNGKey, int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            ],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = []
        y = []
        dydx = []
        for i in range(n_datasets):
            key = jax.random.PRNGKey(i)
            x_i, y_i, dydx_i = function_type(key, jax.random.randint(key, (1,), 1, 10))
            x.append(x_i)
            y.append(y_i)
            dydx.append(dydx_i)
        return x, y, dydx
