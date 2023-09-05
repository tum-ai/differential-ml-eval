import jax.numpy as jnp

from jax import grad
from jax import random


class FunctionGenerator:

    def __init__(self, n_samples, n_dim):
        self.n_samples = n_samples
        self.n_dim = n_dim

    def generate_trigonometric_polynomial_data(self, polynomial_degree=3, alpha=0.5):
        # Generate random points in the given range for each dimension
        key = random.PRNGKey(1234)
        x = random.uniform(key, shape=(self.n_samples, self.n_dim))
        y = jnp.zeros(self.n_samples)
        dydx = jnp.array([jnp.zeros(self.n_dim) for _ in range(self.n_samples)])
        gradient = grad(FunctionGenerator.polynomial_and_trigonometric_function)

        for i, point in enumerate(x):
            fx = FunctionGenerator.polynomial_and_trigonometric_function(point, polynomial_degree, alpha)
            y = y.at[i].set(fx)
            dydx = dydx.at[i].set(gradient(point))

        return x.reshape(-1, self.n_dim), y.reshape(-1, 1), dydx.reshape(-1, self.n_dim)

    def generate_step_function_data(self, nsteps):
        x = jnp.random.uniform(-10, 10, size=(self.n_samples, self.n_dim))
        y = jnp.zeros(self.n_samples)
        dydx = jnp.array([jnp.zeros(self.n_dim) for _ in range(self.n_samples)])
        step_points = sorted(jnp.random.uniform(-10, 10, size=nsteps))
        step_values = sorted(jnp.random.uniform(-10, 10, size=nsteps))
        for i, point in enumerate(x):
            y[i] = sum([step_values[jnp.argmax(step_points > dim)] for dim in point])

        return x.reshape(-1, self.n_dim), y.reshape(-1, 1), dydx.reshape(-1, self.n_dim)

    @staticmethod
    def polynomial_and_trigonometric_function(x, polynomial_degree=3, alpha=0.5):
        return jnp.sum(
           jnp.array([(1 - alpha) * x[dim] ** (polynomial_degree - dim)
                      + alpha * jnp.sin(x[dim]) * 10 ** (polynomial_degree - dim)
                      for dim in range(len(x))])
       )

