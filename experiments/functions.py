import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import grad
from jax import random


class FunctionGenerator:

    def __init__(self, n_samples, n_dim):
        self.n_samples = n_samples
        self.n_dim = n_dim

    def generate_trigonometric_polynomial_data(self, polynomial_degree=3):
        # Generate random points in the given range for each dimension
        key = random.PRNGKey(1234)
        x = random.uniform(key, shape=(self.n_samples, self.n_dim))
        y = jnp.zeros(self.n_samples)
        dydx = jnp.array([jnp.zeros(self.n_dim) for _ in range(self.n_samples)])
        alpha = random.uniform(key, shape=(self.n_dim,))
        gradient = grad(FunctionGenerator.polynomial_and_trigonometric_function)

        for i, point in enumerate(x):
            fx = FunctionGenerator.polynomial_and_trigonometric_function(point, alpha, polynomial_degree)
            y = y.at[i].set(fx)
            dydx = dydx.at[i].set(gradient(point, alpha, polynomial_degree))

        return x.reshape(-1, self.n_dim), y.reshape(-1, 1), dydx.reshape(-1, self.n_dim)

    def generate_step_function_data(self, n_steps):
        x_key = random.PRNGKey(5678)
        step_point_key, step_value_key = random.split(x_key)
        x = random.uniform(x_key, shape=(self.n_samples, self.n_dim))
        y = jnp.zeros(self.n_samples)
        dydx = jnp.array([jnp.zeros(self.n_dim) for _ in range(self.n_samples)])
        step_points = jnp.sort(random.uniform(step_point_key, shape=(n_steps,)))
        step_values = jnp.sort(random.uniform(step_value_key, shape=(n_steps,)))
        for i, point in enumerate(x):
            #return the value of the step function at the first index larger than the current point
            y = y.at[i].set(sum([step_values[jnp.argmax(step_points > dim)] for dim in point]))

        return x.reshape(-1, self.n_dim), y.reshape(-1, 1), dydx.reshape(-1, self.n_dim)

    @staticmethod
    def polynomial_and_trigonometric_function(x, alpha, polynomial_degree=3):
        return jnp.sum(
           jnp.array([(1 - alpha[dim]) * x[dim] ** (polynomial_degree - dim)
                      + alpha[dim] * jnp.sin(x[dim]) * 10 ** (polynomial_degree - dim)
                      for dim in range(len(x))])
       )

gen = FunctionGenerator(100, 3)
x, y, dydx = gen.generate_trigonometric_polynomial_data(7)
#plt.scatter(x, y)
#plt.show()


