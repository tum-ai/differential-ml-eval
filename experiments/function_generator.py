import jax.numpy as jnp

from jax import grad
from jax import random
from typing import Callable
from typing import Tuple


class FunctionGenerator:

    def __init__(self, n_samples: int, n_dim: int):
        self.n_samples = n_samples
        self.n_dim = n_dim

    def generate_trigonometric_polynomial_data(self, key: random.PRNGKey, polynomial_degree: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    def generate_step_function_data(self, key: random.PRNGKey, n_steps: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        step_point_key, step_value_key = random.split(key)
        x = random.uniform(key, shape=(self.n_samples, self.n_dim))
        y = jnp.zeros(self.n_samples)
        dydx = jnp.array([jnp.zeros(self.n_dim) for _ in range(self.n_samples)])
        step_points = jnp.sort(random.uniform(step_point_key, shape=(n_steps,)))
        step_values = jnp.sort(random.uniform(step_value_key, shape=(n_steps,)))
        for i, point in enumerate(x):
            #return the value of the step function at the first index larger than the current point
            y = y.at[i].set(sum([step_values[jnp.argmax(step_points > dim)] for dim in point]))

        return x.reshape(-1, self.n_dim), y.reshape(-1, 1), dydx.reshape(-1, self.n_dim)

    def generate_n_datasets(self, n_datasets: int, function_type: Callable[[random.PRNGKey, int], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = []
        y = []
        dydx = []
        for i in range(n_datasets):
            key = random.PRNGKey(i)
            x_i, y_i, dydx_i = function_type(key, random.randint(key, (1,),1, 10))
            x.append(x_i)
            y.append(y_i)
            dydx.append(dydx_i)
        return x, y, dydx

    @staticmethod
    def polynomial_and_trigonometric_function(x, alpha, polynomial_degree=3):
        return jnp.sum(
           jnp.array([(1 - alpha[dim]) * x[dim] ** (polynomial_degree - dim)
                      + alpha[dim] * jnp.sin(x[dim]) * 10 ** (polynomial_degree - dim)
                      for dim in range(len(x))])
       )

gen = FunctionGenerator(100, 1)
x, y, dydx = gen.generate_n_datasets(3, gen.generate_trigonometric_polynomial_data)
print(len(x), len(y), len(dydx))
print(x[0].shape, y[0].shape, dydx[0].shape)
print(x[0][0], y[0][0], dydx[0][0])






