import unittest

import jax
from matplotlib import pyplot as plt

from functions.function_generator import FunctionGenerator


class TestFunctionGenerator(unittest.TestCase):
    def test_mixed_1d(self):
        generator = FunctionGenerator(n_dim=1)
        key = jax.random.PRNGKey(1)
        x, y, dy_dx = generator.generate_half_step_half_continuous(
            n_samples=100,
            key=key,
            polynomial_degree=8,
            alpha=0.9,
            frequency=2,
        )
        plt.scatter(x, y)
        plt.scatter(x, dy_dx, label="dy_dx")
        plt.show()

    def test_step_1d(self):
        generator = FunctionGenerator(n_dim=1)
        key = jax.random.PRNGKey(1)
        x, y, dy_dx = generator.generate_step_function_data(n_samples=1000, key=key, n_steps=2,)
        plt.scatter(x, y)
        plt.scatter(x, dy_dx, label="dy_dx")
        plt.show()

    def test_step_2d(self):
        generator = FunctionGenerator(n_dim=2)
        key = jax.random.PRNGKey(1)
        x, y, dy_dx = generator.generate_step_function_data(n_samples=1000, key=key, n_steps=2,)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], y)
        plt.show()
    def test_generate_trigonometric_polynomial_data_1d(self):
        generator = FunctionGenerator(n_dim=1)
        key = jax.random.PRNGKey(1)
        x, y, dy_dx = generator.generate_trigonometric_polynomial_data(
            n_samples=1000,
            key=key,
            polynomial_degree=8,
            alpha=0.9,
            frequency=2,
        )
        plt.scatter(x, y)
        plt.scatter(x, dy_dx, label="dy_dx")
        plt.show()

    def test_generate_trigonometric_polynomial_data_2d(self):
        generator = FunctionGenerator(n_dim=2)
        key = jax.random.PRNGKey(1)
        x, y, dy_dx = generator.generate_trigonometric_polynomial_data(
            n_samples=10000,
            key=key,
            polynomial_degree=8,
            alpha=0.9,
            frequency=2,
        )
        # 3d scatter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], y)
        plt.show()

    def test_black_scholes_generator(self):
        generator = FunctionGenerator(n_dim=1)
        x, y, dydx = generator.generate_black_scholes_dataset(0.2, 110, 2, 200)

        plt.scatter(x, y)
        plt.scatter(x, dydx, label="dy_dx")
        plt.show()
