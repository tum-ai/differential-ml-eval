import unittest

from functions.function_classes import polynomial_function
import jax.numpy as jnp

class TestFunctionClasses(unittest.TestCase):
    def test_polynomial(self):
        result = polynomial_function(
            x=jnp.array([1, 2, 3]),
            polynomial_degree=2,
            coefficients=jnp.array([[1, 2], [3, 4], [5, 6]]),
        )
        self.assertEqual(result, 109)
