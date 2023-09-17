import jax.numpy as jnp


def polynomial_function(
    x: jnp.ndarray, polynomial_degree: int, coefficients: jnp.ndarray
):
    """
    Create a polynomial function of degree polynomial_degree

    :param x:
    :param polynomial_degree:
    :return:
    """

    return jnp.sum(jnp.sum(
        jnp.array(
            [
                [
                    coefficients[dim, exponent] * x[dim] ** exponent
                    for exponent in range(polynomial_degree + 1)
                ]
                for dim in range(x.shape[0])
            ]
        ),
    ))


def polynomial_and_trigonometric_function(
    x: jnp.ndarray,
    alpha: float,
    polynomial_degree: int,
    coefficients: jnp.ndarray,
    frequency: float,
):
    """

    :param x:
    :param alpha: Interpolation parameter, alpha = 0 is polynomial, alpha = 1 is trigonometric
    :param polynomial_degree:
    :return:
    """
    result = alpha * polynomial_function(x, polynomial_degree, coefficients) + (
        1 - alpha
    ) * jnp.sum(
        jnp.array(
            [jnp.cos(frequency * x[dim]) for dim in range(x.shape[0])]
        )
    )
    return result
