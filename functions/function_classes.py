import numpy as np
import jax.numpy as jnp

from scipy.stats import norm


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


def black_scholes(S, K, sigma, T):
    d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / sigma / np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def black_scholes_delta(S, K, sigma, T):
    d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / sigma / np.sqrt(T)
    return norm.cdf(d1)


# helper analytics
def bachPrice(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return vol * np.sqrt(T) * (d * norm.cdf(d) + norm.pdf(d))


def bachDelta(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return norm.cdf(d)


def bachVega(spot, strike, vol, T):
    d = (spot - strike) / vol / np.sqrt(T)
    return np.sqrt(T) * norm.pdf(d)


#

# generates a random correlation matrix
def genCorrel(n):
    randoms = np.random.uniform(low=-1., high=1., size=(2 * n, n))
    cov = randoms.T @ randoms
    invvols = np.diag(1. / np.sqrt(np.diagonal(cov)))
    return np.linalg.multi_dot([invvols, cov, invvols])


#

class Bachelier:

    def __init__(self,
                 n,
                 T1=1,
                 T2=2,
                 K=1.10,
                 volMult=1.5):

        self.n = n
        self.T1 = T1
        self.T2 = T2
        self.K = K
        self.volMult = volMult

    # training set: returns S1 (mxn), C2 (mx1) and dC2/dS1 (mxn)
    def trainingSet(self, m, anti=True, seed=None, bktVol=0.2):

        np.random.seed(seed)

        # spots all currently 1, without loss of generality
        self.S0 = np.repeat(1., self.n)
        # random correl
        self.corr = genCorrel(self.n)

        # random weights
        self.a = np.random.uniform(low=1., high=10., size=self.n)
        self.a /= np.sum(self.a)
        # random vols
        vols = np.random.uniform(low=5., high=50., size=self.n)
        # normalize vols for a given volatility of basket,
        # helps with charts without loss of generality
        avols = (self.a * vols).reshape((-1, 1))
        v = np.sqrt(np.linalg.multi_dot([avols.T, self.corr, avols]).reshape(1))
        self.vols = vols * bktVol / v
        self.bktVol = bktVol

        # Choleski etc. for simulation
        diagv = np.diag(self.vols)
        self.cov = np.linalg.multi_dot([diagv, self.corr, diagv])
        self.chol = np.linalg.cholesky(self.cov) * np.sqrt(self.T2 - self.T1)
        # increase vols for simulation of X so we have more samples in the wings
        self.chol0 = self.chol * self.volMult * np.sqrt(self.T1 / (self.T2 - self.T1))
        # simulations
        normals = np.random.normal(size=[2, m, self.n])
        inc0 = normals[0, :, :] @ self.chol0.T
        inc1 = normals[1, :, :] @ self.chol.T

        S1 = self.S0 + inc0

        S2 = S1 + inc1
        bkt2 = np.dot(S2, self.a)
        pay = np.maximum(0, bkt2 - self.K)

        # two antithetic paths
        if anti:

            S2a = S1 - inc1
            bkt2a = np.dot(S2a, self.a)
            paya = np.maximum(0, bkt2a - self.K)

            X = S1
            Y = 0.5 * (pay + paya)

            # differentials
            Z1 = np.where(bkt2 > self.K, 1.0, 0.0).reshape((-1, 1)) * self.a.reshape((1, -1))
            Z2 = np.where(bkt2a > self.K, 1.0, 0.0).reshape((-1, 1)) * self.a.reshape((1, -1))
            Z = 0.5 * (Z1 + Z2)

        # standard
        else:

            X = S1
            Y = pay

            # differentials
            Z = np.where(bkt2 > self.K, 1.0, 0.0).reshape((-1, 1)) * self.a.reshape((1, -1))

        return X, Y.reshape(-1, 1), Z

    # test set: returns an array of independent, uniformly random spots
    # with corresponding baskets, ground true prices, deltas and vegas
    def testSet(self, lower=0.5, upper=1.50, num=4096, seed=None):

        np.random.seed(seed)
        # adjust lower and upper for dimension
        adj = 1 + 0.5 * np.sqrt((self.n - 1) * (upper - lower) / 12)
        adj_lower = 1.0 - (1.0 - lower) * adj
        adj_upper = 1.0 + (upper - 1.0) * adj
        # draw spots
        spots = np.random.uniform(low=adj_lower, high=adj_upper, size=(num, self.n))
        # compute baskets, prices, deltas and vegas
        baskets = np.dot(spots, self.a).reshape((-1, 1))
        prices = bachPrice(baskets, self.K, self.bktVol, self.T2 - self.T1).reshape((-1, 1))
        deltas = bachDelta(baskets, self.K, self.bktVol, self.T2 - self.T1) @ self.a.reshape((1, -1))
        vegas = bachVega(baskets, self.K, self.bktVol, self.T2 - self.T1)
        return spots, baskets, prices, deltas, vegas
