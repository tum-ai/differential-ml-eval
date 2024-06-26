import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "differential_ml"))

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

from train_and_eval import train, train_only


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

    def compare_pricing_approximation(self, x_test, y_test, y_preds, title):

        plt.figure(figsize=(10, 6))
        plt.scatter(x_test, y_test, color='blue', label='Actual')
        plt.scatter(x_test, y_preds, color='red', alpha=0.5, label='Predicted')
        plt.title(title)
        plt.xlabel('x_test')
        plt.ylabel('y values')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # basket / bachelier dimension
    basketDim = 1

    # simulation set sizes to perform
    sizes = [4096, 8192, 16384]

    # show delta?
    showDeltas = True
    deltidx = 0  # show delta to first stock

    # seed
    simulSeed = 6004
    # simulSeed = np.random.randint(0, 10000)
    print("using seed %d" % simulSeed)
    testSeed = None
    weightSeed = None

    # number of test scenarios
    nTest = 4096

    # go
    bachelier = Bachelier(basketDim)
    x_train, y_train, dydx_train = bachelier.trainingSet(4096, False, simulSeed)
    x_test, xAxis, y_test, dydx_test, _ = bachelier.testSet(num=1024, seed=simulSeed)

    #dml_x_test, dml_y_test, preds_dml = train_only(x_train, y_train, dydx_train, x_test, y_test, dydx_test, shuffle=False)
    #vanilla_x_test, vanilla_y_test, preds_vanilla = train_only(x_train, y_train, dydx_train, x_test, y_test, dydx_test, lambda_=0, shuffle=False)
    #bachelier.compare_pricing_approximation(xAxis, dml_y_test, preds_dml, "DML Fit")
    #bachelier.compare_pricing_approximation(xAxis, vanilla_y_test, preds_vanilla, "Vanilla Fit")

    plt.scatter(xAxis, dydx_test, s=.1)
    plt.scatter(xAxis, y_test, s=.1)
    # plt.scatter(x_test_shuffled, y_out_test)
    # plt.scatter(x_test_shuffled, targets_test)
    plt.savefig("dydx_preds")
    plt.show()
    print("-- done! ---")

