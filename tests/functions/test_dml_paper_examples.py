import numpy as np
import matplotlib.pyplot as plt
import pytest

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from functions.function_generator import FunctionGenerator
from differential_ml.util.data_util import DataNormalizer
from train_and_eval import train


@pytest.fixture
def gen():
    return FunctionGenerator(1)


@pytest.fixture
def linreg():
    return make_pipeline(PolynomialFeatures(degree=5), LinearRegression())


@pytest.fixture
def ridgereg():
    alphas = np.exp(np.linspace(np.log(1e-05), np.log(1e02), 100))
    return make_pipeline(PolynomialFeatures(degree=5), RidgeCV(alphas=alphas))


@pytest.fixture
def normalizer():
    return DataNormalizer()


def test_black_scholes_differential_fit(gen, linreg, ridgereg, normalizer):
    x_train, y_train, dydx_train = gen.generate_black_scholes_dataset(200)
    x_test, y_test, dydx_test = gen.generate_black_scholes_dataset(5000)

    normalizer.initialize_with_data(x_raw=x_train, y_raw=y_train, dydx_raw=dydx_train)
    x_train_normalized, y_train_normalized, dy_dx_train_normalized = normalizer.normalize_all(
        x_train,
        y_train,
        dydx_train,
    )
    x_test_normalized, y_test_normalized, dydx_test_normalized = normalizer.normalize_all(
        x_test,
        y_test,
        dydx_test,
    )

    linreg.fit(x_train_normalized, y_train_normalized)
    linpred = linreg.predict(x_test_normalized)
    linloss = np.sum((linpred - y_test_normalized) ** 2)

    ridgereg.fit(x_train_normalized, y_train_normalized)
    ridgepred = ridgereg.predict(x_test_normalized)
    ridgeloss = np.sum((ridgepred - y_test_normalized) ** 2)

    diff_loss = train(
        data_generator=gen.generate_black_scholes_dataset,
        n_train=200,
        n_test=5000,
        plot_when_finished=True
    )

    vanilla_loss = train(
        data_generator=gen.generate_black_scholes_dataset,
        n_train=200,
        n_test=5000,
        lambda_=0,
        plot_when_finished=True
    )

    assert diff_loss < linloss and diff_loss < ridgeloss and diff_loss < vanilla_loss


def test_bachelier_differential_fit(gen, normalizer):
    diff_loss = train(
        data_generator=gen.generate_bachelier_dataset,
        n_train=1000,
        n_test=5000,
        plot_when_finished=True
    )

    vanilla_loss = train(
        data_generator=gen.generate_bachelier_dataset,
        n_train=1000,
        n_test=5000,
        lambda_=0,
        plot_when_finished=True
    )

    assert diff_loss < vanilla_loss
