import threading
from typing import Callable

import jax
import jax.numpy as jnp

import optuna
import torch.nn
import matplotlib.pyplot as plt
from optuna import Trial

from functions.function_generator import FunctionGenerator
from paper_examples import Bachelier
from train_and_eval import train, train_only


def build_objective(vanilla_network: bool, data_dict, n_datapoints: int):
    def objective(trial: Trial):
        n_train = n_datapoints
        lr_dml = trial.suggest_float('lr_dml', 0.001, 0.01, log=True)
        if not vanilla_network:
            lambda_ = .001  # trial.suggest_float('lambda_', 0.01, 10, log=True)
        else:
            lambda_ = 0
        n_layers = trial.suggest_int('n_layers', 1, 5)
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [32, 64, 128, 256])
        batch_size = 1024  # trial.suggest_categorical('batch_size', [64, 128, 256])
        n_epochs = trial.suggest_categorical('epochs', [200, 500, 1000])
        #activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])
        regularization_scale = trial.suggest_float('regularization_scale', 0.00001, 1, log=True)
        test_loss = train_only(
            data_dict["x_train"],
            data_dict["y_train"],
            data_dict["dydx_train"],
            data_dict["x_test"],
            data_dict["y_test"],
            data_dict["dydx_test"],
            shuffle=False,
            lr_dml=lr_dml,
            lambda_=lambda_,
            n_layers=n_layers,
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            n_epochs=n_epochs,
            activation_identifier='softplus',
            regularization_scale=regularization_scale
        )
        return test_loss

    return objective


def main(vanilla_network: bool, dimensions: int, n_datapoints: int):
    bachelier = Bachelier(dimensions)  # basket dimension
    generator = FunctionGenerator(n_dim=dimensions)
    # sizes = [4096, 8192, 16384]

    # show delta?
    # showDeltas = True
    # deltidx = 0  # show delta to first stock

    # seed
    simulSeed = 6000
    # simulSeed = np.random.randint(0, 10000)
    # print("using seed %d" % simulSeed)
    # testSeed = None
    # weightSeed = None

    # number of training examples
    n_train = int(jnp.floor(n_datapoints * 0.8))

    # number of test scenarios
    n_test = int(jnp.floor(n_datapoints * 0.2))

    x_train, y_train, dydx_train = bachelier.trainingSet(n_train, False, simulSeed)
    x_test, xAxis, y_test, dydx_test, _ = bachelier.testSet(num=n_test, seed=simulSeed)
    jax_key = jax.random.PRNGKey(0)

    # x_train, y_train, dydx_train = generator.generate_trigonometric_data(
    #         n_samples=n_train,
    #         key=jax_key,
    #         frequencies=jax.random.uniform(key=jax_key, minval=0.1, maxval=10, shape=(dimensions,)),
    #         amplitudes=jax.random.uniform(key=jax_key, minval=0.1, maxval=10, shape=(dimensions,)),
    #     )
    # x_test, y_test, dydx_test = generator.generate_trigonometric_data(
    #         n_samples=n_test,
    #         key=jax_key,
    #         frequencies=jax.random.uniform(key=jax_key, minval=0.1, maxval=10, shape=(dimensions,)),
    #         amplitudes=jax.random.uniform(key=jax_key, minval=0.1, maxval=10, shape=(dimensions,)),
    #     )

    data_dict = {"x_train": x_train, "y_train": y_train, "dydx_train": dydx_train, "x_test": x_test, "xAxis": xAxis,
                 "y_test": y_test, "dydx_test": dydx_test}

    plt.scatter(x_train, y_train)
    plt.show()
    plt.scatter(xAxis, y_test, s=.1)
    plt.show()
    plt.scatter(xAxis, dydx_test, s=.1)
    plt.show()
    plt.scatter(x_train, dydx_train)
    plt.show()

    vanilla_indicator = "vanilla" if vanilla_network else "dml"
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=f"bachelier={n_datapoints}-dimensions={dimensions}-{vanilla_indicator}",
        direction="minimize",
        load_if_exists=True,
    )
    objective = build_objective(vanilla_network=vanilla_network, data_dict=data_dict, n_datapoints=n_datapoints)
    study.optimize(objective, n_trials=30)

    study.best_params


if __name__ == "__main__":
    # start main two times in parallel with different arguments
    # one with vanilla_network=True and one with vanilla_network=False
    # Parallel start using threads
    # main(True, 2, 2000)

    #t1 = threading.Thread(target=main, args=(True, 1, 4096))
    t2 = threading.Thread(target=main, args=(False, 1, 4096))
    #t1.start()
    t2.start()
