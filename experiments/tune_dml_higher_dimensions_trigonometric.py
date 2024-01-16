import threading
from typing import Callable

import jax
import jax.numpy as jnp

import optuna
import torch.nn
from optuna import Trial

from functions.function_generator import FunctionGenerator
from train_and_eval import train

def build_objective(vanilla_network: bool, data_generator: Callable, n_datapoints: int):
    def objective(trial: Trial):
        n_train = n_datapoints
        lr_dml = trial.suggest_float('lr_dml', 0.001, 0.01, log=True)
        if not vanilla_network:
            lambda_ = 1  # trial.suggest_float('lambda_', 0.01, 10, log=True)
        else:
            lambda_ = 0.5
        n_layers = trial.suggest_int('n_layers', 1, 5)
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [32, 64, 128, 256])
        batch_size = 1024  # trial.suggest_categorical('batch_size', [64, 128, 256])
        n_epochs = 1000  # trial.suggest_categorical('epochs', [200, 500, 1000])
        activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])
        regularization_scale = trial.suggest_float('regularization_scale', 0.00001, 1, log=True)
        test_loss = train(
            data_generator=data_generator,
            n_train=n_train,
            lr_dml=lr_dml,
            lambda_=lambda_,
            n_layers=n_layers,
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            n_epochs=n_epochs,
            activation_identifier=activation,
            regularization_scale=regularization_scale,
        )
        return test_loss
    return objective


def main(vanilla_network: bool, dimensions: int, n_datapoints: int):
    generator = FunctionGenerator(n_dim=dimensions)

    def f(n_data: int):
        return generator.generate_trigonometric_data(
            n_samples=n_data,
            key=jax.random.PRNGKey(0),
            frequencies=jnp.array([5, 3, 1, 1, 0.1, 0.3, 0.1, 0.1]),
            amplitudes=jnp.array([1, 2, 1, 0.5, 0.1, 0.1, 1, 1]),
            #frequencies=jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            #amplitudes=jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            #frequencies=jnp.array([1, 1]),  # jnp.array([5, 3, 1, 1, 0.1, 0.3, 0.1, 0.1, 0.1]),
            #amplitudes=jnp.array([1, 1]),  # jnp.array([1, 2, 1, 0.5, 0.1, 0.1, 1, 1, 1]),
        )

    vanilla_indicator = "vanilla" if vanilla_network else "dml"
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=f"reg-trigonometric-data={n_datapoints}-dimensions={dimensions}-{vanilla_indicator}",
        direction="minimize",
        load_if_exists=True,
    )
    objective = build_objective(vanilla_network=vanilla_network, data_generator=f, n_datapoints=n_datapoints)
    study.optimize(objective, n_trials=100)

    study.best_params


if __name__ == "__main__":
    # start main two times in parallel with different arguments
    # one with vanilla_network=True and one with vanilla_network=False
    # Parallel start using threads
    # main(True, 2, 2000)
    t1 = threading.Thread(target=main, args=(True, 8, 512))
    t2 = threading.Thread(target=main, args=(False, 8, 512))
    t1.start()
    t2.start()
