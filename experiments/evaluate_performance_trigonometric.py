import datetime
import json
import os
import threading
from typing import Callable

import jax
import jax.numpy as jnp

import optuna
import torch.nn
from optuna import Trial

from functions.function_generator import FunctionGenerator
from train_and_eval import train


def build_objective(
    vanilla_network: bool,
    data_generator: Callable,
    n_datapoints: int,
):
    def objective(trial: Trial):
        n_train = n_datapoints
        lr_dml = trial.suggest_float('lr_dml', 0.0001, 0.01, log=True)
        if not vanilla_network:
            lambda_ = 1
        else:
            lambda_ = 0
        n_layers = trial.suggest_int('n_layers', 1, 5)
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [8, 32, 64, 256])
        batch_size = trial.suggest_categorical('batch_size', [1024])
        n_epochs = trial.suggest_categorical('epochs', [1000])
        activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])
        regularization_scale = trial.suggest_categorical('regularization_scale', [0])
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
            progress_bar_disabled=True,
            plot_when_finished=False,
        )

        return test_loss
    return objective


def tune_hyperparameters(vanilla_network: bool, dimensions: int, n_datapoints: int, data_generator: Callable, n_tuning_steps: int):
    vanilla_indicator = "vanilla" if vanilla_network else "dml"
    identifier = f"systematic-trigonometric-data={n_datapoints}-dimensions={dimensions}-{vanilla_indicator}"
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=identifier,
        direction="minimize",
        load_if_exists=True,
    )
    objective = build_objective(
        vanilla_network=vanilla_network,
        data_generator=data_generator,
        n_datapoints=n_datapoints,
    )
    study.optimize(objective, n_trials=n_tuning_steps)

    print(f"Best value ({vanilla_indicator}): {study.best_params}")
    output_dir = f"results/trigonometric-polynomial-{dimensions}dim-{n_datapoints}data"
    os.makedirs(output_dir, exist_ok=True)
    json.dump(study.best_params, open(f"{output_dir}/best_params_{identifier}.json", "w"))

    n_runs = 10
    test_losses = {}
    for i in range(n_runs):
        jax_key = jax.random.PRNGKey(i)
        def f(n_data: int):
            return generator.generate_trigonometric_data(
                n_samples=n_data,
                key=jax_key,
                frequencies=jax.random.uniform(key=jax_key, minval=0.1, maxval=10, shape=(n_dimensions,)),
                amplitudes=jax.random.uniform(key=jax_key, minval=0.1, maxval=10, shape=(n_dimensions,)),
            )
        test_losses[i] = train(
            data_generator=f,
            n_train=n_datapoints,
            lr_dml=study.best_params['lr_dml'],
            lambda_=0 if vanilla_network else 1,
            n_layers=study.best_params['n_layers'],
            hidden_layer_sizes=study.best_params['hidden_layer_sizes'],
            batch_size=study.best_params['batch_size'],
            n_epochs=study.best_params['epochs'],
            activation_identifier=study.best_params['activation'],
            regularization_scale=study.best_params['regularization_scale'],
            progress_bar_disabled=True,
            plot_when_finished=False,
        )
        #print(f"Running loss computation {vanilla_indicator} = {test_losses / (i + 1)}")
    #print(f"Loss after tuning {identifier} = {test_losses / n_runs} ")
    json_path = os.path.join(output_dir, f"losses_{vanilla_indicator}.json")
    json.dump(test_losses, open(json_path, "w"))


if __name__ == "__main__":
    n_data_points = 512
    n_dimensions = 2
    n_tuning_steps = 20

    generator = FunctionGenerator(n_dim=n_dimensions)

    def f_trigonometric(n_data: int):
        jax_key = jax.random.PRNGKey(0)
        return generator.generate_trigonometric_data(
            n_samples=n_data,
            key=jax_key,
            frequencies=jax.random.uniform(key=jax_key, minval=0.1, maxval=10, shape=(n_dimensions,)),
            amplitudes=jax.random.uniform(key=jax_key, minval=0.1, maxval=10, shape=(n_dimensions,)),
        )

    def f_trigonometric_and_polynomial(n_data: int):
        jax_key = jax.random.PRNGKey(0)
        return generator.generate_trigonometric_polynomial_data(
            n_samples=n_data,
            key=jax_key,
            polynomial_degree=3,
            alpha=0.9,
            frequency=2,
        )

    thread_vanilla = threading.Thread(target=tune_hyperparameters, args=(True, n_dimensions, n_data_points, f_trigonometric_and_polynomial, n_tuning_steps))
    thread_dml = threading.Thread(target=tune_hyperparameters, args=(False, n_dimensions, n_data_points, f_trigonometric_and_polynomial, n_tuning_steps))

    thread_vanilla.start()
    thread_dml.start()

    # wait for threads to end and get return value
    thread_vanilla.join()
    thread_dml.join()
