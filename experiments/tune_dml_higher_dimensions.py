import threading
from typing import Callable

import jax

import optuna
from optuna import Trial

from functions.function_generator import FunctionGenerator
from train_and_eval import train

def build_objective(vanilla_network: bool, data_generator: Callable):
    def objective(trial: Trial):
        n_train = 200
        lr_dml = trial.suggest_float('lr_dml', 0.001, 0.01, log=True)
        if not vanilla_network:
            lambda_ = 1  # trial.suggest_float('lambda_', 0.1, 10, log=True)
        else:
            lambda_ = 0
        n_layers = trial.suggest_int('n_layers', 1, 8)
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [8, 32, 128, 256])
        batch_size = 256  # trial.suggest_categorical('batch_size', [64, 128, 256])
        n_epochs = trial.suggest_categorical('epochs', [200, 500, 1000])
        test_loss = train(
            data_generator=data_generator,
            n_train=n_train,
            lr_dml=lr_dml,
            lambda_=lambda_,
            n_layers=n_layers,
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )
        return test_loss
    return objective


def main(vanilla_network: bool, dimensions: int):
    generator = FunctionGenerator(n_dim=dimensions)

    def f(n_data: int):
        return generator.generate_trigonometric_polynomial_data(
            n_samples=n_data,
            key=jax.random.PRNGKey(0),
            polynomial_degree=8,
            alpha=0.9,
            frequency=2,
        )

    vanilla_indicator = "vanilla" if vanilla_network else "dml"
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=f"small-training-set-higher-dimensions-{dimensions}-{vanilla_indicator}",
        direction="minimize",
        load_if_exists=True,
    )
    objective = build_objective(vanilla_network=vanilla_network, data_generator=f)
    study.optimize(objective, n_trials=50)

    study.best_params


if __name__ == "__main__":
    # start main two times in parallel with different arguments
    # one with vanilla_network=True and one with vanilla_network=False
    # Parallel start using threads
    t1 = threading.Thread(target=main, args=(True, 6))
    t2 = threading.Thread(target=main, args=(False, 6))
    t1.start()
    t2.start()