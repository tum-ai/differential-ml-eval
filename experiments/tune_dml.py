from example_problem import train

import optuna
from optuna import Trial

def objective(trial: Trial):
    n_train = 10000
    lr_dml = trial.suggest_float('lr_dml', 0.0001, 0.1)
    lambda_ = trial.suggest_float('lambda_', 0.1, 5)
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [16, 32, 128, 512])
    test_loss = train(
        n_train=n_train,
        lr_dml=lr_dml,
        lambda_=lambda_,
        n_layers=n_layers,
        hidden_layer_sizes=hidden_layer_sizes,
    )
    return test_loss


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="stupid-parameter-tuning-2",
    direction="minimize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}
