from example_problem import train

import optuna
from optuna import Trial

def objective(trial: Trial):
    n_train = 100
    lr_dml = trial.suggest_float('lr_dml', 0.00001, 0.1)
    lambda_ = 0  #trial.suggest_float('lambda_', 0.1, 10)
    n_layers = trial.suggest_int('n_layers', 1, 8)
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [8, 16, 32, 64, 128, 256])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64, 128])
    n_epochs = trial.suggest_categorical('epochs', [200, 500, 1000])
    test_loss = train(
        n_train=n_train,
        lr_dml=lr_dml,
        lambda_=lambda_,
        n_layers=n_layers,
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )
    return test_loss


study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="small-training-set-more-options-vanilla",
    direction="minimize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=1000)

study.best_params