# Differential Machine Learning Evaluation Framework

## Installing this repo
### Pulling the latest version of DML

```bash
git clone git@github.com:tum-ai/differential-ml-eval.git
cd differential-ml-eval
git submodule update --init --recursive
cd ..
```

More information about git submodules: https://devconnected.com/how-to-add-and-update-git-submodules/

### Setting up the virtual environment with Python 3.11.0

Assumes a working installation of [pyenv](https://github.com/pyenv/pyenv) and [poetry](https://github.com/python-poetry/poetry). Make sure to have poetry version 1.6.1 installed, since newer versions of poetry cause issues in the pyproject.toml file.

```bash
pyenv install 3.11.0
pyenv virtualenv 3.11.0 differential-ml-eval
pyenv local differential-ml-eval
```

### Installing dependencies

```bash
poetry install
cd differential_ml && poetry install
```

Test if it works
```bash
python3 example_problem.py
```

## Start Optuna Dashboard
```bash
cd experiments
optuna-dashboard sqlite:///db.sqlite3
```