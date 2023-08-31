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
### Installation of requirements

```bash
pip install -r requirements.txt
cd differential_ml && pip install -r requirements.txt && cd ..
```

Test if it works
```bash
python3 example_problem.py
```

## Start Optuna Dashboard
optuna-dashboard sqlite:///db.sqlite3