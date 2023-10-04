import json
import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_experiment_results(function_name: str, dimensions: int, n_datapoints: int):
    with open(f"results/{function_name}-{dimensions}dim-{n_datapoints}data/losses_dml.json", "r") as dml:
        dml_results = json.load(dml)

    with open(f"results/{function_name}-{dimensions}dim-{n_datapoints}data/losses_vanilla.json", "r") as vanilla:
        vanilla_results = json.load(vanilla)

    dml_results = np.asarray(list(dml_results.values()))
    vanilla_results = np.asarray(list(vanilla_results.values()))

    # Convert results to log scale
    dml_results = np.log(dml_results)
    vanilla_results = np.log(vanilla_results)

    dml_mean = np.mean(dml_results, axis=0)
    dml_std = np.std(dml_results, axis=0)
    vanilla_mean = np.mean(vanilla_results, axis=0)
    vanilla_std = np.std(vanilla_results, axis=0)

    fig, ax = plt.subplots()
    ax.bar(
        x=np.arange(len(dml_results)),
        height=dml_results,
        label="DML",
    )
    ax.bar(
        x=np.arange(len(vanilla_results)),
        height=vanilla_results,
        label="Vanilla",
    )
    ax.set_ylabel("MSE")
    ax.set_title("Comparison of DML and Vanilla MSEs")
    ax.legend()
    plt.show()


def visualize_results_across_dimensions(function_name: str, n_datapoints: int):
    dimensions = [2, 8, 10, 20]
    dml_results = {}
    vanilla_results = {}
    for dimension in dimensions:
        with open(f"results/{function_name}-{dimension}dim-{n_datapoints}data/losses_dml.json", "r") as dml:
            dml_results[dimension] = json.load(dml)

        with open(f"results/{function_name}-{dimension}dim-{n_datapoints}data/losses_vanilla.json", "r") as vanilla:
            vanilla_results[dimension] = json.load(vanilla)

    dml_results = {k: np.asarray(list(v.values())) for k, v in dml_results.items()}
    vanilla_results = {k: np.asarray(list(v.values())) for k, v in vanilla_results.items()}

    # dml_results = {k: np.log(v) for k, v in dml_results.items()}
    # vanilla_results = {k: np.log(v) for k, v in vanilla_results.items()}

    dml_mean = {k: np.mean(v, axis=0) for k, v in dml_results.items()}
    dml_std = {k: np.std(v, axis=0) for k, v in dml_results.items()}

    vanilla_mean = {k: np.mean(v, axis=0) for k, v in vanilla_results.items()}
    vanilla_std = {k: np.std(v, axis=0) for k, v in vanilla_results.items()}

    # Compare average DML and Vanilla MSEs next to each other across four dimensions
    fig, ax = plt.subplots()
    ax.bar(
        x=np.arange(len(dimensions)),
        height=[dml_mean[dimension] for dimension in dimensions],
        yerr=[dml_std[dimension] for dimension in dimensions],
        label="DML",
        zorder=2,
    )
    ax.bar(
        x=np.arange(len(dimensions)),
        height=[vanilla_mean[dimension] for dimension in dimensions],
        yerr=[vanilla_std[dimension] for dimension in dimensions],
        label="Vanilla",
    )

    ax.set_ylabel("MSE")
    ax.set_title("Comparison of DML and Vanilla MSEs")
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_xticklabels(dimensions)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    visualize_results_across_dimensions(function_name="trigonometric-polynomial", n_datapoints=512)
