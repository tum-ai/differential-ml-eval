import json
import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_results_across_dimensions(function_name: str, n_datapoints: int):
    dimensions = [2, 8, 10, 20, 100]
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
    bar_width = 0.35
    x1 = np.arange(len(dimensions))
    x2 = x1 + bar_width

    ax.bar(
        x=x1,
        width=bar_width,
        height=[dml_mean[dimension] for dimension in dimensions],
        yerr=[dml_std[dimension] for dimension in dimensions],
        ecolor="orange",
        #log=True,
        label="DML",
    )
    ax.bar(
        x=x2,
        width=bar_width,
        height=[vanilla_mean[dimension] for dimension in dimensions],
        yerr=[vanilla_std[dimension] for dimension in dimensions],
        ecolor="blue",
        #log=True,
        label="Vanilla",
    )

    ax.set_ylabel("MSE")
    ax.set_xlabel("Dimension")
    ax.set_title("Comparison of DML and Vanilla MSEs")
    ax.set_xticks(x1 + bar_width / 2)
    ax.set_xticklabels(dimensions)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    visualize_results_across_dimensions(function_name="trigonometric-polynomial", n_datapoints=1024)
