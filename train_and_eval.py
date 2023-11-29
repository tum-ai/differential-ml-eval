import os
import sys
from typing import Callable

import jax.random

from differential_ml.pt.modules.DmlDataLoader import SimpleDataLoader
from differential_ml.pt.modules.device import global_device
from functions.function_generator import FunctionGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), "differential_ml"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from differential_ml.pt.DmlTrainer import DmlTrainer
from differential_ml.pt.modules.DmlDataset import DmlDataset
from differential_ml.pt.modules.DmlFeedForward import DmlFeedForward
from differential_ml.pt.modules.DmlLoss import DmlLoss
from differential_ml.util.data_util import DataNormalizer

# Make pytorch debuggable
# torch.autograd.set_detect_anomaly(True)

def polynomial_and_trigonometric(n):
    # function y = 0.1*x³ + 0.2x² + 3*sin(3x)
    # derivative dydx = 3x² + 4x + cos(x)
    x = np.random.uniform(low=-10, high=10, size=n)
    y = 0.1 * (x ** 3) + 0.2 * (x ** 2) + 3 * np.sin(3 * x) + 10
    dydx = 0.3 * (x ** 2) + 0.4 * x + 9 * np.cos(3 * x)

    # plt.scatter(x, y)
    # plt.show()
    return x.reshape(-1, 1), y.reshape(-1, 1), dydx.reshape(-1, 1)


def train(
        data_generator: Callable,
        n_train: int = 1000,
        n_test: int = 1000,
        n_epochs: int = 100,
        batch_size: int = 32,
        lr_dml: float = 0.1,
        n_layers: int = 2,
        hidden_layer_sizes: int = 100,
        lambda_: float = 1,
        activation_identifier: str = 'sigmoid',
        plot_when_finished: bool = True,
        regularization_scale: float = 0.0,
        progress_bar_disabled: bool = False,
        seed = None
):
    if activation_identifier == 'sigmoid':
        activation = torch.nn.Sigmoid()
    elif activation_identifier == 'relu':
        activation = torch.nn.ReLU()
    else:
        raise ValueError(f"Activation identifier {activation_identifier} not recognized.")
    x_train, y_train, dydx_train = data_generator(n_train)
    x_test, y_test, dydx_test = data_generator(n_test)

    normalizer = DataNormalizer()

    normalizer.initialize_with_data(x_raw=x_train, y_raw=y_train, dydx_raw=dydx_train)
    x_train_normalized, y_train_normalized, dy_dx_train_normalized = normalizer.normalize_all(
        x_train,
        y_train,
        dydx_train,
    )
    x_test_normalized, y_test_normalized, dydx_test_normalized = normalizer.normalize_all(
        x_test,
        y_test,
        dydx_test,
    )
    dml_dataset = DmlDataset(x_train_normalized, y_train_normalized, dy_dx_train_normalized)
    test_set = DmlDataset(x_test_normalized, y_test_normalized, dydx_test_normalized)
    train_size = int(0.8 * len(dml_dataset))
    valid_size = len(dml_dataset) - train_size
    train_set, validation_set = torch.utils.data.random_split(dml_dataset, [train_size, valid_size])
    shuffle = True  # Set to True if you want to shuffle the data

    # Create a DataLoader using the custom dataset
    #dataloader_train = SimpleDataLoader(x_train_normalized, y_train_normalized, dy_dx_train_normalized, batch_size=batch_size, shuffle=shuffle)
    #dataloader_valid = SimpleDataLoader(x_train_normalized, y_train_normalized, dy_dx_train_normalized, batch_size=batch_size, shuffle=shuffle)
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    dataloader_valid = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


    # Network Architecture
    dml_net = DmlFeedForward(
        normalizer.input_dimension,
        normalizer.output_dimension,
        n_layers,
        hidden_layer_sizes,
        activation,
    )

    dml_loss = DmlLoss(
        _lambda=lambda_,  # Weight of differentials in Loss
        _input_dim=normalizer.input_dimension,
        _lambda_j=normalizer.lambda_j,
        regularization_scale=regularization_scale
    )
    dml_sgd = torch.optim.Adam(lr=lr_dml, params=dml_net.parameters())
    dml_trainer = DmlTrainer(dml_net, dml_loss, optimizer=dml_sgd)

    for epoch in tqdm(range(n_epochs), disable=progress_bar_disabled):
        pbar_train = tqdm(dataloader_train, disable=True)
        pbar_valid = tqdm(dataloader_valid, disable=True)
        # Train loop
        for batch in pbar_train:
            inputs = batch['x']  # Access input features
            targets = batch['y']  # Access target labels
            gradients = batch['dydx']  # Access dy/dx values
            step_dml = dml_trainer.step(
                inputs,
                targets,
                gradients,
            )
            pbar_train.set_description(f"Train Loss: {step_dml.loss.item()}")
        with torch.no_grad():
            valid_error_dml = 0
            for batch in pbar_valid:
                inputs = batch['x']  # Access input features
                targets = batch['y']  # Access target labels

                outputs_dml = dml_net(inputs)

                valid_error_dml += float(MSELoss()(outputs_dml, targets))
                pbar_train.set_description(f"Validation Loss: {valid_error_dml}")

    dml_net.eval()

    pbar_test = tqdm(dataloader_test, disable=True)
    test_error_dml = 0
    y_out_test = np.empty((0, 1))
    x_test_shuffled = np.empty((0, x_train.shape[1]))
    for batch in pbar_test:
        inputs = batch['x']  # Access input features
        targets = batch['y']  # Access target labels

        outputs_dml = dml_net(inputs)
        test_error_dml += float(MSELoss()(outputs_dml, targets))
        y_out_test = np.append(y_out_test, normalizer.unscale_y(outputs_dml.detach().cpu().numpy()), axis=0)
        x_test_shuffled = np.append(x_test_shuffled, normalizer.unscale_x(inputs.detach().cpu().numpy()), axis=0)
    if plot_when_finished:
        if x_train.shape[1] == 2:
            # 3d scatter
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_test_shuffled[:, 0], x_test_shuffled[:, 1], y_out_test)
            ax.scatter(x_test[:, 0], x_test[:, 1], y_test)
            plt.show()
        if x_train.shape[1] == 1:
            plt.scatter(x_test_shuffled[:, 0], y_out_test, s=0.1)
            plt.scatter(x_test[:, 0], y_test, s=0.1)
            plt.show()

    return test_error_dml

def train_only(
        x_train,
        y_train,
        dydx_train,
        x_test,
        y_test,
        dydx_test,
        n_epochs: int = 100,
        batch_size: int = 32,
        lr_dml: float = 0.1,
        n_layers: int = 2,
        hidden_layer_sizes: int = 100,
        lambda_: float = 1,
        shuffle: bool = True,
        activation: Callable = torch.nn.Softmax(-1),
):

    normalizer = DataNormalizer()

    normalizer.initialize_with_data(x_raw=x_train, y_raw=y_train, dydx_raw=dydx_train)
    x_train_normalized, y_train_normalized, dy_dx_train_normalized = normalizer.normalize_all(
        x_train,
        y_train,
        dydx_train,
    )
    x_test_normalized, y_test_normalized, dydx_test_normalized = normalizer.normalize_all(
        x_test,
        y_test,
        dydx_test,
    )
    dml_dataset = DmlDataset(x_train_normalized, y_train_normalized, dy_dx_train_normalized)
    test_set = DmlDataset(x_test_normalized, y_test_normalized, dydx_test_normalized)
    train_size = int(0.8 * len(dml_dataset))
    valid_size = len(dml_dataset) - train_size
    train_set, validation_set = torch.utils.data.random_split(dml_dataset, [train_size, valid_size])

    # Create a DataLoader using the custom dataset
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    dataloader_valid = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle)
    dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)


    # Network Architecture
    dml_net = DmlFeedForward(
        normalizer.input_dimension,
        normalizer.output_dimension,
        n_layers,
        hidden_layer_sizes,
        activation,
    )

    dml_loss = DmlLoss(
        _lambda=lambda_,  # Weight of differentials in Loss
        _input_dim=normalizer.input_dimension,
        _lambda_j=normalizer.lambda_j,
    )
    dml_sgd = torch.optim.Adam(lr=lr_dml, params=dml_net.parameters())
    dml_trainer = DmlTrainer(dml_net, dml_loss, optimizer=dml_sgd)

    for epoch in tqdm(range(n_epochs)):
        pbar_train = tqdm(dataloader_train, disable=True)
        pbar_valid = tqdm(dataloader_valid, disable=True)
        # Train loop
        for batch in pbar_train:
            inputs = batch['x']  # Access input features
            targets = batch['y']  # Access target labels
            gradients = batch['dydx']  # Access dy/dx values
            step_dml = dml_trainer.step(
                inputs,
                targets,
                gradients,
            )
            pbar_train.set_description(f"Train Loss: {step_dml.loss.item()}")
        with torch.no_grad():
            valid_error_dml = 0
            for batch in pbar_valid:
                inputs = batch['x']  # Access input features
                targets = batch['y']  # Access target labels

                outputs_dml = dml_net(inputs)

                valid_error_dml += float(MSELoss()(outputs_dml, targets))
                pbar_train.set_description(f"Validation Loss: {valid_error_dml}")

    dml_net.eval()

    pbar_test = tqdm(dataloader_test, disable=True)
    test_error_dml = 0
    y_out_test = np.empty((0, 1))
    x_test_shuffled = np.empty((0, x_train.shape[1]))
    y_test_shuffled = np.empty((0, 1))
    for batch in pbar_test:
        inputs = batch['x']  # Access input features
        targets = batch['y']  # Access target labels

        outputs_dml = dml_net(inputs)
        test_error_dml += float(MSELoss()(outputs_dml, targets))
        y_out_test = np.append(y_out_test, normalizer.unscale_y(outputs_dml.detach().numpy()), axis=0)
        x_test_shuffled = np.append(x_test_shuffled, normalizer.unscale_x(inputs.detach().numpy()), axis=0)
        y_test_shuffled = np.append(y_test_shuffled, normalizer.unscale_y(targets.detach().numpy()), axis=0)

    return x_test_shuffled, y_test_shuffled, y_out_test


