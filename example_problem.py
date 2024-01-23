import os
import sys
from typing import Callable

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
        n_train: int = 32,
        n_test: int = 1000,
        n_epochs: int = 10000,
        batch_size: int = 4096,
        lr_dml: float = 0.00001,
        n_layers: int = 1,
        hidden_layer_sizes: int = 1024,
        lambda_: float = 1,
        activation: Callable = torch.nn.Sigmoid(),
):
    x_train, y_train, dydx_train = polynomial_and_trigonometric(n_train)
    x_test, y_test, dydx_test = polynomial_and_trigonometric(n_test)

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

    losses = {
        'training_loss_y': [],
        'training_loss_dydx': [],
        'validation_loss_y': [],
        'validation_loss_dydx': []
    }
    for epoch in tqdm(range(n_epochs)):
        pbar_train = tqdm(dataloader_train, disable=True)
        pbar_valid = tqdm(dataloader_valid, disable=True)
        train_error = 0
        train_error_dml = 0
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
            train_error += float(MSELoss()(step_dml.y_out, targets))
            train_error_dml += float(MSELoss()(step_dml.greek_out, gradients))
        with torch.no_grad():
            valid_error = 0
            valid_error_dml = 0
            for batch in pbar_valid:
                inputs = batch['x']  # Access input features
                targets = batch['y']  # Access target labels
                targets_gradients = batch['dydx']  # Access dy/dx values
                outputs_dml, outputs_dml_greeks = dml_net.forward_with_greek(inputs)
                valid_error += float(MSELoss()(outputs_dml, targets))
                valid_error_dml += float(MSELoss()(outputs_dml_greeks, targets_gradients))
                pbar_train.set_description(f"Validation Loss: {valid_error}")
                pbar_train.set_description(f"Validation Loss Differentials: {valid_error_dml}")
        losses['training_loss_y'].append(train_error)
        losses['training_loss_dydx'].append(train_error_dml)
        losses['validation_loss_y'].append(valid_error)
        losses['validation_loss_dydx'].append(valid_error_dml)
    # Plotting the losses
    plt.plot(np.log(losses['training_loss_y']), label='training_loss_y')
    plt.plot(np.log(losses['training_loss_dydx']), label='training_loss_dydx')
    plt.plot(np.log(losses['validation_loss_y']), label='validation_loss_y')
    plt.plot(np.log(losses['validation_loss_dydx']), label='validation_loss_dydx')
    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses')
    # Adding legend
    plt.legend()
    # Saving the figure
    plt.savefig('loss_figure.png')
    plt.show()
    dml_net.eval()

    pbar_test = tqdm(dataloader_test, disable=True)
    test_error_dml = 0
    y_out_test = np.empty((0, 1))
    targets_test = np.empty((0, 1))
    x_test_shuffled = np.empty((0, 1))
    dydx_test_shuffled = np.empty((0, 1))
    dydx_out_shuffled = np.empty((0, 1))
    for batch in pbar_test:
        inputs = batch['x']  # Access input features
        targets = batch['y']  # Access target labels
        differentials = batch['dydx']

        outputs_dml, output_greeks = dml_net.forward_with_greek(inputs)
        test_error_dml += float(MSELoss()(outputs_dml, targets))
        y_out_test = np.append(y_out_test, normalizer.unscale_y(outputs_dml.detach().numpy()), axis=0)
        targets_test = np.append(targets_test, normalizer.unscale_y(targets.detach().numpy()), axis=0)
        x_test_shuffled = np.append(x_test_shuffled, normalizer.unscale_x(inputs.detach().numpy()), axis=0)
        dydx_test_shuffled = np.append(dydx_test_shuffled, normalizer.unscale_dy_dx(differentials.detach().reshape(-1, 1).numpy()), axis=0)
        dydx_out_shuffled = np.append(dydx_out_shuffled, normalizer.unscale_dy_dx(output_greeks.detach().reshape(-1, 1).numpy()), axis=0)
    print('Test Error DML:', test_error_dml)
    plt.scatter(x_test_shuffled, dydx_test_shuffled, s=.1)
    plt.scatter(x_test_shuffled, dydx_out_shuffled, s=.1)
    #plt.scatter(x_test_shuffled, y_out_test)
    #plt.scatter(x_test_shuffled, targets_test)
    plt.savefig("dydx_preds")
    plt.show()
    return test_error_dml


# print(train())
train()
