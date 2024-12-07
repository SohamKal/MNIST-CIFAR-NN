"""
TODO: Finish and submit your code for logistic regression, neural network, and hyperparameter search.

"""
import random
import itertools
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import namedtuple
from tqdm import tqdm


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.linear(x)


""" - Part 1 - """


def logistic_regression(device):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(
        root='./MNIST_dataset', train=True, download=True, transform=transform)

    train_size = 48000
    val_size = 12000
    train_set, _ = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = LogisticRegressionModel().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(),
                          lr=0.011, weight_decay=0.001)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    results = dict(
        model=model
    )

    return results


""" - Part 2 - """


class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes

        self.fc1 = nn.Linear(32 * 32 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = torch.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        if self.loss_type != "ce":
            output = F.softmax(output, dim=1)

        return output

    def get_loss(self, output, target):
        if self.loss_type in ['cross_entropy', 'ce']:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


""" - Part 3 - """


def tune_hyper_parameter(target_metric, device):
    learning_rates = [0.001, 0.0005]
    weight_decays = [1e-4]
    batch_sizes = [128, 200, 256]

    HyperParams = namedtuple(
        'HyperParams', ['learning_rate', 'batch_size', 'weight_decay'])

    best_logistic_params = None
    best_fnn_params = None
    best_logistic_metric = float('-inf')
    best_fnn_metric = float('-inf')

    def create_dataloaders(batch_size, dataset_name="MNIST"):
        if dataset_name == "MNIST":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.MNIST(
                root='./MNIST_dataset', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(
                root='./MNIST_dataset', train=False, download=True, transform=transform)
            train_size = 50000
            val_size = 10000
            train_set, val_set = random_split(
                train_dataset, [train_size, val_size])
            train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(
                val_set, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False)

        elif dataset_name == "CIFAR10":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.CIFAR10(
                root='./CIFAR10_dataset', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(
                root='./CIFAR10_dataset', train=False, download=True, transform=transform)
            # Using smaller training size to speed up tuning
            train_size = 45000
            val_size = 5000
            train_set, val_set = random_split(
                train_dataset, [train_size, val_size])
            train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(
                val_set, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return train_loader, val_loader, test_loader

    def train(model, optimizer, train_loader, device, epochs=5, patience=2):
        model.train()
        criterion = nn.CrossEntropyLoss()
        best_val_acc = -float('inf')
        no_improve_epochs = 0

        for epoch in range(epochs):
            running_loss = 0.0
            model.train()
            pbar = tqdm(train_loader, ncols=100, position=0, leave=True)
            for batch_idx, (data, target) in enumerate(pbar):
                optimizer.zero_grad()
                data, target = data.to(device), target.to(device)
                output = model(data)

                if hasattr(model, 'get_loss'):
                    loss = model.get_loss(output, target)
                else:
                    loss = criterion(output, target)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

            val_accuracy = validation(model, val_loader, device)
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

    def validation(model, validation_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total

    for lr, wd, batch_size in itertools.product(learning_rates, weight_decays, batch_sizes):
        print(f"Testing Logistic Regression with lr: {
              lr}, weight_decay: {wd}, batch_size: {batch_size}")
        train_loader, val_loader, _ = create_dataloaders(
            batch_size=batch_size, dataset_name="MNIST")
        model = LogisticRegressionModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        train(model, optimizer, train_loader,
              device, epochs=5)
        val_accuracy = validation(
            model, val_loader, device)
        if val_accuracy > best_logistic_metric:
            best_logistic_metric = val_accuracy
            best_logistic_params = HyperParams(lr, batch_size, wd)

    for lr, wd, batch_size in itertools.product(learning_rates, weight_decays, batch_sizes):
        print(f"Testing FNN with lr: {lr}, weight_decay: {
              wd}, batch_size: {batch_size}")
        train_loader, val_loader, _ = create_dataloaders(
            batch_size=batch_size, dataset_name="CIFAR10")
        model = FNN(loss_type='cross_entropy', num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        train(model, optimizer, train_loader,
              device, epochs=5)
        val_accuracy = validation(
            model, val_loader, device)
        if val_accuracy > best_fnn_metric:
            best_fnn_metric = val_accuracy
            best_fnn_params = HyperParams(lr, batch_size, wd)

    best_params = [
        {
            "logistic_regression": {
                "learning_rate": best_logistic_params.learning_rate,
                "batch_size": best_logistic_params.batch_size,
                "weight_decay": best_logistic_params.weight_decay
            }
        },
        {
            "FNN": {
                "learning_rate": best_fnn_params.learning_rate,
                "batch_size": best_fnn_params.batch_size,
                "weight_decay": best_fnn_params.weight_decay
            }
        }
    ]

    best_metric = [
        {
            "logistic_regression": {
                "accuracy": best_logistic_metric
            }
        },
        {
            "FNN": {
                "accuracy": best_fnn_metric
            }
        }
    ]

    return best_params, best_metric
