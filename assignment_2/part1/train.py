################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from tqdm.auto import tqdm

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except:
        pass


def set_device():
    """
    Function for setting the device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    try:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
    except:
        device = torch.device('cpu')
    return device


def save_model(model, model_path):
    """
    Function for saving the model.
    """
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, "model.tar")
    torch.save(model.state_dict(), model_file)


def load_model(model_path, num_classes=100):
    """
    Function for loading the model.
    """
    # Get the saved model file path
    model_file = os.path.join(model_path, "model.tar")
    if not os.path.isfile(model_file):
        raise Exception(f"File {model_file} doesn't exist")
    
    # Initialize a new model instance
    model = get_model(num_classes)

    # Load the weights
    model.load_state_dict(torch.load(model_file))
    
    return model


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all params
    model.requires_grad_(False)

    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Initialize the parameters
    model.fc.bias.data.fill_(0)
    model.fc.weight.data.normal_(0, 0.01)

    # Unfreeze the last layer
    model.fc.requires_grad_(True)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir)

    # Create dataloaders
    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Initialize a loss function
    loss_module = nn.CrossEntropyLoss()

    # Training loop with validation after each epoch. Save the best model.
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_accuracy = 0
    for epoch in range(epochs):
        epoch_loss, epoch_accuracy = 0, 0

        # Set model to training model
        model.train()

        true_predictions, count = 0, 0

        t = tqdm(train_dataloader, leave=False)
        for imgs, labels in t:
            # Transfer data to device
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            logits = model(imgs)

            # Compute loss
            loss = loss_module(logits, labels)
            epoch_loss += loss.cpu().detach().numpy()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters
            optimizer.step()

            count += labels.shape[0]
            true_predictions += (logits.argmax(dim=-1) == labels).float().mean()

        # Add train metrics
        train_accuracies.append(true_predictions / count)
        train_losses.append(epoch_loss / count)

        # Evaluate the model
        val_accuracy = evaluate_model(model, val_dataloader, device)
        val_accuracies.append(val_accuracy)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            print("\t   New best performance, saving model...")
            best_val_accuracy = val_accuracy
            save_model(model, checkpoint_name)
        
        # Report the epoch info
        print(f"Epoch: {epoch+1:2d} | train loss: {train_losses[-1]:.5f} | train acc: {train_accuracies[-1]:.2f} | val acc: {val_accuracy:.2f}")

    # Load the best model on val accuracy and return it.
    model = load_model(checkpoint_name).to(device)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    true_predictions, count = 0., 0
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(imgs)
            true_predictions += (logits.argmax(dim=-1) == labels).sum().item()
            count += labels.shape[0]
    accuracy = true_predictions / count

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise, checkpoint_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = set_device()

    # Load the model
    model = get_model().to(device)

    # Get the augmentation to use
    # TODO: do this
    # pass

    # Train the model
    model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name)

    # Evaluate the model on the test set
    test_dataset = get_test_set(data_dir, test_noise)
    test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)
    test_accuracy = evaluate_model(model, test_dataloader, device)

    print(f"Test accuracy: {test_accuracy:.2f}")

    return test_accuracy
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')
    parser.add_argument('--checkpoint_name', default='models/best_model', type=str,
                        help='The model checkpoint path.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
