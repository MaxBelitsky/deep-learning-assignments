  ################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the number of classes
    n_classes = predictions.shape[-1]

    # Get the prediction ids
    pred_ids = predictions.argmax(axis=1)

    # Create a confusion matrix
    conf_mat = torch.zeros((n_classes, n_classes))
    for pred, target in zip(pred_ids, targets):
        conf_mat[target.item(), pred.item()] += 1.0

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    metrics = {}
    metrics['accuracy'] = (torch.trace(confusion_matrix) / confusion_matrix.sum()).item()
    metrics['precision'] = (torch.diag(confusion_matrix) / torch.sum(confusion_matrix, axis=0)).detach().numpy()
    metrics['recall'] = (torch.diag(confusion_matrix) / torch.sum(confusion_matrix, axis=1)).detach().numpy()
    metrics['f1_beta'] = (1 + beta**2)*metrics['precision']*metrics['recall'] / (beta**2*(metrics['precision'] + metrics['recall']))

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10, device='cpu'):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Put model to device
    model.to(device)

    all_predictions, all_targets = [], []
    for data, targets in data_loader:
        # Put data to device
        data = data.to(device)
        targets = targets.to(device)

        # Collect predictions
        batch_size, n_inputs = data.shape[0], np.prod(data.shape[1:])
        predictions = model.forward(data.reshape(batch_size, n_inputs))
        all_predictions.append(predictions)
        all_targets.append(targets)

    all_targets = torch.concat(all_targets)
    all_predictions = torch.concat(all_predictions)

    conf_mat = confusion_matrix(all_predictions, all_targets)
    metrics = confusion_matrix_to_metrics(conf_mat)

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            device = torch.device('mps')
    except:
        device = torch.device('cpu')
    

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    n_classes = len(cifar10['train'].dataset.classes)
    n_inputs = np.prod(cifar10['train'].dataset.data.shape[1:])

    # Initialize model and loss module
    model = MLP(n_inputs, hidden_dims, n_classes, use_batch_norm).to(device)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)

    best_val_accuracy = 0
    best_model = None
    logging_info = {
        'train_losses': [],
        'train_accuracies': [],
        'validation_losses': [],
        'validation_accuracies': []
    }

    for epoch in tqdm(range(epochs)):
        epoch_losses = []
        # Set model to training mode
        model.train()

        for data, labels in cifar10_loader['train']:
            data = data.to(device)
            labels = labels.to(device)

            # Forwards pass
            logits = model(data)

            # Compute loss
            loss = loss_module(logits, labels.long())
            epoch_losses.append(loss.cpu().detach().numpy())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([loss_module(model(d.to(device)), l.to(device).long()).cpu().detach().numpy() for d, l in cifar10_loader['validation']])
            val_accuracy = evaluate_model(model, cifar10_loader['validation'], device=device)['accuracy']
            
            # deepcopy the best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = deepcopy(model)
        
        # Log info
        logging_info['train_losses'].append(np.mean(epoch_losses))
        logging_info['validation_losses'].append(val_loss)
        logging_info['validation_accuracies'].append(val_accuracy)

        print(f"Epoch: {epoch} | train loss: {logging_info['train_losses'][-1]} | val loss: {logging_info['validation_losses'][-1]} | val acc: {logging_info['validation_accuracies'][-1]}")
        
    # Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])['accuracy']

    val_accuracies = logging_info['validation_accuracies']

    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    