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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


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
    conf_mat = np.zeros((n_classes, n_classes))
    for pred, target in zip(pred_ids, targets):
        conf_mat[target, pred] += 1

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
        recall: 1D float array of size [n_classes], the recall for each class
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    metrics = {}
    metrics['accuracy'] = np.trace(confusion_matrix) / confusion_matrix.sum()
    metrics['precision'] = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    metrics['recall'] = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    metrics['f1_beta'] = (1 + beta**2)*metrics['precision']*metrics['recall'] / (beta**2*metrics['precision'] + metrics['recall'])

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    all_predictions, all_targets = [], []
    for data, targets in data_loader:
        batch_size, n_inputs = data.shape[0], np.prod(data.shape[1:])
        predictions = model.forward(data.reshape(batch_size, n_inputs))
        all_predictions.append(predictions)
        all_targets.append(targets)

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    conf_mat = confusion_matrix(all_predictions, all_targets)
    metrics = confusion_matrix_to_metrics(conf_mat)

    model.clear_cache()

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def optimization_step(model, lr):
    for module in model.modules:
        if not hasattr(module, 'params'):
            continue
        # Update weights
        module.params['weight'] += -lr*module.grads['weight']
        module.grads['weight'] = np.zeros_like(module.grads['weight'])

        # Update biases
        module.params['bias'] += -lr*module.grads['bias']
        module.grads['bias'] = np.zeros_like(module.grads['bias'])


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    n_classes = len(cifar10['train'].dataset.classes)
    n_inputs = np.prod(cifar10['train'].dataset.data.shape[1:])

    model = MLP(n_inputs, hidden_dims, n_classes)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation
    logging_dict = {
        'train_losses': [],
        'train_accuracies': [],
        'validation_losses': [],
        'validation_accuracies': []
    }
    best_model = None
    best_val_acc = 0

    for epoch in tqdm(range(epochs)):
        epoch_train_losses = []
        for data_inputs, data_labels in cifar10_loader['train']:
          # Clear cache
          model.clear_cache()

          # Forwards pass
          outputs = model.forward(data_inputs.reshape(batch_size, n_inputs))

          # Compute loss
          loss = loss_module.forward(outputs, data_labels)
          epoch_train_losses.append(loss)

          # Compute gradients of L wrt the output
          grad = loss_module.backward(outputs, data_labels)
          # Propagate gradients backwards
          model.backward(grad)

          # Make an optimization step
          optimization_step(model, lr)


        # Evaluate the model on validation data
        val_loss = np.mean([loss_module.forward(model.forward(d.reshape(d.shape[0], n_inputs)), l) for d, l in cifar10_loader['validation']])
        val_accuracy = evaluate_model(model, cifar10_loader['validation'])['accuracy']
        model.clear_cache()

        # deepcopy the best model
        if val_accuracy > best_val_acc:
            best_model = deepcopy(model)
            best_val_acc = val_accuracy

        # Log
        logging_dict['train_losses'].append(np.mean(epoch_train_losses))
        logging_dict['validation_losses'].append(val_loss)
        logging_dict['validation_accuracies'].append(val_accuracy)

        print(f"Epoch: {epoch} | train loss: {logging_dict['train_losses'][-1]} | val loss: {logging_dict['validation_losses'][-1]} | val acc: {logging_dict['validation_accuracies'][-1]}")

    val_accuracies = logging_dict['validation_accuracies']

    # Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])['accuracy']

    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
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
    