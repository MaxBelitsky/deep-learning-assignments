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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.modules = []
        units = [n_inputs, *n_hidden, n_classes]
        n_units = len(units)

        # Initialize linear layers
        for i in range(n_units-1):
           input_layer = True if i == 0 else False
           self.modules.append(LinearModule(units[i], units[i+1], input_layer=input_layer))
           activation_func = SoftMaxModule() if i == n_units-2 else ELUModule()
           self.modules.append(activation_func)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for module in self.modules:
           x = module.forward(x)

        out = x
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for i in range(len(self.modules)-1, -1, -1):
            dout = self.modules[i].backward(dout)
        
        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for module in self.modules:
           module.clear_cache()
        
        #######################
        # END OF YOUR CODE    #
        #######################
