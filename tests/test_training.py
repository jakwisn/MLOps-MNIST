# FILEPATH: /home/jakwisn/git/MLOps-MNIST/tests/test_train_model.py

import os
import torch
import pytest
from unittest import mock
from torch.utils.data import TensorDataset
from mlops.models.model import MyNeuralNet



def test_model_output_shape():
    # Create a random tensor of the right shape
    input_tensor = torch.randn(64, 1, 28, 28)

    # Create an instance of the model
    model = MyNeuralNet(1, 10)

    # Run the tensor through the model
    output_tensor = model(input_tensor)

    # Check if the output tensor has the expected shape
    assert output_tensor.shape == (64, 10), "Output shape is incorrect"