import os.path
import pytest 
import torch
    
@pytest.mark.skipif(not os.path.exists('data/processed/train_images.pt'), reason="Data files not found")
def test_data():
    import numpy as np


    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    assert len(train_set) == 30000, "Dataset did not have the correct number of samples"
    assert len(test_set) == 5000, "Dataset did not have the correct number of samples" 

    assert  list(train_images.shape) == [len(train_images), 1, 28,28], "Train Images did not have the correct shape"
    assert  list(test_images.shape) == [len(test_images), 1, 28,28], "Test Images did not have the correct shape"


    assert len(train_target) == len(train_images), "Train Target did not have the correct shape"
    assert len(test_target) == len(test_images), "Test Target did not have the correct shape"

    assert sorted(list(np.unique(train_target.numpy()))) == [0,1,2,3,4,5,6,7,8,9], "Train Target did not have the correct shape"
    assert sorted(list(np.unique(test_target.numpy()))) == [0,1,2,3,4,5,6,7,8,9], "Test Target did not have the correct shape"


@pytest.mark.parametrize("test_input, expected", [
    ("data/processed/train_images.pt", [30000, 1, 28, 28]), 
    ("data/processed/train_target.pt", [30000]), 
    ("data/processed/test_images.pt", [5000, 1, 28, 28]), 
    ("data/processed/test_target.pt", [5000])])
def test_eval(test_input, expected):
    assert list(torch.load(test_input).shape) == expected, "The shape does not match"

