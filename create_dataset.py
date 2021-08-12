#!/usr/bin/python3

from torch.utils.data import DataLoader;
from torchvision import datasets;
from torchvision.transforms import ToTensor;

def load_dataset(batch_size = 32):
  mnist_train = datasets.MNIST(root = 'data', train = True, download = True, transform = ToTensor());
  mnist_test = datasets.MNIST(root = 'data', train = False, download = False, transform = ToTensor());
  mnist_trainset = DataLoader(mnist_train, batch_size = batch_size);
  mnist_testset = DataLoader(mnist_test, batch_size = batch_size);
  return mnist_trainset, mnist_testset;

