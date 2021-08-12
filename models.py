#!/usr/bin/python3

import torch;
from torch import nn;

class LeNet(nn.Module):
  def __init__(self, **kwargs):
    super(LeNet, self).__init__(**kwargs);
    self.conv1 = nn.Conv2d(1, 20, (5,5), padding = 0);
    self.pool1 = nn.MaxPool2d((2,2), (2,2), padding = 0);
    self.conv2 = nn.Conv2d(20, 50, (5,5), padding = 0);
    self.pool2 = nn.MaxPool2d((2,2), (2,2), padding = 0);
    self.flatten = nn.Flatten();
    self.dense1 = nn.Linear(800, 500);
    self.relu = nn.ReLU();
    self.dense2 = nn.Linear(500, 10);
  def forward(self, inputs):
    results = self.conv1(inputs);
    results = self.pool1(results);
    results = self.conv2(results);
    results = self.pool2(results);
    results = self.flatten(results);
    results = self.dense1(results);
    results = self.relu(results);
    results = self.dense2(results);
    return results;

if __name__ == "__main__":

  lenet = LeNet();
  inputs = torch.randn(1,1,28,28);
  print(lenet(inputs).shape);

