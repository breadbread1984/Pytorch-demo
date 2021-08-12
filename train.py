#!/usr/bin/python3

from absl import app, flags;
import numpy as np;
from torch.optim import Adam;
from torch.nn import CrossEntropyLoss;
from models import LeNet;
from create_dataset import load_dataset;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('epochs', 100, 'training epoch number');
flags.DEFINE_integer('batch_size', 32, 'batch size');
flags.DEFINE_integer('print_interval', 100, 'how many training steps for each console output');

def main(unused_argv):

  trainset, testset = load_dataset(FLAGS.batch_size);
  lenet = LeNet();
  optimizer = Adam(lenet.parameters(), lr = 1e-3);
  crossentropy = CrossEntropyLoss();
  for epoch in range(FLAGS.epochs):
    for batch_id, (images, labels) in enumerate(trainset):
      optimizer.zero_grad(); # zero gradients
      preds = lenet(images);
      loss = crossentropy(preds, labels);
      loss.backward(); # calculate gradients
      optimizer.step(); # apply gradients
      if batch_id % FLAGS.print_interval == 0:
        print('loss = %f' % loss);
    count = 0;
    correct_count = 0;
    for batch_id, (images, labels) in enumerate(testset):
      preds = lenet(images);
      idx = np.argmax(preds, axis = -1);
      correct_count += np.sum(idx == labels);
      count += FLAGS.batch_size;
    print('accuracy = %f' % (correct_count / count));

if __name__ == "__main__":

  app.run(main);

