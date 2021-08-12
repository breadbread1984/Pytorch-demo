#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from absl import app, flags;
import numpy as np;
from torch.optim import Adam;
from torch.nn import CrossEntropyLoss, DataParallel;
from torch import device, save, no_grad;
from torch.cuda import device_count;
from models import LeNet;
from create_dataset import load_dataset;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('epochs', 100, 'training epoch number');
flags.DEFINE_integer('batch_size', 32, 'batch size');
flags.DEFINE_integer('print_interval', 100, 'how many training steps for each console output');
flags.DEFINE_integer('checkpoint_steps', 1000, 'how many training steps for a checkpoint');
flags.DEFINE_enum('device', default = 'cpu', enum_values = ['cpu', 'cuda'], help = 'device');

def main(unused_argv):

  location = device(FLAGS.device);
  global_batch_size = FLAGS.batch_size if FLAGS.device != 'cuda' \
                      else FLAGS.batch_size * device_count();
  trainset, testset = load_dataset(global_batch_size);
  lenet = LeNet();
  if FLAGS.device == 'cuda' and device_count() > 1:
    lenet = DataParallel(lenet);
  lenet.to(location);
  optimizer = Adam(lenet.parameters(), lr = 1e-3);
  crossentropy = CrossEntropyLoss();
  for epoch in range(FLAGS.epochs):
    lenet.train(); # model in training mode
    for batch_id, (images, labels) in enumerate(trainset):
      images, labels = images.to(location), labels.to(location); # move to device
      optimizer.zero_grad(); # zero gradients
      preds = lenet(images);
      loss = crossentropy(preds, labels);
      loss.backward(); # calculate gradients
      optimizer.step(); # apply gradients
      if batch_id % FLAGS.print_interval == 0:
        print('loss = %f' % loss);
      if batch_id % FLAGS.checkpoint_steps == 0:
        if not exists('models'): mkdir('models');
        save(lenet, join('models', 'lenet.pkl'));
    count = 0;
    correct_count = 0;
    with no_grad():
      lenet.eval(); # model in evaluating mode
      for batch_id, (images, labels) in enumerate(testset):
        images, labels = images.to(location), labels.to(location);
        preds = lenet(images);
        idx = np.argmax(preds.cpu().detach().numpy(), axis = -1);
        correct_count += np.sum(idx == labels.cpu().detach().numpy());
        count += FLAGS.batch_size;
    print('accuracy = %f' % (correct_count / count));
  # save model at the end
  if not exists('models'): mkdir('models');
  save(lenet, join('models', 'lenet.pkl'));

if __name__ == "__main__":

  app.run(main);

