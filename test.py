#!/usr/bin/python3

from os.path import join, exists;
from absl import app, flags;
import numpy as np;
from torch import load;
from create_dataset import load_dataset;

FLAGS = flags.FLAGS;
flags.DEFINE_integer('batch_size', 32, 'batch size');

def main(unused_argv):

  if not exists('models'):
    print('train the model first!');
    exit(1);
  trainset, testset = load_dataset(FLAGS.batch_size);
  with open(join('models', 'lenet.pkl'), 'rb') as f:
    lenet = load(f);
  count = 0;
  correct_count = 0;
  for batch_id, (images, labels) in enumerate(testset):
    preds = lenet(images);
    idx = np.argmax(preds.detach().numpy(), axis = -1);
    correct_count += np.sum(idx == labels.detach().numpy());
    count += FLAGS.batch_size;
  print('accuracy = %f' % (correct_count / count));

if __name__ == "__main__":

  app.run(main);

