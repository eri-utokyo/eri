#!/user/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import numpy as np


mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data, mnist.target.astype('int32'))

mnist_X = mnist_X / 255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=44)
ids_x = np.array([train_y == i for i in range(10)])
means = None
variances = None
inv_vars = None
for ids in ids_x:
  x = train_X[ids]
  mean_x = np.sum(x, axis=0) / len(x)
  _x = np.array([f - mean_x for f in x])
  variance_x = np.array([np.dot(_x.T, _x) / len(_x)])
  if means == None:
    means = mean_x
    variances = variance_x
    variance_x = variance_x + np.identity(variance_x.shape[1]) * 0.001
    inv_vars = np.linalg.inv(variance_x)
  else:
    means = np.vstack([means, mean_x])
    variances = np.vstack([variances, variance_x])
    variance_x = variance_x + np.identity(variance_x.shape[1]) * 0.001
    inv_vars = np.vstack([inv_vars, np.linalg.inv(variance_x)])


def gaussian(x, mean, var, inv_var):
  d = len(mean)
  exp_val = np.dot(np.dot(x.T - mean.T, inv_var), x - mean)
  return exp_val


result = np.array([[gaussian(x, mean, var, inv_var) for mean, var, inv_var in zip(means, variances, inv_vars)] for x in test_X])

def accuracy(result, test_y):
  score = 0
  for y, t in zip(result, test_y):
    if y.argmin() == t:
      score += 1
  return score / len(test_y)

print accuracy(result, test_y)
