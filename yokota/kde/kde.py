#!/user/bin/env python
# -*- coding: utf-8 -*-

import math

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA


search_band_size = False

def kde(x, t, b=0.6):
    n, d = t.shape
    _x = (x - t) / b
    return np.sum(np.exp(_x[:, :, np.newaxis] * _x[:, :, np.newaxis] / -2) / (2 * np.pi ** (d / 2))) / (n * b ** d)

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data, mnist.target.astype('int32'))

mnist_X = mnist_X / 255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.1, random_state=44)
ids_x = np.array([train_y == i for i in range(10)])
idx_y = np.array([test_y == i for i in range(10)])

'''
dim = 150
pca = PCA(n_components=dim)
pca.fit(train_X)
train_X = pca.transform(train_X)

pca = PCA(n_components=dim)
pca.fit(test_X)
test_X = pca.transform(test_X)
'''

if search_band_size == True:
  for j in xrange(10):
      answers, bs = [], []
      print '----------- {} th --------------'.format(j)
      N = 50
      for n in xrange(N):
          ans, b = -100, -100
          for i in xrange(15, 100):
              b = float(i) / 100
              _t = test_X[idx_y[j]]
              if kde(_t[n], train_X[ids_x[j]].astype('float64')) > ans:
                  ans = kde(_t[n], train_X[ids_x[j]].astype('float64'), b)
                  best_b = b
          answers.append(ans)
          bs.append(best_b)
      print sum(answers) / N, sum(bs) / N
else:
  bs = [0.6 for _ in xrange(10)]
  ans = np.array([[kde(_x, train_X[ids_x[i]].astype('float64'), b) for i, b in zip(xrange(10), bs)] for _x in test_X])
  print accuracy_score(test_y, np.argmax(ans, axis=1))
