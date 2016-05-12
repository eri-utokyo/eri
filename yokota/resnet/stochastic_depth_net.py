#!/user/bin/env python
# -*- coding: utf-8 -*-

import cupy as cp
import chainer
from chainer import cuda, Variable, optimizers
import chainer.functions as F
import chainer.links as L

from stochastic_depth_block import StochasticDepthBlock

import math
import re


class StochasticDepthNet(chainer.Chain):
  def __init__(self, n=30):
    super(StochasticDepthNet, self).__init__()
    self.L = 3 * n
    w = math.sqrt(2)
    self.links = [('conv1', L.Convolution2D(3, 16, 3, 1, 1, w))]
    self.links += [('bn1', L.BatchNormalization(16))]
    self.add_blocks(n, 16, 32)
    self.add_blocks(n, 32, 64)
    self.add_blocks(n, 64, 128)
    self.links += [('average_pool{}'.format(len(self.links)), F.AveragePooling2D(3, 1, 0, False, True))]
    self.links += [('linear{}'.format(len(self.links)), L.Linear(128, 10))]
    for link in self.links:
      if not link[0].startswith('average_pool'):
        self.add_link(*link)
    self.forward = self.links

  def add_blocks(self, n, n_in, n_out):
    for i in xrange(n):
      if i == n - 1:
        self.links += [('res{}__last'.format(len(self.links)),
                        StochasticDepthBlock(n_out if i > 0 else n_in, n_out))]
      else:
        self.links += [('res{}'.format(len(self.links)),
                        StochasticDepthBlock(n_out if i > 0 else n_in, n_out))]

  def clear(self):
    self.loss = None
    self.accuracy = None

  def __call__(self, x, t, is_train=True):
    self.clear()
    for name, f in self.forward:
      if 'res' in name:
        l = float(re.findall('[0-9]+' ,name)[0]) - 1.0
        x = f(x, float(len(self.links)), l, is_train)
        if '__last' in name:
          x = F.max_pooling_2d(x, 2, 2)
      else:
        x = f(x)
    self.loss = F.softmax_cross_entropy(x, t)
    self.accuracy = F.accuracy(x, t)
    return self.loss
