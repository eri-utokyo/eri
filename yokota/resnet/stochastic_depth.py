#!/user/bin/env python
# -*- coding: utf-8 -*-

import cupy as cp
import chainer
from chainer import cuda, Variable, optimizers
import chainer.functions as F
import chainer.links as L

import math
import random


class StochasticDepth(chainer.Chain):
  def __init__(self, n_in, n_out, k=3, p=1, pl=1.0, pL=0.5):
    w = math.sqrt(2)
    self.pl = pl
    self.pL = pL
    super(StochasticDepth, self).__init__(
        # Convolution2d(in, out, ksize, stride, padding, init_weight)
        conv1 = L.Convolution2D(n_in, n_out, k, 1, p, w),
        bn1 = L.BatchNormalization(n_out),
        conv2 = L.Convolution2D(n_out, n_out, k, 1, p, w),
        bn2 = L.BatchNormalization(n_out),
        conv1x1 = L.Convolution2D(n_in, n_out, 1, 1, 0, w)
        )

  def __call__(self, x, L, l, is_train=True):
    h = F.relu(self.bn1(self.conv1(x), test = not is_train))
    h = F.relu(self.bn2(self.conv2(h), test = not is_train))
    # h = F.max_pooling_2d(h, 2, 2)
    # p = 1.0 if random.random() < self.pl else 0.0
    p = 1.0 - l / L * (1 - self.pL)
    if is_train:
      p = 1.0 if random.random() < p else 0.0
    if h.data.shape != x.data.shape:
      xp = chainer.cuda.get_array_module(x.data)
      n, c, hh, ww = x.data.shape
      pad_c = h.data.shape[1] - c
      x = self.conv1x1(x)
      # p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
      # p = chainer.Variable(p, volatile=not is_train)
      # x = F.concat((p, x))
      if x.data.shape[2:] != h.data.shape[2:]:
        x = F.average_pooling_2d(x, 1, 2)
    return F.relu(p * h + x)
