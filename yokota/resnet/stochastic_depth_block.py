#!/user/bin/env python
# -*- coding: utf-8 -*-

import cupy as cp
import chainer
from chainer import cuda, Variable, optimizers
import chainer.functions as F
import chainer.links as L

import math
import random


class StochasticDepthBlock(chainer.Chain):
  def __init__(self, n_in, n_out, k=3, p=1, pl=1.0, pL=0.5):
    self.pl = pl
    self.pL = pL
    w = math.sqrt(2)
    super(StochasticDepth, self).__init__(
        # Convolution2d(in, out, ksize, stride, padding, weight_scale)
        conv1 = L.Convolution2D(n_in, n_out, k, 1, p, w),
        bn1 = L.BatchNormalization(n_out),
        conv2 = L.Convolution2D(n_out, n_out, k, 1, p, w),
        bn2 = L.BatchNormalization(n_out),
        conv1x1 = L.Convolution2D(n_in, n_out, 1, 1, 0, w)
        )

  def __call__(self, x, L, l, is_train=True):
    h = F.relu(self.bn1(self.conv1(x), test = not is_train))
    h = F.relu(self.bn2(self.conv2(h), test = not is_train))
    P = 1.0 - l / L * (1 - self.pL)
    if is_train:
      P = 1.0 if random.random() < P else 0.0
    if h.data.shape != x.data.shape:
      xp = chainer.cuda.get_array_module(x.data)
      n, c, hh, ww = x.data.shape
      pad_c = h.data.shape[1] - c
      x = self.conv1x1(x)
      if x.data.shape[2:] != h.data.shape[2:]:
        x = F.average_pooling_2d(x, 1, 2)
    return F.relu(P * h + x)
