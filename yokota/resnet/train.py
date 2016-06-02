#!/user/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import logging

import cupy as cp
from cupy import cuda
import chainer
from chainer import cuda, Variable, optimizers
import chainer.functions as F
import chainer.links as L

from stochastic_depth_net import StochasticDepthNet

date_format = "%Y-%m-%d_%H-%M-%S"
logging.basicConfig(
    filename='results/cifar10_train-{}.log'.format(time.strftime(date_format, time.localtime())),
    level=logging.DEBUG)

DEVICE1 = 0
cuda.get_device(DEVICE1).use()

train_x = np.load('data/train_x.npy')
train_y = np.load('data/train_y.npy')
test_x = np.load('data/test_x.npy')
test_y = np.load('data/test_y.npy')
print test_y

BATCH_SIZE = 128
NUM_TRAIN = train_x.shape[0]
MAX_TRAIN = NUM_TRAIN / 10
NUM_TEST = test_x.shape[0]
NUM_EPOCH = 300

cuda.check_cuda_available()
model = StochasticDepthNet()
model.to_gpu(DEVICE1)

optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(1, NUM_EPOCH + 1):
  print('epoch{} start'.format(epoch))
  perm = np.random.permutation(NUM_TRAIN)
  sum_accuracy, sum_loss = 0, 0
  start = time.time()
  for i in range(0, MAX_TRAIN, BATCH_SIZE):
    x = chainer.Variable(cuda.to_gpu(train_x[perm[i:i + BATCH_SIZE]], DEVICE1), volatile='off')
    t = chainer.Variable(cuda.to_gpu(train_y[perm[i:i + BATCH_SIZE]], DEVICE1), volatile='off')

    optimizer.update(model, x, t)

    sum_loss += float(model.loss.data) * len(t.data)
    sum_accuracy += float(model.accuracy.data) * len(t.data)
  end = time.time()
  elapsed_time = end - start
  throughput = NUM_TRAIN / elapsed_time
  logging.info('[TRAIN]epoch%d: loss=%.4f, accuracy=%.4f, throughput=%.4f', epoch, sum_loss/MAX_TRAIN, sum_accuracy/MAX_TRAIN, throughput)
  sum_accuracy = 0
  sum_loss = 0
  for i in range(0, NUM_TEST, BATCH_SIZE):
    x = chainer.Variable(cuda.to_gpu(test_x[i:i + BATCH_SIZE], DEVICE1), volatile='on')
    t = chainer.Variable(cuda.to_gpu(test_y[i:i + BATCH_SIZE], DEVICE1), volatile='on')
    loss = model(x, t, False)
    sum_loss += float(model.loss.data) * len(t.data)
    sum_accuracy += float(model.accuracy.data) * len(t.data)
  logging.info('[TEST]epoch%d: loss=%.4f, accuracy=%.4f', epoch, sum_loss/NUM_TEST, sum_accuracy/NUM_TEST)
