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
from keras.datasets import cifar10 # 手抜き

from res_net import ResNet
from utils import crop, zoom_and_crop, flip

logging.basicConfig(filename='cifar10_train.log', level=logging.DEBUG)

DEVICE1 = 2
cuda.get_device(DEVICE1).use()

BATCH_SIZE = 128
NUM_TRAIN = 50000
NUM_TEST = 10000
NUM_EPOCH = 300

(train_x, train_y), (test_x, test_y) = cifar10.load_data()
train_x = train_x.astype('float32')
train_y = train_y.astype('int32').reshape((-1,))
test_x = test_x.astype('float32')
test_y = test_y.astype('int32').reshape((-1,))
train_x /= 255
test_x /= 255


cuda.check_cuda_available()
model = ResNet()
model.to_gpu(DEVICE1)

optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(1, NUM_EPOCH + 1):
  print('epoch{} start'.format(epoch))
  perm = np.random.permutation(NUM_TRAIN)
  sum_accuracy, sum_loss = 0, 0
  start = time.time()
  for i in range(0, NUM_TRAIN, BATCH_SIZE):
    # x = chainer.Variable(cp.asarray(train_x[perm[i:i + BATCH_SIZE]]), volatile='off')
    # t = chainer.Variable(cp.asarray(train_y[perm[i:i + BATCH_SIZE]]), volatile='off')
    x = chainer.Variable(cuda.to_gpu(train_x[perm[i:i + BATCH_SIZE]], DEVICE1), volatile='off')
    t = chainer.Variable(cuda.to_gpu(train_y[perm[i:i + BATCH_SIZE]], DEVICE1), volatile='off')

    optimizer.update(model, x, t)

    sum_loss += float(model.loss.data) * len(t.data)
    sum_accuracy += float(model.accuracy.data) * len(t.data)
  end = time.time()
  elapsed_time = end - start
  throughput = NUM_TRAIN / elapsed_time
  logging.info('[TRAIN]epoch%d: loss=%.4f, accuracy=%.4f, throughput=%.4f', epoch, sum_loss/NUM_TRAIN, sum_accuracy/NUM_TRAIN, throughput)
  sum_accuracy = 0
  sum_loss = 0
  for i in range(0, NUM_TEST, BATCH_SIZE):
    # x = chainer.Variable(cp.asarray(test_x[i:i + BATCH_SIZE]), volatile='on')
    # t = chainer.Variable(cp.asarray(test_y[i:i + BATCH_SIZE]), volatile='on')
    x = chainer.Variable(cuda.to_gpu(test_x[i:i + BATCH_SIZE], DEVICE1), volatile='on')
    t = chainer.Variable(cuda.to_gpu(test_y[i:i + BATCH_SIZE], DEVICE1), volatile='on')
    loss = model(x, t, False)
    sum_loss += float(model.loss.data) * len(t.data)
    sum_accuracy += float(model.accuracy.data) * len(t.data)
  logging.info('[TEST]epoch%d: loss=%.4f, accuracy=%.4f', epoch, sum_loss/NUM_TRAIN, sum_accuracy/NUM_TRAIN)
