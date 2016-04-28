#!/user/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cupy as cp
import chainer
from chainer import cuda, Variable, optimizers
import chainer.functions as F
import chainer.links as L

from keras.datasets import cifar10


class MLP(chainer.Chain):
  def __init__(self):
    super(MLP, self).__init__(
        linear1 = L.Linear(3072, 1024),
        bn1= L.BatchNormalization(1024),
        linear2 = L.Linear(1024, 512),
        bn2 = L.BatchNormalization(512),
        linear3 = L.Linear(512, 256),
        bn3 = L.BatchNormalization(256),
        linear4 = L.Linear(256, 10),
        bn4 = L.BatchNormalization(10),
        )

  def clear(self):
    self.loss = None
    self.accuracy = None

  def __call__(self, x, t, is_train=True):
    self.clear()
    x = F.relu(self.bn1(self.linear1(x), test = not is_train))
    x = F.relu(self.bn2(self.linear2(x), test = not is_train))
    x = F.relu(self.bn3(self.linear3(x), test = not is_train))
    x = F.relu(self.bn4(self.linear4(x), test = not is_train))
    self.loss = F.softmax_cross_entropy(x, t)
    self.accuracy = F.accuracy(x, t)
    if is_train:
      return self.loss
    else:
      return x


BATCH_SIZE = 128
NUM_TRAIN = 50000
NUM_TEST = 10000
NUM_EPOCH = 50
NUM_MODELS = 100

(train_x, train_y), (test_x, test_y) = cifar10.load_data()
train_x = train_x.astype('float32').reshape((train_x.shape[0], 3072))
train_y = train_y.astype('int32').reshape((-1,))
test_x = test_x.astype('float32').reshape((test_x.shape[0], 3072))
test_y = test_y.astype('int32').reshape((-1,))
train_x /= 255
test_x /= 255

cuda.check_cuda_available()
all_results = None
for j in range(NUM_MODELS):
  model = MLP()
  model.to_gpu()

  optimizer = optimizers.Adam()
  optimizer.setup(model)

  for epoch in range(1, NUM_EPOCH + 1):
    perm = np.random.permutation(NUM_TRAIN)
    train_accuracy, train_loss = 0, 0
    for i in range(0, NUM_TRAIN, BATCH_SIZE):
      x = chainer.Variable(cp.asarray(train_x[perm[i:i + BATCH_SIZE]]), volatile='off')
      t = chainer.Variable(cp.asarray(train_y[perm[i:i + BATCH_SIZE]]), volatile='off')

      optimizer.update(model, x, t)

      train_loss += float(model.loss.data) * len(t.data)
      train_accuracy += float(model.accuracy.data) * len(t.data)

    epoch_result = None
    test_accuracy, test_loss = 0, 0
    for i in range(0, NUM_TEST, BATCH_SIZE):
      x = chainer.Variable(cp.asarray(test_x[i:i + BATCH_SIZE]), volatile='on')
      t = chainer.Variable(cp.asarray(test_y[i:i + BATCH_SIZE]), volatile='on')
      batch_result = model(x, t, False)
      if epoch == NUM_EPOCH:
        if i == 0:
          epoch_result = batch_result
        else:
          epoch_result.data = cp.vstack((epoch_result.data, batch_result.data))
      test_loss += float(model.loss.data) * len(t.data)
      test_accuracy += float(model.accuracy.data) * len(t.data)
    print 'epoch-{}-{}: [TRAIN] loss: {} accuracy: {}, [TEST] loss: {} accuracy: {}'.format(j+1, epoch,
        train_loss / NUM_TRAIN, train_accuracy / NUM_TRAIN, test_loss / NUM_TEST, test_accuracy / NUM_TEST)
    if epoch == NUM_EPOCH:
      if j == 0:
        all_results = epoch_result
      else:
        all_results += epoch_result
        y = chainer.Variable(cp.asarray(test_y), volatile='on')
        en_acc = F.accuracy(all_results, y)
        print 'ENSUMBLE RESULT:\n\t accuracy: {}'.format(en_acc.data)
