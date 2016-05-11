#!/user/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import cifar10


def crop(X, Y, crop_size, stride, is_test=False):
  x, y = None, None
  if is_test:
    pad = (X.shape[2] - crop_size) / 2
    x = X[:, :, pad:crop_size+pad, pad:crop_size+pad]
    y = Y
  else:
    for i in xrange(0, X.shape[2] - crop_size + 1, stride):
      for j in range(0, X.shape[3] - crop_size + 1, stride):
        if x is None:
          x = X[:, :, i:crop_size+i, j:crop_size+j]
          y = Y
        else:
          x = np.vstack((x, X[:, :, i:crop_size+i, j:crop_size+j] ))
          y = np.hstack((y, Y))
  return (x, y)

def zoom_and_crop(X, Y, crop_size, stride, output_size):
  x, y = None, None
  for i in range(0, X.shape[2] - crop_size + 1, stride):
    for j in range(0, X.shape[3] - crop_size + 1, stride):
      if x is None:
        x = X[:, :, i:crop_size+i, i:crop_size+i]
        y = Y
      else:
        x = np.vstack((x, X[:, :, i:crop_size+i, j:crop_size+j] ))
        y = np.hstack((y, Y))
  cut_size = (crop_size - output_size) / 2
  x = x[:, :, cut_size:-cut_size, cut_size:-cut_size]
  return (x, y)

def flip(X, Y):
  x = np.vstack((X, X[:, :, :, ::-1]))
  y = np.hstack((Y, Y))
  return (x, y)

if __name__ == '__main__':
  (train_x, train_y), (test_x, test_y) = cifar10.load_data()
  train_x = train_x.astype('float32')
  train_y = train_y.astype('int32').reshape((-1,))
  test_x = test_x.astype('float32')
  test_y = test_y.astype('int32').reshape((-1,))
  train_x /= 255
  test_x /= 255

  # [train] data augmentation
  crop_size, stride = 24, 4
  cropped_x, cropped_y = crop(train_x, train_y, crop_size, stride)
  print cropped_x.shape, cropped_y.shape
  crop_and_zoom_x, crop_and_zoom_y = zoom_and_crop(train_x, train_y, 28, 2, 24)
  print crop_and_zoom_x.shape, crop_and_zoom_y.shape
  train_x = np.vstack((cropped_x, crop_and_zoom_x))
  train_y = np.hstack((cropped_y, crop_and_zoom_y))
  print train_x.shape, train_y.shape
  train_x, train_y = flip(train_x, train_y)
  print train_x.shape, train_y.shape

  # [test] center cropped
  test_x, test_y = crop(test_x, test_y, crop_size, stride, True)

  np.save('data/train_x.npy', train_x)
  np.save('data/train_y.npy', train_y)
  np.save('data/test_x.npy', test_x)
  np.save('data/test_y.npy', test_y)
