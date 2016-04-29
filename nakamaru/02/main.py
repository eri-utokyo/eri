# coding:utf-8

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import (Activation,
                          Convolution2D,
                          Dense,
                          Flatten,
                          MaxPooling2D)


# load cifar10
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

train_x = train_x.astype('float32')
train_x /= 255.0
train_y = np_utils.to_categorical(train_y, 10)

test_x = test_x.astype('float32')
test_x /= 255.0
test_y = np_utils.to_categorical(test_y, 10)


# sequential model
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32)))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(10))
model.add(Activation('softmax'))


# compile
model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])


# learn & test
model.fit(train_x,
          train_y,
          batch_size=20,
          nb_epoch=2,
          validation_data=(test_x, test_y),
          shuffle=True)
