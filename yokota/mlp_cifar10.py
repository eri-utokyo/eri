#!/user/bin/env python
# -*- coding: utf-8 -*-

from keras.datasets import cifar10
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization


BATCH_SIZE = 128
NUM_CLASS = 10
NUM_EPOCH = 100

(train_x, train_y), (test_x, test_y) = cifar10.load_data()
train_y = np_utils.to_categorical(train_y, NUM_CLASS)
test_y = np_utils.to_categorical(test_y, NUM_CLASS)

model = Sequential()
model.add(Dense(1024, input_shape=(3072,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(NUM_CLASS))
model.add(BatchNormalization())
model.add(Activation('softmax'))

adam = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

train_x = train_x.astype('float32').reshape([train_x.shape[0], 3072])
test_x = test_x.astype('float32').reshape([test_x.shape[0], 3072])
train_x /= 255
test_x /= 255

model.fit(train_x, train_y, batch_size=BATCH_SIZE,
          nb_epoch=NUM_EPOCH, validation_data=(test_x, test_y), shuffle=True)
