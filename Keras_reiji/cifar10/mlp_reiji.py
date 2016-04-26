#coding: utf-8
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils

# バッチサイズ,クラス数,反復回数,データAugmentation
batch_size = 32
nb_classes = 10
nb_epoch = 50
data_augmentation = True

# 入力画像の次元数
img_rows, img_cols = 32, 32
# CIFAR10の画像チャネル数
img_channels = 3

# CIFAR10のデータをトレーニング用とテスト用にランダムにわける
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# クラスをベクトルに変換
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# MLP用に入力データを一次元に変換
X_train = X_train.reshape(50000, img_channels*img_cols*img_rows)
X_test = X_test.reshape(10000, img_channels*img_cols*img_rows)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(output_dim=500, input_shape=(img_channels*img_cols*img_rows,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(output_dim=200, input_dim=500, W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=nb_classes, input_dim=200, W_regularizer=l2(0.01)))
model.add(Activation('softmax'))

model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model.fit(X_train, Y_train,
		  batch_size=batch_size,
		  nb_epoch=nb_epoch,
		  validation_data=(X_test, Y_test),
		  shuffle=True)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy', score[1])

