'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
	def f(input):
		conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=(1, 1), init="he_normal", border_mode="same")(input)
		norm = BatchNormalization(axis=1)(conv)
		return Activation("relu")(norm)
	return f

#Helper to build a BN -> relu -> conv block
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
	def f(input):
		norm = BatchNormalization(axis=1)(input)
		activation = Activation("relu")(norm)
		return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, init="he_normal", border_mode="same")(activation)
	return f

def _bottleneck(nb_filters, init_subsample=(1, 1)):
	def f(input):
		conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
		conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
		residual = _bn_relu_conv(nb_filters*4, 1, 1)(conv_3_3)
		return _shortcut(input, residual)
	return f

def _basic_block(nb_filters, init_subsample=(1, 1)):
	def f(input):
		conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
		residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
		return _shortcut(input, residual)
	return f

def _shortcut(input, residual):
	stride_width = input._keras_shape[2] / residual._keras_shape[2]
	stride_height = input._keras_shape[2] / residual._keras_shape[2]
	equal_channels = residual._keras_shape[1] == input._keras_shape[1]

	shortcut = input
	if stride_width > 1 or stride_height > 1 or not equal_channels:
		shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1, subsample=(stride_width, stride_height), init="he_normal", border_mode="valid")(input)
	return merge([shortcut, residual], mode="sum")

def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
	def f(input):
		for i in range(repetations):
			init_subsample = (1, 1)
			if i == 0 and not is_first_layer:
				init_subsample = (2, 2)
			input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
		return input
	return f

def resnet():
	input = Input(shape=(3, img_rows, img_cols))
	conv1 = _conv_bn_relu(nb_filter=64, nb_row=5, nb_col=5, subsample=(2, 2))(input)
	pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

	block_fn = _bottleneck
	block1 = _residual_block(block_fn, nb_filters = 64, repetations=3, is_first_layer=True)(pool1)
	#block2 = _residual_block(block_fn, nb_filters=128, repetations=4)(block1)
#	block3 = _residual_block(block_fn, nb_filters=256, repetations=6)(block2)
#	block4 = _residual_block(block_fn, nb_filters=512, repetations=3)(block3)
#	block4 = _residual_block(block_fn, nb_filters = 64, repetations=1, is_first_layer=True)(pool1)

	pool2 = AveragePooling2D(pool_size=(5, 5), strides=(1, 1), border_mode="same")(block1)
	flatten1 = Flatten()(pool2)
	dense = Dense(output_dim=10, init="he_normal", activation="softmax")(flatten1)

	model = Model(input=input, output=dense)
	return model


model = resnet()

model.summary()
# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))
