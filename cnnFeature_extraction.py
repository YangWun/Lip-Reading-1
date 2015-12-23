import numpy as np
np.random.seed(1337)  # for reproducibility

from datetime import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, AutoEncoder, ActivityRegularization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.layers import containers
from keras.regularizers import l2

# import visualization lib
# from keras.utils.visualize_util import plot
# import cv2 as cv
import theano
from read_data import read_Cdata as read_avletters

print 'Loading data...'

cha = 1
(X_train, Y_train), (X_test, Y_test) = read_avletters(test_split=0.1, data_split=0.01, channel=cha, shulffle=True)
print X_train.shape
print X_test.shape

# image size, number of classes, number of iterations
img_rows, img_cols = 60, 80
num_class = 26
num_epoch = 1
filter_size = 6
pool_size = 3
b_size = 32

print 'building model...'

model = Sequential()
# input: 60x80 images with 1 channels -> (1, 60, 80) tensors.
# this applies 32 convolution filters
model.add(Convolution2D(32, filter_size, filter_size, border_mode='valid', input_shape=(cha, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, filter_size, filter_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, filter_size, filter_size, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, filter_size, filter_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))  # model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class))
model.add(Activation('softmax'))


# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=opt)

model.fit(X_train, Y_train, batch_size=b_size, nb_epoch=num_epoch, show_accuracy=True, verbose=1, shuffle=True,
          validation_data=(X_train, Y_train))

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print 'writing log...'
log = open('unknown_predict_log.txt', 'a+')
log.write('%s \nUse CNN to classify the avletter database\n' % datetime.now())
log.write('training data:' + str(X_train.shape) + '\ttesting data:' + str(X_test.shape) + '\n')
log.write('filter size: %d\t pool size:%d batch size:%d\t epoch no.:%d\n' % (filter_size, pool_size, b_size, num_epoch))
log.write('Optimizer: ' + str(opt.get_config()) + '\n')
log.write('Test score:%f\t Test accuracy:%f\n\n' % (score[0], score[1]))
log.close()
print 'done'


## output feature vectors

get_feature = theano.function([model.layers[0].input], model.layers[15].get_output(train=False), allow_input_downcast=False)
feature = get_feature(X_test)
np.save('cnnFeatures', feature)

## visualize

