from __future__ import absolute_import
import numpy as np
import scipy as sp
from six.moves import range
from six.moves import zip

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy and ctc cost
    '''
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1

    if len(y.shape)==1:
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
    else:
        Y = np.zeros()


    return Y