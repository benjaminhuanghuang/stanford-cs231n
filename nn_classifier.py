'''
    Nearest Neighbor
'''

import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        # set training data
        self.Xtr = X
        self.ytr = y
    
    def predict(self, Xtest):
        # each row in Xtest is an example to be predicted
        num_test = Xtest.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in xrange(num_test):
            # find the nearest taining image to the i'th test image
            # axis=1 means sum elements in row
            distances = np.sum(np.abs(self.Xtr - Xtest[i, :]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred
