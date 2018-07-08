'''
    Nearest Neighbor Classifer
    Train O(1), Predict O(N)
    This is bad: we want fast prediction; slow for training is ok
'''

import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        '''
        The nearest neighbot classfier simply remember all the training data
        X is the training images, y is the labels
        X is N*D matrix, each row is an example
        '''
        self.Xtr = X
        self.ytr = y
    
    def predict(self, Xtest):
        '''
        Xtest is the test images
        each row in Xtest is an example to be predicted
        '''
        num_test = Xtest.shape[0]  # get the size of row (first dimension)
        # make sure the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows to find the cloest tain image to the i'th test image
        for i in xrange(num_test):
            # using L1 distance (sum of absolute value differences)
            # axis=1 means computing sum of each column
            distances = np.sum(np.abs(self.Xtr - Xtest[i, :]), axis=1)
            # get the index with smallest distance
            min_index = np.argmin(distances)   
            # predict the label of the nearest example
            Ypred[i] = self.ytr[min_index]

        return Ypred
