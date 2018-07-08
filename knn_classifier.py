'''
    Nearest Neighbor Classifer
    Train O(1), Predict O(N)
    This is bad: we want fast prediction; slow for training is ok
'''

import numpy as np
import operator

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
    
    def predict(self, Xtest, k):
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
            
            dict = {}
            for x in range(distances):
                dict[distances[x]] = x
            
            distances.sort()
            k_neighbors =[]
            for j in range(k):
                k_neighbors.append(dict[distances[j]])
            
            voted = self.voteNdighbors(k_neighbors)
            if voted > 0:
                Ypred[i] = self.ytr[dict[voted]]
            else:
                Ypred[i] = -1

        return Ypred

    def voteNdighbors(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]