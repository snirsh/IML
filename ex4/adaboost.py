"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m,d = X.shape
        D = np.array([1/m] * m)
        for t in range(self.T):
            self.h[t] = self.WL(D,X,y)
            h_t = self.h[t].predict(X)
            epsilon_t = np.sum(np.logical_not(y==h_t) * D)
            self.w[t] = 0.5 * np.log((1/epsilon_t) - 1)
            n = np.sum([D * np.exp(-1 * self.w[t] * y * h_t)])
            D = (D * np.exp(-1 * self.w[t] * y * h_t)) / n

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        h_predict = [self.h[t].predict(X) for t in range(self.T)]
        return np.sign([ np.sum([h_predict[t][i] * self.w[t] for t in range(self.T)])  for i in range(len(X))])

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        ans = self.predict(X)
        return np.sum(np.logical_not(np.equal(ans,y))) / len(X)
