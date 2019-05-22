"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
from ex4_tools import *


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
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m, d = X.shape
        D = np.zeros((m,)) + 1/m
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            y_t = self.h[t].predict(X)
            e_t = np.sum(D[y != y_t])
            self.w[t] = 0.5 * np.log(np.true_divide(1, e_t) - 1)
            D = np.true_divide(D * np.exp(-1 * self.w[t] * y * y_t), np.sum([D * np.exp(-1 * self.w[t] * y * y_t)]))

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        h_predict = [self.h[t].predict(X) for t in range(max_t)]
        return np.sign([np.dot(h_predict[t], self.w[t]) for t in range(max_t)])

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        prediction = self.predict(X, max_t)
        return np.sum(np.logical_not(np.equal(prediction, y))) / len(X)


def q8():
    X, y = generate_data(5000, 0)
    training_err = np.zeros((500,))
    test_err = np.zeros((500,))
    for t in range(1, 500):
        h = AdaBoost(DecisionStump, 500)
        h.train(X, y)
        test_set, labels = generate_data(200, 0)
        training_err[t] = h.error(X, y, t)
        test_err[t] = h.error(test_set, labels, t)
    x = np.arange(1, 500)
    plt.plot(x, training_err)
    plt.plot(x, test_err)
    plt.show()


if __name__ == '__main__':
    q8()
