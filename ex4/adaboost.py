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
        m = X.shape[0]
        D = np.zeros((m,)) + 1 / m
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            y_t = self.h[t].predict(X)
            e_t = np.sum(D[y != y_t])
            self.w[t] = 0.5 * np.log(np.true_divide(1, e_t) - 1)
            v = np.exp(-self.w[t] * y * y_t)
            g = D * v
            c = np.sum(g)
            D = g/c

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        h_predict = np.array([self.h[t].predict(X) for t in range(max_t)])
        return np.sign(np.dot(np.array(self.w)[:max_t], h_predict))

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        y_t = self.predict(X, max_t)
        return np.true_divide(np.count_nonzero(y != y_t), X.shape[0])


def q8():
    training_err = np.zeros((500,))
    test_err = np.zeros((500,))
    for t in range(1, 501):
        X, y = generate_data(5000, 0)
        h = AdaBoost(DecisionStump, t)
        h.train(X, y)
        training_err[t - 1] = h.error(X, y, t)
        test_set, labels = generate_data(200, 0)
        test_err[t - 1] = h.error(test_set, labels, t)
    plt.plot(range(500), training_err, label='Training error')
    plt.plot(range(500), test_err, label='Test error')
    plt.title('question 8')
    plt.xlabel('T')
    plt.ylabel('Error')
    plt.show()
    plt.savefig('q8')


def q9():
    T = [5, 10, 50, 100, 200, 500]
    X, y = generate_data(100, 0.01)
    i = 0
    for t in T:
        i += 1
        plt.subplot(3, 3, i, autoscale_on=True)
        h = AdaBoost(DecisionStump, t)
        h.train(X, y)
        decision_boundaries(h, X, y, t)
    plt.show()
    plt.savefig('q9')


if __name__ == '__main__':
    q8()
    # q9()
