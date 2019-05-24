"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
from ex4_tools import DecisionStump, decision_boundaries, generate_data, load_images
import matplotlib.pyplot as plt
from adaboost import AdaBoost
from face_detection import integral_image, WeakImageClassifier
from sklearn.svm import SVC
from perceptron import *

mean = np.zeros((2,))
cov = np.identity(2)
w_t = np.array([0.3, -0.5, 0.1]).T
true_h = lambda x: np.sign(np.dot(x, np.array([0.3, -0.5]) + 0.1))
u_0 = np.array([-3, 3])
M = [5, 10, 15, 25, 70]
T = [5, 10, 50, 100, 200, 500]
TEST_SIZE = 10000


def interval(w, u0):
    return -(w[2] + w[0] * u0) / w[1]


def get_accuracy(y, y_t):
    return np.count_nonzero(y - y_t == 0) / TEST_SIZE


def Q4():
    for m in M:
        X = np.random.multivariate_normal(mean, cov, m)
        x1, y1 = X.T
        X = np.hstack((X, np.ones((m, 1))))
        y = np.sign(np.dot(X, w_t))
        while (len(np.argwhere(y == 1)) == 0) or (len(np.argwhere(y == -1)) == 0):
            X = np.random.multivariate_normal(mean, cov, m)
            x1, y1 = X.T
            X = np.hstack((X, np.ones((m, 1))))
            y = np.sign(np.dot(X, w_t))

        svmcls = SVC(C=1e10, kernel='linear')
        p = perceptron()
        svmcls.fit(X, y)
        w_p = p.fit(X, y)
        plt.scatter(x1, y1, c=y)
        plt.plot(u_0, interval(w_t, u_0), label='true')
        plt.plot(u_0, interval(w_p, u_0), label='perceptron')
        plt.plot(u_0, interval(np.squeeze(svmcls.coef_), u_0), label='svm')
        plt.title('SVM and perceptron as fuction of m={}'.format(m))
        plt.legend()
        plt.savefig('m{}'.format(m))
        plt.show()


def Q5():
    errs1 = np.zeros((5, 1))
    errs2 = np.zeros((5, 1))
    for k, m in enumerate(M):
        for i in range(500):
            # ****************** GENERATING DATA ****************** #
            X = np.random.multivariate_normal(mean, cov, m)
            y = true_h(X)
            while (len(np.argwhere(y == 1)) == 0) or (len(np.argwhere(y == -1)) == 0):
                X = np.random.multivariate_normal(mean, cov, m)
                y = true_h(X)
            # ****************** CLASSIFIERS ****************** #
            # SVM
            svmcls = SVC(C=1e10, kernel='linear')
            svmcls.fit(X, y)
            # Perceptron
            p = perceptron()
            X = np.hstack((X, np.ones((m, 1))))
            p.fit(X, y)
            # ****************** GENERATING TEST ****************** #
            test_points = np.random.multivariate_normal(mean, cov, TEST_SIZE)
            test_labels = true_h(test_points)
            # ****************** MAKE PREDICTIONS ****************** #
            svms_y_hat = svmcls.predict(test_points)
            test_points = np.hstack((test_points, np.ones((TEST_SIZE, 1))))
            ps_y_hat = p.predict(test_points)
            # ****************** GET ACCURACY ****************** #
            errs1[k] += get_accuracy(test_labels, svms_y_hat)
            errs2[k] += get_accuracy(test_labels, ps_y_hat)
    plt.plot(M, np.true_divide(errs1, 500), label='SVM')
    plt.plot(M, np.true_divide(errs2, 500), label="Perceptron")
    plt.title('Mean accuracy as function of m')
    plt.xlabel('samples(m)')
    plt.ylabel('Mean Accuracy')
    plt.legend()
    plt.savefig('Q5')
    plt.show()


def Q8():
    training_err = np.zeros((500,))
    test_err = np.zeros((500,))
    X, y = generate_data(5000, 0)
    for t in range(1, 501):
        h = AdaBoost(DecisionStump, t)
        h.train(X, y)
        training_err[t - 1] = h.error(X, y, t)
        test_set, labels = generate_data(200, 0)
        test_err[t - 1] = h.error(test_set, labels, t)
    plt.plot(range(500), training_err, label='Training error')
    plt.plot(range(500), test_err, label='Test error')
    plt.title('question 8')
    plt.legend(loc='upper right')
    plt.xlabel('T')
    plt.ylabel('Error')
    plt.savefig('Q8')
    plt.show()


def Q9():
    err = [0] * len(T)
    X, y = generate_data(100, 0)
    i = 0
    for t in T:
        i += 1
        plt.subplot(3, 3, i, autoscale_on=True)
        h = AdaBoost(DecisionStump, t)
        h.train(X, y)
        err[i - 1] = h.error(X, y, t)
        decision_boundaries(h, X, y, t)
    plt.savefig('Q9')
    plt.show()
    return np.array(err)


def Q10():
    X, y = generate_data(1000, 0)
    T = [5, 10, 50, 100, 200, 500]
    i = int(np.argmin(Q9()))
    T_min = T[i]
    optimal_h = AdaBoost(DecisionStump, T_min)
    optimal_h.train(X, y)
    decision_boundaries(optimal_h, X, y)
    plt.savefig('Q10')
    plt.show()


def Q11():
    'TODO complete this function'


def Q12():
    'TODO complete this function'


def Q17():
    'TODO complete this function'


def Q18():
    'TODO complete this function'


if __name__ == '__main__':
    Q4()
    Q5()
    Q8()
    Q9()
    Q10()
