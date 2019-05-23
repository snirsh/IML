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
svmcls = SVC(C=1e10, kernel='linear')
M = [5, 10, 15, 25, 70]
T = [5, 10, 50, 100, 200, 500]
TEST_SIZE = 10000


def interval(w, u0):
    return -(w[2] + w[0] * u0) / w[1]


def err_rate(y, y_t):
    indices = np.count_nonzero(y - y_t == 0)
    return indices / 10000


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
            X = np.random.multivariate_normal(mean, cov, m)
            y = true_h(X)
            while (len(np.argwhere(y == 1)) == 0) or (len(np.argwhere(y == -1)) == 0):
                X = np.random.multivariate_normal(mean, cov, m)
                y = true_h(X)
            svmcls.fit(X, y)
            ones = np.ones((m, 1))
            test_set = np.random.multivariate_normal(mean, cov, TEST_SIZE)
            test_labels = true_h(test_set)
            X = np.hstack((X, ones))
            p = perceptron()
            p.fit(X, y)
            svms_y_hat = svmcls.predict(test_set)
            test_set = np.hstack((test_set, np.ones((TEST_SIZE, 1))))
            ps_y_hat = p.predict(test_set)
            errs1[k] += err_rate(test_labels, svms_y_hat)
            errs2[k] += err_rate(test_labels, ps_y_hat)
    svm_mean_err = errs1 / 500
    prs_mean_err = errs2 / 500
    plt.plot(M, prs_mean_err, label="Perceptron's error rate")
    plt.plot(M, svm_mean_err, label='SVM error rate')
    plt.title('Mean accuracy as function of m')
    plt.xlabel('samples(m)')
    plt.xlabel('Mean Accuracy')
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
        err[i-1] = h.error(X, y, t)
        decision_boundaries(h, X, y, t)
    plt.savefig('Q9')
    plt.show()
    return np.array(err)


def Q10():
    X, y = generate_data(1000, 0)
    T = [5, 10, 50, 100, 200, 500]
    i = np.argmin(Q9())
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
    'TODO complete this function'
    Q4()
    Q5()
    Q8()
    Q9()
    Q10()