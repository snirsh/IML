import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions

from sklearn.svm import SVC

mean = np.zeros((2,))
cov = np.eye(2, 2)
w_true = np.array([0.3, -0.5, 0.1]).T
true_h = lambda x: np.sign(np.dot(x, np.array([0.3, -0.5])) + 0.1)
u0 = np.array([-3, 3])


def all():
    for m in [5, 10, 15, 25, 70]:

        X = np.random.multivariate_normal(mean, cov, m)
        X = np.hstack((X, np.ones((m, 1))))
        y = true_h(X)

        while (len(np.argwhere(y == 1)) == 0) or (len(np.argwhere(y == -1))) == 0:
            X = np.random.multivariate_normal(mean, cov, m)
            X = np.hstack((X, np.ones((m, 1))))
            y = true_h(X)

        svmclf = SVC(1e10, kernel='linear')
        svmclf.fit(X, y)

        w = svmclf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (svmclf.intercept_[0]) / w[1]

        # s = np.ones((X.shape[0], X.shape[1]+1))
        # s[:,1:] = X
        p = Perceptron(X, y)
        w2 = p.fit()

        b = -w2[0] / w2[1]
        d = w[1] / w[0]
        inter = w[1] - d * w[0]
        yy2 = b * xx - inter / w2[1]

        w3 = w_true
        q = -w3[0] / w3[1]
        f = w3[1] / w3[0]
        intercer = w3[1] - f * w3[0]
        yy3 = q * xx - (intercer) / w3[1]

        positive = X[np.nonzero(y > 0)]
        negative = X[np.nonzero(y < 0)]

        plt.figure()
        plt.plot()
        plt.plot(xx, yy, 'b')
        plt.plot(xx, yy2, 'r')
        plt.plot(xx, yy3, 'y')

        plt.scatter(positive[:, 0], positive[:, 1], c='b', label='pos')
        plt.scatter(negative[:, 0], negative[:, 1], c='r', label='neg')

        plt.legend()

        plt.savefig('Mean')
        plt.show()


def calculate_error_rate(y, y_hat):
    # indecis = np.argwhere(y == y_hat)
    print(y.shape, y_hat.shape)
    indecis = np.count_nonzero(y - y_hat == 0)
    error_rate = indecis / 10000
    return error_rate


svmclf = SVC(C=1e10, kernel='linear')


def all2():
    # error_svm = np.zeros((5, 500))
    # error_per = np.zeros((5, 500))
    error_svm = np.zeros((5, 1))
    error_per = np.zeros((5, 1))
    m_values = np.array([5, 10, 15, 25, 70])

    for midx, m in enumerate(m_values):

        for i in range(500):
            print(m)
            X = np.random.multivariate_normal(mean, cov, m)
            y = true_h(X)

            while (len(np.argwhere(y == 1)) == 0) or (len(np.argwhere(y == -1))) == 0:
                X = np.random.multivariate_normal(mean, cov, m)
                y = true_h(X)

            test_set_X = np.random.multivariate_normal(mean, cov, 10000)
            test_set_y = true_h(test_set_X)

            svmclf.fit(X, y)

            X = np.hstack((X, np.ones((m, 1))))

            perclf = Perceptron(X, y)

            y_hat_svm = svmclf.predict(test_set_X)
            print(test_set_X)
            test_set_X = np.hstack((test_set_X, np.ones((10000, 1))))
            print(test_set_X)
            y_hat_per = perclf.predict(test_set_X)

            error_svm[midx] += calculate_error_rate(test_set_y, y_hat_svm)
            error_per[midx] += calculate_error_rate(test_set_y, y_hat_per)

    # mean_error_svm = np.mean(error_svm, axis=1 )
    # mean_error_per = np.mean(error_per, axis=1 )
    mean_error_svm = error_svm / 500
    mean_error_per = error_per / 500

    plt.plot(m_values, mean_error_svm, label='svm')
    plt.plot(m_values, mean_error_per, label='perceptron')
    plt.title('performance comparision svm & perceptron ')
    plt.xlabel('m samples')
    plt.ylabel('true rate')
    plt.legend()
    plt.show()
    plt.savefig('q5')


all2()
