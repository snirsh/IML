import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


class perceptron:

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        while True:
            w_t = self.w
            for i in range(len(y)):
                product = y[i] * np.dot(self.w, X[i])
                if product < 1:
                    self.w = self.w + y[i] * X[i]
            if np.array_equal(w_t, self.w):
                return w_t

    def predict(self, x):
        return np.inner(x, self.w) >= 0


def err_rate(y, y_hat, s=10000):
    indices = np.count_nonzero(y != y_hat)
    return indices / s


f = lambda x: np.sign(np.inner(np.array([0.3, -0.5]), x + 0.1))


def q5():
    svm_classifier = svm.SVC(C=1e10, kernel='linear')
    M = [x * 5 for x in range(1, 4)] + [25, 70]
    i = 0
    for m in M:
        # plt.subplot(3, 3, i + 1, autoscale_on=True)
        i += 1
        X = np.random.multivariate_normal(np.zeros((2,)), cov=np.identity(2), size=m)
        # X = np.hstack((X, np.ones((m, 1))))
        Y = f(X)
        while (len(np.argwhere(Y == 1)) == 0) or (len(np.argwhere(Y == -1))) == 0:
            X = np.random.multivariate_normal(np.zeros((2,)), cov=np.identity(2), size=m)
            X = np.hstack((X, np.ones((m, 1))))
            Y = f(X)

        xx = np.linspace(-5, 5)
        svm_classifier.fit(X, Y)
        w = svm_classifier.coef_[0]
        yy = (-w[0] / w[1]) * xx - (svm_classifier.intercept_[0]) / w[1]

        p = perceptron()
        w2 = p.fit(X, Y)
        b = -w2[0] / w2[1]
        d = w[1] / w[0]
        inter = w[1] - d * w[0]
        yy2 = b * xx - inter / w2[1]

        w3 = np.array([0.3, -0.5, 0.1])
        q1 = -w3[0] / w3[1]
        q2 = w3[1] / w3[0]
        inter2 = w3[1] - q2 * w3[0]
        yy3 = q1 * xx - (inter2) / w3[0]

        x, y = X.T
        # plt.figure()
        # plt.plot()
        plt.subplot(3, 2, i, autoscale_on=True)
        plt.scatter(x, y, c=Y)
        plt.plot(xx, yy, 'b', label='SVM')
        plt.plot(xx, yy2, 'r', label='Perceptron')
        plt.plot(xx, yy3, 'g', label='True')
        plt.title('Mean for M={0}'.format(m))
    plt.legend()
    plt.savefig('Mean')
    plt.show()


def err_rate(y, y_t):
    indices = np.count_nonzero(y - y_t)
    return indices / 10000


def q6():
    svm_classifier = svm.SVC(C=1e10, kernel='linear')
    svm_err = np.zeros((5, 1))
    perceptron_err = np.zeros((5, 1))
    M = np.array([5, 10, 15, 25, 70])
    for x, m in enumerate(M):
        for i in range(500):
            X = np.random.multivariate_normal(np.zeros((2,)), cov=np.identity(2), size=m)
            Y = f(X)
            while (len(np.argwhere(Y == 1)) == 0) or (len(np.argwhere(Y == -1))) == 0:
                X = np.random.multivariate_normal(np.zeros((2,)), cov=np.identity(2), size=m)
                X = np.hstack((X, np.ones((m, 1))))
                Y = f(X)

            X_test_set = np.random.multivariate_normal(np.zeros((2,)), cov=np.identity(2), size=10000)
            Y_test_set = f(X_test_set)

            svm_classifier.fit(X, Y)
            X = np.hstack((X, np.ones((m, 1))))

            p = perceptron()
            p.fit(X, Y)

            svm_y_t = svm_classifier.predict(X_test_set)
            X_test_set = np.hstack((X_test_set, np.ones(10000, 1)))
            p_y_t = p.predict(X_test_set)

            svm_err += err_rate(Y_test_set, svm_y_t)
            perceptron_err += err_rate(Y_test_set, p_y_t)

    mean_svm = svm_err / 500
    mean_per = perceptron_err / 500

    plt.plot(M, mean_svm, label='SVM mean')
    plt.plot(M, mean_per, label='per mean')
    plt.title('SVM vs Perceptron')
    plt.xlabel('samples')
    plt.ylabel('rate')
    plt.legend()
    plt.show()
    plt.savefig('q6')


if __name__ == '__main__':
    # q5()
    q6()
