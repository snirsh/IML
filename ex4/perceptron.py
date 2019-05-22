import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


class perceptron:

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.w = np.zeros(len(X))
        while True:
            w_t = self.w
            for i in range(len(y)):
                product = np.dot(y[i], np.inner(self.w, X[i]))
                if product < 1:
                    self.w = self.w + np.dot(y[i], X[i])
            if np.array_equal(w_t, self.w):
                return w_t

    def predict(self, x):
        return np.inner(x, self.w) >= 0

def err_rate(y,y_hat, s = 10000):
    indices = np.count_nonzero(y != y_hat)
    return indices/s


if __name__ == '__main__':
    svm_classifier = svm.SVC(C=1e10, kernel='linear')
    f = lambda x: np.sign(np.inner(np.array([0.3, -0.5]), x + 0.1))
    # cal_D = np.random.multivariate_normal(np.zeros(2), cov=np.identity(2))
    # X = f(cal_D)
    M = [x * 5 for x in range(1, 4)] + [25, 70]
    i = 0
    for m in M:
        plt.subplot(2, 3, i + 1, autoscale_on=True)
        i += 1
        X = np.random.multivariate_normal(np.zeros(2), cov=np.identity(2), size=m)
        Y = np.apply_along_axis(f, 1, X)
        x, y = X.T
        fig = plt.figure()
        ax = plt.axes()
        plt.scatter(x, y, c=Y)
        p = perceptron()
        p.fit(X, Y)
        true_pred = Y
        per_pred = p.predict(X)
    plt.show()
