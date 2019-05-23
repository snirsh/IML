import numpy as np


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
        return np.dot(x, self.w) >= 0
