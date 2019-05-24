import numpy as np


class perceptron:

    def __init__(self):
        self.w = None

    def fit(self, X, y):
        self.w = np.zeros((X.shape[1], ))
        while True:
            w_t = self.w
            ywx = y*np.dot(X, self.w)
            neg_prod = np.argwhere(ywx <= 0)
            if len(neg_prod) != 0:
                self.w = self.w + y[neg_prod[0][0]] * X[neg_prod[0][0]]
            else:
                return w_t

    def predict(self, x):
        return np.sign(np.dot(x, self.w))
