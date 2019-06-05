import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

mean = np.zeros((2,))
id_2 = np.identity(2)


def warmup():
    mu, sigma = 0, 1
    s = np.random.normal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='r')
    plt.show()


def q7_a():
    s1 = np.random.normal((1, 1), id_2)
    s2 = np.random.normal((-1, -1), id_2)
    count, bins, ignored = plt.hist(s1, 30, density=True)
    # count, bins, ignored = plt.hist(s2, 30, density=True)
    # plt.plot(s1)
    # plt.plot(s2)
    plt.show()


if __name__ == '__main__':
    warmup()
    q7_a()