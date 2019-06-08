import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

mean = np.zeros((2,))
id_2 = np.identity(2)
theta = np.radians(45)
c, s = np.cos(theta), np.sin(theta)
rotation_matrix = np.array(((c, -s), (s, c)))


def warm_up():
    mu, sigma = 0, 1
    s = np.random.normal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='r')
    plt.show()


def q7_a():
    S1 = np.random.multivariate_normal((1, 1), id_2, 1000)
    S2 = np.random.multivariate_normal((-1, -1), id_2, 1000)
    return S1, S2


def q7_b():
    S1, S2 = q7_a()
    x1, D1 = S1.T
    x2, D2 = S2.T
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.scatter(x1, D1, label='N~((1,1), Id2)')
    ax.scatter(x2, D2, label='N~((-1,-1), Id2)')
    plt.legend()
    plt.show()


def q7_c():
    S1, S2 = q7_a()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    x11, x21 = S1.T
    x12, x22 = S2.T
    ax1.hist(x11, bins=100, label='x1 by Dist1', color='r')
    ax1.hist(x12, bins=100, label='x1 by Dist2', color='g')
    ax2.hist(x21, bins=100, label='x2 by Dist1', color='r')
    ax2.hist(x22, bins=100, label='x2 by Dist2', color='g')
    ax1.legend()
    ax2.legend()
    plt.show()


def q7_d():
    S1, S2 = q7_a()
    x1, D1 = S1.T
    x2, D2 = S2.T
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    r_s1 = np.dot(S1, rotation_matrix.T)
    r_s2 = np.dot(S2, rotation_matrix.T)
    r_x1, r_d1 = r_s1.T
    r_x2, r_d2 = r_s2.T
    ax.scatter(r_x1, r_d1, label='N~((1,1), Id2) rotated by 45deg')
    ax.scatter(r_x2, r_d2, label='N~((-1,-1), Id2) rotated by 45deg')
    plt.legend()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.hist(r_x1, bins=100, label='x1 by Dist1', color='r')
    ax1.hist(r_x2, bins=100, label='x1 by Dist2', color='g')
    ax2.hist(r_d1, bins=100, label='x2 by Dist1', color='r')
    ax2.hist(r_d2, bins=100, label='x2 by Dist2', color='g')
    ax1.legend()
    ax2.legend()
    plt.show()

def q8():
    pass

if __name__ == '__main__':
    # q7_b()
    # q7_c()
    # q7_d()
    q8()
