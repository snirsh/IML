import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr
from numpy import exp

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
scaling_matrix = np.diag(np.array([.1, .5, 2]))
scaled_matrix = np.dot(scaling_matrix, x_y_z)
projection_matrix = np.diag(np.array([1, 1, 0]))
data = np.random.binomial(1, 0.25, (100000, 1000))
epsilons = [0.5, 0.25, 0.1, 0.01, 0.001]
# f = lambda d, m: np.apply_along_axis(np.mean, 0, d[:m])
cummean = lambda x: x.cumsum() / np.arange(1, len(x) + 1)
pct = lambda x, epsilon: np.true_divide(np.where(x >= epsilon)[0].shape[0], 100000)
means = np.apply_along_axis(cummean, 1, data)


def hoeffding(m, eps):
    return min(2 * exp(- 2 * m * pow(eps, 2)), 1)


def chevyshev(m, eps):
    return min(1 / (4 * m * pow(eps, 2)), 1)


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def q_23():
    plot_3d(x_y_z)
    # plt.show()
    plt.title('data representation')
    plt.savefig('q23.png')


def q_24():
    plot_3d(scaled_matrix)
    # plt.show()
    plt.title('Covariance after scaling')
    plt.savefig('q24.png')
    # the covariance matrix will look like S^2 now since the original was Id_3
    # scaled_covariance = scaled_matrix ** 2


def q_25():
    random_orthogonal_matrix = get_orthogonal_matrix(3)
    evd_decomposition = np.dot(random_orthogonal_matrix, scaled_matrix)
    plot_3d(evd_decomposition)
    # plt.show()
    plt.title('Covariance after random orthogonal matrix')
    plt.savefig('q25.png')
    # the covariance matrix looks like the EVD decomposition of S^2
    # random_orthogonal_matrix * scaled_matrix ** 2 * random_orthogonal_matrix.T


def q_26():
    projection_onto_xy = np.dot(projection_matrix, scaled_matrix)
    plot_2d(projection_onto_xy)
    # plt.show()
    plt.title('Projection onto xy axis')
    plt.savefig('q26.png')
    # it looks like the gaussian's mean


def q_27():
    ranged_pts = np.where((scaled_matrix[2, :] > -0.4) & (scaled_matrix[2, :] < 0.1))[0]
    points_within_range = np.take(scaled_matrix, ranged_pts, 1)
    plot_2d(np.dot(projection_matrix, points_within_range))
    plt.title('Projection onto xy axis for all z in (-0.4,0.1)')
    # plt.show()
    plt.savefig('q27.png')


def q_29_a():
    for i in range(5):
        plt.plot(means[i], label='X of row number ' + str(i + 1))
    plt.legend(bbox_to_anchor=(1, 1))
    # plt.show()
    plt.title('Cumulative mean')
    plt.xlabel('tosses')
    plt.ylabel('mean')
    plt.savefig('q29_a.png')
    plt.close()


def q_29_b():
    for i in range(5):
        plt.subplot(2, 3, i + 1, autoscale_on=True)
        hf_g = [hoeffding(m, epsilons[i]) for m in range(1, 1001)]
        ch_g = [chevyshev(m, epsilons[i]) for m in range(1, 1001)]
        plt.plot(hf_g, label='hoeffding bound')
        plt.plot(ch_g, label='chevyshev bound')
        plt.title('epsilon=' + str(epsilons[i]))
        plt.ylabel('upper bound')
        plt.xlabel('m tosses')
        plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.2, .75))
    # plt.show()
    plt.savefig('q29_b.png')
    plt.close()


def q_29_c():
    new_means = np.abs(means - 0.25)
    for i in range(5):
        plt.subplot(2, 3, i + 1, autoscale_on=True)
        hf_g = [hoeffding(m, epsilons[i]) for m in range(1, 1001)]
        ch_g = [chevyshev(m, epsilons[i]) for m in range(1, 1001)]
        pct_matrix = np.apply_along_axis(pct, 0, new_means, epsilons[i])
        plt.plot(hf_g, label='hoeffding bound')
        plt.plot(ch_g, label='chevyshev bound')
        plt.plot(pct_matrix, label='precentage')
        plt.ylabel('upper bound')
        plt.xlabel('m tosses')
        plt.title('epsilon=' + str(epsilons[i]))
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.legend(bbox_to_anchor=(1.2, .75))
    # plt.show()
    plt.savefig('q29_c.png')


if __name__ == '__main__':
    # q_23()
    # q_24()
    # q_25()
    # q_26()
    # q_27()
    # q_29_a()
    # q_29_b()
    q_29_c()
