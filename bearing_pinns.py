"""
@author: Maziar Raissi
"""

import sys

sys.path.insert(0, '../../Utilities/')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import deepxde as dde

np.random.seed(1234)
tf.set_random_seed(1234)

# 定义已知参数
m_in = 50
c_in = 1376.8
k_in = 7.42e7
m_out = 5
c_out = 2210.7
k_out = 1.51e7
g = 9.81
fr = 1797 / 60
e = 50e-6
N = 8
D_b = 6.746
D_p = 28.5
C_r = 2e-6
K = 8.753e9  # 需要定义K的值
omega = 2 * np.pi * fr
alpha = 0

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, t_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub

        self.t_f = t_f[:]

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # tf Graphs
        self.x_in, self.y_in, self.x_out, self.y_out = self.net(self.t_f_tf)
        self.dx_in_dt, self.dy_in_dt, self.dx_out_dt, self.dy_out_dt = self.net_dt(self.t_f_tf)
        self.d2x_in_dt2, self.d2y_in_dt2, self.d2x_out_dt2, self.d2y_out_dt2 = self.net_dt2(self.t_f_tf)
        self.F_HX, self.F_HY = self.get_F(self.t_f_tf, self.x_in, self.y_in, self.x_out, self.y_out)


        # Loss
        self.loss = tf.reduce_mean((-c_in * self.dx_in_dt - k_in * self.x_in - self.F_HX + e * m_in * omega ** 2 * tf.cos(omega * t)) / (m_in * self.d2x_in_dt2)) + \
                    tf.reduce_mean((-c_in * self.dy_in_dt - k_in * self.y_in + m_in * g - self.F_HY + e * m_in * omega ** 2 * tf.sin(omega * t)) / (m_in * self.d2y_in_dt2)) + \
                    tf.reduce_mean(self.F_HX - c_out * self.dx_out_dt - k_out * self.x_out) / (m_out * self.d2x_out_dt2) + \
                    tf.reduce_mean(self.F_HY + m_out * g - c_out * self.dy_out_dt - k_out * self.y_out) / (m_out * self.d2y_out_dt2)

        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        print(X.shape)
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        print(H)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net(self, t):
        xy = self.neural_net(t, self.weights, self.biases)

        dx_in = xy[:, 0:1]
        dy_in = xy[:, 1:2]
        dx_out = xy[:, 2:3]
        dy_out = xy[:, 3:4]
        print(dx_in.shape)

        return dx_in, dy_in, dx_out, dy_out

    def net_dt(self, t):

        dx_in, dy_in, dx_out, dy_out = self.net(t)

        dx_in_dt = tf.gradients(dx_in, t)[0]
        dy_in_dt = tf.gradients(dy_in, t)[0]
        dx_out_dt = tf.gradients(dx_out, t)[0]
        dy_out_dt = tf.gradients(dy_out, t)[0]

        return dx_in_dt, dy_in_dt, dx_out_dt, dy_out_dt

    def net_dt2(self, t):
        dx_in_dt, dy_in_dt, dx_out_dt, dy_out_dt = self.net_dt(t)

        dx_in_dt2 = tf.gradients(dx_in_dt, t)[0]
        dy_in_dt2 = tf.gradients(dy_in_dt, t)[0]

        dx_out_dt2 = tf.gradients(dx_out_dt, t)[0]
        dy_out_dt2 = tf.gradients(dy_out_dt, t)[0]

        return dx_in_dt2, dy_in_dt2, dx_out_dt2, dy_out_dt2

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter):

        tf_dict = {self.t_f_tf: self.t_f}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, t):

        tf_dict = {self.t_f_tf: t[:]}

        u_star = self.sess.run(self.x_in, tf_dict)
        v_star = self.sess.run(self.y_in, tf_dict)

        tf_dict = {self.t_f_tf: t[:]}

        f_u_star = self.sess.run(self.x_out, tf_dict)
        f_v_star = self.sess.run(self.y_out, tf_dict)

        return u_star, v_star, f_u_star, f_v_star

    def get_F(self, t, x_in, x_out, y_in, y_out):
        # 初始化 F_HX 和 F_HY
        F_HX = tf.zeros_like(x_in, dtype=tf.float32)
        F_HY = tf.zeros_like(y_in, dtype=tf.float32)
        # 计算 omega_c
        omega_c = (1 / 2) * (1 - (D_b / D_p) * np.cos(alpha)) * omega
        pi = 3.14159265358979323846
        # 计算求和
        for i in range(1, N + 1):
            tmp = tf.Variable(tf.zeros_like(F_HX, dtype=tf.float32))
            theta_i = 2 * pi * (i - 1) / N + omega_c * t
            delta_i = (x_in - x_out) * tf.sin(theta_i) + (y_in - y_out) * tf.cos(theta_i) - C_r
            # 获取 delta_i 大于 0 的索引
            condition = delta_i > 0
            # 将 delta_i 大于 0 的部分赋值给 tmp
            tmp = tf.where(condition, delta_i, tf.zeros_like(delta_i, dtype=tf.float32))
            # 更新 F_HX 和 F_HY
            F_HX += K * tmp ** 1.5 * tf.sin(theta_i)
            F_HY += K * tmp ** 1.5 * tf.cos(theta_i)
        return F_HX, F_HY


if __name__ == "__main__":
    noise = 0.0

    # Doman bounds
    lb = np.array([0.0])  # (t)
    ub = np.array([0.5])  # (t)

    N_f = 20000
    layers = [1, 100, 100, 100, 100, 4]

    data = scipy.io.loadmat('examples/dataset/97.mat')

    t = data['T'].flatten()[:]
    Exact = data['X097_DE_time'].T.flatten()[:]

    ###########################

    t_f = lb + (ub - lb) * lhs(1, N_f)

    model = PhysicsInformedNN(t_f, layers, lb, ub)

    start_time = time.time()
    model.train(50000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(t)

    error_u = np.linalg.norm(Exact - u_pred, 2) / np.linalg.norm(Exact, 2)

    print('Error u: %e' % (error_u))


    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[75] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[100] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[125] * np.ones((2, 1)), line, 'k--', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc='best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|h(t,x)|$', fontsize=10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact_h[:, 75], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.set_title('$t = %.2f$' % (t[75]), fontsize=10)
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact_h[:, 100], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact_h[:, 125], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[125, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[125]), fontsize=10)

    # savefig('./figures/NLS')

