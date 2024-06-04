"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
from scipy.io import loadmat
# Backend tensorflow.compat.v1 or tensorflow
# from deepxde.backend import tf
# Backend pytorch
import torch
# Backend paddle
# import paddle

# 定义已知参数
m_in = 50
c_in = 1376.8
k_in = 7.42e7
m_out = 5
c_out = 2210.7
k_out = 1.51e7
g = 9.81
fr = 1797 / 60
# e = 50e-6
N = 8
D_b = 6.746
D_p = 28.5
C_r = 2e-6
K = 8.753e9  # 需要定义K的值
omega = 2 * np.pi * fr
alpha = 0

e = dde.Variable(0.0)


def bearing_system(t, x):
    """Bearing system"""

    # 初始化
    x_in, x_out, y_in, y_out = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]

    # 一阶导数
    dx_in_dt = dde.grad.jacobian(x, t, i=0)
    dx_out_dt = dde.grad.jacobian(x, t, i=1)
    dy_in_dt = dde.grad.jacobian(x, t, i=2)
    dy_out_dt = dde.grad.jacobian(x, t, i=3)

    # 二阶导数
    # d2x_in_dt2 = dde.grad.hessian(x, t, component=0, i=0)
    # d2x_out_dt2 = dde.grad.hessian(x, t, component=1, i=0)
    # d2y_in_dt2 = dde.grad.hessian(x, t, component=2, i=0)
    # d2y_out_dt2 = dde.grad.hessian(x, t, component=3, i=0)
    d2x_in_dt2 = dde.grad.jacobian(dx_in_dt, t, i=0)
    d2x_out_dt2 = dde.grad.jacobian(dx_out_dt, t, i=0)
    d2y_in_dt2 = dde.grad.jacobian(dy_in_dt, t, i=0)
    d2y_out_dt2 = dde.grad.jacobian(dy_out_dt, t, i=0)

    # F_HX， F_HY
    F_HX, F_HY = get_F(t, x_in, x_out, y_in, y_out)
    # print(f"F_HX: {F_HX}")
    # print(f"F_HY: {F_HY}")

    # 方程
    s1 = -c_in * dx_in_dt - k_in * x_in - F_HX + e * m_in * omega ** 2 * torch.cos(omega * t) - m_in * d2x_in_dt2
    s2 = -c_in * dy_in_dt - k_in * y_in + m_in * g - F_HY + e * m_in * omega ** 2 * torch.sin(omega * t) - m_in * d2y_in_dt2
    s3 = F_HX - c_out * dx_out_dt - k_out * x_out - m_out * d2x_out_dt2
    s4 = F_HY + m_out * g - c_out * dy_out_dt - k_out * y_out - m_out * d2y_out_dt2
    return [s1, s2, s3, s4]



def get_F(t, x_in, x_out, y_in, y_out):
    # 初始化F_HX
    F_HX = torch.zeros_like(x_in)
    F_HY = torch.zeros_like(y_in)
    # 计算omega_c
    omega_c = (1 / 2) * (1 - (D_b / D_p) * np.cos(alpha)) * omega
    pi = 3.14159265358979323846
    # 计算求和
    for i in range(1, N + 1):
        tmp = torch.zeros_like(F_HX)
        theta_i = 2 * pi * (i - 1) / N + omega_c * t
        delta_i = (x_in - x_out) * torch.sin(theta_i) + (y_in - y_out) * torch.cos(theta_i) - C_r
        # print(f"theta_i = {theta_i}")
        # print(f"delta_i = {delta_i}")
        # print(f"theta_i：{theta_i.shape}")
        condition = delta_i > 0  # 获取delta_i大于0的索引
        tmp[condition] = delta_i[condition]
        # print("tmp: ", tmp.shape)
        # if delta_i > 0:
        F_HX += K * tmp ** 1.5 * torch.sin(theta_i)
        F_HY += K * tmp ** 1.5 * torch.cos(theta_i)
    return F_HX, F_HY


def func(x):
    pi = np.pi
    solution1 = (np.cos(2 * pi * 29.99 * x + 3.02) + np.cos(2 * pi * 59.98 * x + -1.92) +
                 np.cos(2 * pi * 24.99 * x + -0.12) + np.cos(2 * pi * 34.99 * x + 3.00) +
                 np.cos(2 * pi * 179.93 * x + 3.08))
    solution2 = (np.cos(2 * pi * 29.99 * x + -1.69) + np.cos(2 * pi * 179.93 * x + -1.65) +
                 np.cos(2 * pi * 59.98 * x + -0.36) + np.cos(2 * pi * 184.92 * x + 1.51) +
                 np.cos(2 * pi * 209.91 * x + -3.05))
    solution3 = (np.cos(2 * pi*0.00 * x + 0.00) + np.cos(2 * pi * 29.99 * x + -1.69) +
                 np.cos(2 * pi * 34.99 * x + -1.75) + np.cos(2 * pi * 24.99 * x + 1.37) +
                 np.cos(2 * pi * 39.98 * x + -1.71))
    solution4 = (np.cos(2 * pi * 29.99 * x + -0.12) + np.cos(2 * pi * 179.93 * x + 0.06) +
                 np.cos(2 * pi * 209.91 * x + -0.96) + np.cos(2 * pi * 199.92 * x + -1.96) +
                 np.cos(2 * pi * 214.91 * x + 2.69))
    return np.hstack((solution1, solution2, solution3, solution4))


def load_training_data():
    data = loadmat("examples/dataset/simulation_results.mat")

    x = data["y"]  # 6001 x 8 double

    t = data["t"]  # 6001 x 1 double
    return t, x


def boundary(_, on_initial):
    return on_initial


observe_t, ob_x = load_training_data()
# observe_y0 = dde.icbc.PointSetBC(observe_t, ob_x, component=0)


geom = dde.geometry.TimeDomain(0, 0.5)
ic1 = dde.icbc.IC(geom, lambda X: 1, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda X: 1, boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda X: -1, boundary, component=2)
ic4 = dde.icbc.IC(geom, lambda X: -1, boundary, component=3)

observe_y0 = dde.icbc.PointSetBC(observe_t, ob_x[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, ob_x[:, 2:3], component=1)
observe_y2 = dde.icbc.PointSetBC(observe_t, ob_x[:, 4:5], component=2)
observe_y3 = dde.icbc.PointSetBC(observe_t, ob_x[:, 6:7], component=3)



data = dde.data.PDE(
    geom,
    bearing_system,
    [ic1, ic2, ic3, ic4, observe_y0, observe_y1, observe_y2, observe_y3],
# [observe_y0, observe_y1, observe_y2, observe_y3],
    num_domain=400,
    num_boundary=2,
    # solution=func,
    # num_test=200,
    anchors=observe_t,
)

layer_size = [1] + [50] * 3 + [4]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
variable = dde.callbacks.VariableValue(e, period=1000, filename="variables.dat")

model.compile(
    "adam",
    lr=0.001,
    # metrics=["l2 relative error"],
    external_trainable_variables=e,
)


losshistory, train_state = model.train(iterations=50000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
