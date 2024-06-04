"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
from scipy.io import loadmat
import torch

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


def load_training_data():
    data = loadmat("../dataset/97.mat")

    x = data["X097_DE_time"]  # 243938 x 1 double

    t = data["T"]
    return x, t


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
    d2x_in_dt2 = dde.grad.hessian(x, t, component=0, i=0)
    d2x_out_dt2 = dde.grad.hessian(x, t, component=1, i=0)
    d2y_in_dt2 = dde.grad.hessian(x, t, component=2, i=0)
    d2y_out_dt2 = dde.grad.hessian(x, t, component=3, i=0)

    # F_HX， F_HY
    F_HX, F_HY = get_F(t, x_in, x_out, y_in, y_out)
    # print(f"F_HX: {F_HX}")
    # print(f"F_HY: {F_HY}")

    # 方程
    s1 = (-c_in * dx_in_dt - k_in * x_in - F_HX + e * m_in * omega ** 2 * torch.cos(omega * t)) / (m_in * d2x_in_dt2)
    s2 = (-c_in * dy_in_dt - k_in * y_in + m_in * g - F_HY + e * m_in * omega ** 2 * torch.sin(omega * t)) / (
            m_in * d2y_in_dt2)
    s3 = (F_HX - c_out * dx_out_dt - k_out * x_out) / (m_out * d2x_out_dt2)
    s4 = (F_HY + m_out * g - c_out * dy_out_dt - k_out * y_out) / (m_out * d2y_out_dt2)
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

def boundary(_, on_initial):
    return on_initial

observe_t, ob_x = load_training_data()
observe_y0 = dde.icbc.PointSetBC(observe_t, ob_x, component=0)



geom = dde.geometry.TimeDomain(0, 0.5)
ic1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 0, boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda x: 0, boundary, component=2)
ic4 = dde.icbc.IC(geom, lambda x: 0, boundary, component=3)

data = dde.data.PDE(geom, bearing_system, [ic1, ic2, ic3, ic4, observe_y0], num_domain=35, num_boundary=2, anchors=observe_t)

layer_size = [1] + [50] * 3 + [4]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=200000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
