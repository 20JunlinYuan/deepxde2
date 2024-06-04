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
pi = np.pi


def load_training_data():
    data = loadmat("examples/dataset/97.mat")

    x = data["X097_DE_time"]  # 243938 x 1 double

    t = data["T"]
    return x, t


def get_F(t, x_in, x_out, y_in, y_out):
    # 初始化F_HX
    F_HX = torch.zeros_like(x_in)
    F_HY = torch.zeros_like(y_in)
    # 计算omega_c
    omega_c = (1 / 2) * (1 - (D_b / D_p) * np.cos(alpha)) * omega
    # pi = 3.14159265358979323846
    # 计算求和
    for i in range(1, N + 1):
        tmp = torch.zeros_like(F_HX)
        theta_i = 2 * pi * (i - 1) / N + omega_c * t
        delta_i = (x_in - x_out) * torch.sin(theta_i) + (y_in - y_out) * torch.cos(theta_i) - C_r

        condition = delta_i > 0  # 获取delta_i大于0的索引
        tmp[condition] = delta_i[condition]

        F_HX += K * tmp ** 1.5 * torch.sin(theta_i)
        F_HY += K * tmp ** 1.5 * torch.cos(theta_i)
    return F_HX, F_HY


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
    s1 = (-c_in * dx_in_dt - k_in * x_in - F_HX + e * m_in * omega ** 2 * torch.cos(omega * t) - m_in * d2x_in_dt2)
    s2 = (-c_in * dy_in_dt - k_in * y_in + m_in * g - F_HY + e * m_in * omega ** 2 * torch.sin(
        omega * t) - m_in * d2y_in_dt2)
    s3 = (F_HX - c_out * dx_out_dt - k_out * x_out - m_out * d2x_out_dt2)
    s4 = (F_HY + m_out * g - c_out * dy_out_dt - k_out * y_out - m_out * d2y_out_dt2)
    return [s1, s2, s3, s4]


# 解析解
# def func(x):
#     """
#     y1 = sin(x)
#     y2 = cos(x)
#     """
#     pi = np.pi
#     solution1 = (2 * np.cos(2 * pi * 29.99 * x + 3.02) + 2 * np.cos(2 * pi * 59.98 * x + -1.92) +
#                  2 * np.cos(2 * pi * 24.99 * x + -0.12) + 2 * np.cos(2 * pi * 34.99 * x + 3.00) +
#                  2 * np.cos(2 * pi * 179.93 * x + 3.08))
#     solution2 = (2 * np.cos(2 * pi * 29.99 * x + -1.69) + 2 * np.cos(2 * pi * 179.93 * x + -1.65) +
#                  2 * np.cos(2 * pi * 59.98 * x + -0.36) + 2 * np.cos(2 * pi * 184.92 * x + 1.51) +
#                  2 * np.cos(2 * pi * 209.91 * x + -3.05))
#     solution3 = (2 * np.cos(2 * pi * 0.00 * x + 0.00) + 2 * np.cos(2 * pi * 29.99 * x + -1.69) +
#                  2 * np.cos(2 * pi * 34.99 * x + -1.75) + 2 * np.cos(2 * pi * 24.99 * x + 1.37) +
#                  2 * np.cos(2 * pi * 39.98 * x + -1.71))
#     solution4 = (2 * np.cos(2 * pi * 29.99 * x + -0.12) + 2 * np.cos(2 * pi * 179.93 * x + 0.06) +
#                  2 * np.cos(2 * pi * 209.91 * x + -0.96) + 2 * np.cos(2 * pi * 199.92 * x + -1.96) +
#                  2 * np.cos(2 * pi * 214.91 * x + 2.69))
#     return np.hstack((solution1, solution2, solution3, solution4))

def x_in_func(t):
    return (
            2 * np.cos(2 * pi * 29.99 * t + 3.02) + 2 * np.cos(2 * pi * 59.98 * t + -1.92) +
            2 * np.cos(2 * pi * 24.99 * t + -0.12) + 2 * np.cos(2 * pi * 34.99 * t + 3.00) +
            2 * np.cos(2 * pi * 179.93 * t + 3.08)
    )


def y_in_func(t):
    return (
            2 * np.cos(2 * pi * 29.99 * t + -1.69) + 2 * np.cos(2 * pi * 179.93 * t + -1.65) +
            2 * np.cos(2 * pi * 59.98 * t + -0.36) + 2 * np.cos(2 * pi * 184.92 * t + 1.51) +
            2 * np.cos(2 * pi * 209.91 * t + -3.05)
    )


def x_out_func(t):
    return (
            2 * np.cos(2 * pi * 0.00 * t + 0.00) + 2 * np.cos(2 * pi * 29.99 * t + -1.69) +
            2 * np.cos(2 * pi * 34.99 * t + -1.75) + 2 * np.cos(2 * pi * 24.99 * t + 1.37) +
            2 * np.cos(2 * pi * 39.98 * t + -1.71)
    )


def y_out_func(t):
    return (
            2 * np.cos(2 * pi * 29.99 * t + -0.12) + 2 * np.cos(2 * pi * 179.93 * t + 0.06) +
            2 * np.cos(2 * pi * 209.91 * t + -0.96) + 2 * np.cos(2 * pi * 199.92 * t + -1.96) +
            2 * np.cos(2 * pi * 214.91 * t + 2.69)
    )


geom = dde.geometry.TimeDomain(0, 0.5)


def boundary(t, on_initial):
    return on_initial and dde.utils.isclose(t[0], 0)


def bc_func0(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 2


# observe_t, ob_x = load_training_data()
# observe_y0 = dde.icbc.PointSetBC(observe_t, ob_x, component=0)


# ic1 = dde.icbc.IC(geom, lambda X: -6.137858542418662e-09, lambda _, on_initial: on_initial, component=0)
# ic2 = dde.icbc.IC(geom, lambda X: 3.403934000590857e-08, lambda _, on_initial: on_initial, component=1)
# ic3 = dde.icbc.IC(geom, lambda X: 0, lambda _, on_initial: on_initial, component=2)
# ic4 = dde.icbc.IC(geom, lambda X: 3.358945331737716e-08, lambda _, on_initial: on_initial, component=3)

boundary_condition_u = dde.icbc.DirichletBC(
    geom, x_in_func, lambda _, on_boundary: on_boundary, component=0
)
boundary_condition_v = dde.icbc.DirichletBC(
    geom, y_in_func, lambda _, on_boundary: on_boundary, component=1
)
boundary_condition_w = dde.icbc.DirichletBC(
    geom, x_out_func, lambda _, on_boundary: on_boundary, component=2
)
boundary_condition_z = dde.icbc.DirichletBC(
    geom, y_out_func, lambda _, on_boundary: on_boundary, component=2
)
initial_condition_u = dde.icbc.IC(
    geom, x_in_func, lambda _, on_initial: on_initial, component=0
)
initial_condition_v = dde.icbc.IC(
    geom, y_in_func, lambda _, on_initial: on_initial, component=1
)
initial_condition_w = dde.icbc.IC(
    geom, x_out_func, lambda _, on_initial: on_initial, component=2
)
initial_condition_z = dde.icbc.IC(
    geom, y_out_func, lambda _, on_initial: on_initial, component=2
)

data = dde.data.TimePDE(geom,
                        bearing_system,
                        # [ic1, ic2, ic3, ic4],
                        # [],
                        [boundary_condition_u, boundary_condition_v, boundary_condition_w, boundary_condition_z,
                         initial_condition_u, initial_condition_v, initial_condition_w, initial_condition_z],
                        num_domain=50000,
                        num_boundary=5000,
                        # num_initial=5000,
                        # solution=func,
                        num_test=10000,
                        )

layer_size = [1] + [50] * 4 + [4]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam",
              lr=0.001,
              # metrics=["l2 relative error"],
              loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 100, 100]
              )

losshistory, train_state = model.train(iterations=40000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
