import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# 动力学方程系统
def dynamic_system(t, y):
    # 解包状态变量
    x_in, u_in, y_in, v_in, x_out, u_out, y_out, v_out = y

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

    # 计算omega_c
    omega_c = (1 / 2) * (1 - (D_b / D_p) * np.cos(alpha)) * omega

    # 初始化F_HX
    F_HX = 0
    F_HY = 0
    # 计算求和
    for i in range(1, N + 1):
        theta_i = 2 * np.pi * (i - 1) / N + omega_c * t
        delta_i = (x_in - x_out) * np.sin(theta_i) + (y_in - y_out) * np.cos(theta_i) - C_r
        if delta_i > 0:
            F_HX += K * delta_i ** 1.5 * np.sin(theta_i)
            F_HY += K * delta_i ** 1.5 * np.cos(theta_i)

    # 动力学方程
    dot_x_in = u_in
    dot_u_in = (-c_in * u_in - k_in * x_in - F_HX + e * m_in * omega ** 2 * np.cos(omega * t)) / m_in
    dot_y_in = v_in
    dot_v_in = (-c_in * v_in - k_in * y_in + m_in * g - F_HY + e * m_in * omega ** 2 * np.sin(omega * t)) / m_in
    dot_x_out = u_out
    dot_u_out = (F_HX - c_out * u_out - k_out * x_out) / m_out
    dot_y_out = v_out
    dot_v_out = (F_HY + m_out * g - c_out * v_out - k_out * y_out) / m_out

    # 返回导数向量
    return [dot_x_in, dot_u_in, dot_y_in, dot_v_in, dot_x_out, dot_u_out, dot_y_out, dot_v_out]


# 初始状态向量
x_in_0, u_in_0, y_in_0, v_in_0, x_out_0, u_out_0, y_out_0, v_out_0 = 0, 0, 0, 0, 0, 0, 0, 0
initial_state = [x_in_0, u_in_0, y_in_0, v_in_0, x_out_0, u_out_0, y_out_0, v_out_0]

# 时间跨度
t_span = (0, 0.5)
t_eval = np.linspace(t_span[0], t_span[1], 12000)

# 调用solve_ivp求解
sol = solve_ivp(dynamic_system, t_span, initial_state, t_eval=t_eval)

# 绘图
# plt.plot(sol.t, sol.y[0], label='x_in')
# plt.plot(sol.t, sol.y[2], label='y_in')
# plt.plot(sol.t, sol.y[4], label='x_out')
# 假设y_out是解矩阵中的最后一行
y_out = sol.y[6]

# 归一化y_out
y_out_min = np.min(y_out)
y_out_max = np.max(y_out)
y_out_normalized = (y_out - y_out_min) / (y_out_max - y_out_min)

# 现在y_out_normalized包含了归一化的y_out
plt.plot(sol.t, y_out_normalized, label='y_out')
plt.legend()
plt.show()
