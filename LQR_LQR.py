from CraneSystem2DOF import Crane2DModel
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

env = Crane2DModel()
done = False

# 状態データを保存するリストを初期化
result_log = env.reset().reshape(6, 1)
total_reward = 0


while not done:
    state = env.state.copy()
    state[0] -= env.x_r
    u_theta = - env.f_angle @ state[2:] 
    u_position = -env.f_position @ state[:2]  
    action = u_theta + u_position
    next_state, reward, done, info = env.step(action, u_theta)
    result_log = np.concatenate([result_log, next_state.reshape(6, 1)], 1)
    total_reward += reward

print(total_reward)
print(state)
env.show_animation(result_log, True)
np.savetxt("LQR_LQR.csv", result_log.T, delimiter=",")