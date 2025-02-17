# Shiokawa Ryuji
import gym
import numpy as np
import math
import control  #Pyton-Controlの設定
from control.matlab import *  #MATLAB®ライクなコマンド形式の設定
import numpy as np  #Numpyの設定
import matplotlib.pyplot as plt  #グラフ表示用モジュールの設定
#配列のprintでの表示時の小数点以下桁数指定
np.set_printoptions(precision=4, floatmode='maxprec')
from scipy.integrate import solve_ivp
from numpy import sin, cos
from matplotlib.animation import FuncAnimation
from gym import spaces


class Crane2DModel(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Crane2DModel, self).__init__()
        self.M = 13.4 * 10 ** (-3)  # 台車質量
        self.m1 = 13.4 * 10 ** (-3)  # フックの質量
        self.m2 = 9.98 * 10 ** (-3)  # 荷物の質量
        self.l1 = 0.18  # ロープの長さ
        self.l2 = 0.09  # 玉掛けロープの長さ
        self.c1 = 7.5 * 10 ** (-4)  # 台車とフック間の減衰係数
        self.c2 = 2.5 * 10 ** (-4)  # フックと荷物間の減衰係数
        self.g = 9.81  # 重力加速度
        self.Tx = 0.1  # 時定数
        self.K = 0.235  # ゲイン
        self.F_max = 5  # 電圧の最大値
        self.dt = 0.005  # タイムステップ幅
        self.x_r = 0.4  # 目標地点

        self.t_span = [0, 40]  # シミュレーション時間
        self.t = np.arange(self.t_span[0], self.t_span[1], self.dt)

        self.H1 = self.c1 / self.m1 / self.l1 ** 2
        self.H2 = (self.l1 + self.l2) * self.c2 / self.m1 / self.l1 ** 2 / self.l2
        self.H3 = self.g / self.l1 
        self.H4 = self.m2 * self.g / self.m1 / self.l1
        self.H5 = 1 / self.l1
        self.H6 = self.c1 / self.m1 / self.l1 / self.l2
        self.H7 = (self.m2 * self.l2 + (self.m1 + self.m2) * self.l1) * self.c2 / self.m1 / self.m2 / self.l1/ self.l2 ** 2
        self.H8 = (self.m1 + self.m2) * self.g / self.m1 / self.l2

        self.state = np.zeros(6)  # 状態変数[x, x_dot, alpha, alpha_dot, beta, beta_dot]
        self.u = 0  # 制御入力
        self.time = 0

    
        high = np.array([np.inf, np.inf, np.pi/2, np.inf, np.pi/2, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.u_max = 0.5
        self.action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(1,), dtype=np.float32)

        # 最適制御ゲイン(all)
        A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, -1 / self.Tx, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, -self.H5 / self.Tx, -(self.H3 + self.H4), -(self.H1+ self.H2), self.H4, self.H2],
        [0, 0, 0, 0, 0, 1],
        [0, 0, self.H8, self.H6 + self.H7, -self.H8, -self.H7]
        ])

        B = np.array([[0], [self.K / self.Tx], [0], [self.K * self.H5 / self.Tx], [0], [0]])
        C = np.eye(6)
        D = np.zeros([6, 1])

        sys1 = ss(A, B, C, D)


        # 重みも設定
        q = 10
        Q1 = np.diag([q, q, q, q, q, q])
        r1 = 1

        # 最適レギュレータの設定
        self.f_all, P1, e1 = lqr(sys1.A, sys1.B, Q1, r1)

        # 最適制御ゲイン(角度のみ)
        A = np.array([
        [0, 1, 0, 0],
        [-(self.H3 + self.H4), -(self.H1+ self.H2), self.H4, self.H2],
        [0, 0, 0, 1],
        [self.H8, self.H6 + self.H7, -self.H8, -self.H7]
        ])

        B = np.array([[0], [self.K * self.H5 / self.Tx], [0], [0]])
        C = np.eye(4)
        D = np.zeros([4, 1])

        sys1 = ss(A, B, C, D)


        # 重みも設定
        q = 10
        Q1 = np.diag([q, q, q, q])
        r1 = 1

        # 最適レギュレータの設定
        self.f_angle, P1, e1 = lqr(sys1.A, sys1.B, Q1, r1)

        # 最適制御ゲイン（位置のみ）
        A = np.array([
        [0, 1],
        [0, -1 / self.Tx],
        ])

        B = np.array([[0], [self.K / self.Tx]])
        C = np.eye(2)
        D = np.zeros([2, 1])

        sys1 = ss(A, B, C, D)


        # 重みも設定
        q = 10
        Q1 = np.diag([q, q])
        r1 = 1

        # 最適レギュレータの設定
        self.f_position, P1, e1 = lqr(sys1.A, sys1.B, Q1, r1)



    def derivs(self, t, state):
        x, x_dot, alpha, alpha_dot, beta, beta_dot = self.state
        dxdt = np.zeros_like(state) 
        dxdt[0] = x_dot
        dxdt[1] = -x_dot / self.Tx + self.K / self.Tx * self.u
        dxdt[2] = alpha_dot
        dxdt[3] = -(self.H1 + self.H2) * alpha_dot + self.H2 * beta_dot - (self.H3 + self.H4) * alpha + self.H4 * beta + self.H5 * dxdt[1]
        dxdt[4] = beta_dot
        dxdt[5] = (self.H6 + self.H7) * alpha_dot - self.H7 * beta_dot + self.H8 * alpha - self.H8 * beta
        self.x_dot_dot = dxdt[1]
        return dxdt
    
    def reward_func(self, action):
        a1 = self.state[2] ** 2
        a2 = self.state[3] ** 2
        a3 = self.state[4] ** 2
        a4 = self.state[5] ** 2
        reward = np.exp(-a1 -a2 -a3 - a4) - np.sum(action ** 2) 
        return reward
    
    def step(self, action, u_theta):
        self.u = np.clip(action, -self.F_max, self.F_max)
        sol = solve_ivp(self.derivs, [0, self.dt], self.state, t_eval=[self.dt], rtol=1e-16, atol=1e-16)
        self.state = sol.y[:, -1]
        self.time += self.dt

        reward = self.reward_func(u_theta)

        if (abs(self.state[2]) >= np.radians(30) or 
           abs(self.state[4]) >= np.radians(30) or
            self.time >= self.t_span[1]):
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info
    
    def reset(self):
        self.state = np.zeros(6)
        self.time = 0
        self.x_dot_dot = 0 
        return self.state
    
    def reset_random(self):
        """
        ランダムな状態にリセットするメソッド。
        角度や速度はランダムな範囲から設定される。
        """
        # 各状態変数のランダムな初期値を設定
        self.state = np.array([
            0.0,  # 台車位置 (x) は 0 に固定
            np.random.uniform(-0.05, 0.05),  # 台車速度 (x_dot)
            np.random.uniform(-np.pi/24, np.pi/24),  # フック角度 (alpha)
            np.random.uniform(-0.01, 0.01),  # フック角速度 (alpha_dot)
            np.random.uniform(-np.pi/24, np.pi/24),  # 荷物角度 (beta)
            np.random.uniform(-0.01, 0.01)   # 荷物角速度 (beta_dot)
        ])
        self.time = 0
        self.x_dot_dot = 0 
        return self.state


    def gen(self):
        """
        フレームごとの描画データを生成するジェネレータ。
        フレームをすべて返し終わったあとで、one_time=Trueならウィンドウを閉じる。
        """
        pi90 = np.pi / 2 

      
        for i in range(self.n_frames):
            # 各フレームでのデータ取り出し
            tt = self.t[i]
            x = self.result_log[0, i]
            alpha = self.result_log[2, i]
            beta = self.result_log[4, i]

            x1 = x
            y1 = 0
            x2 = x1 + self.l1 * cos(alpha - pi90)
            y2 = y1 + self.l1 * sin(alpha - pi90)
            x3 = x2 + self.l2 * cos(beta - pi90)
            y3 = y2 + self.l2 * sin(beta - pi90)

            # 値を返す(この値が animate に渡される)
            yield (tt, x1, y1, x2, y2, x3, y3)

        # すべてのフレームを返し終わったら
        if self.one_time:            # 変更点: 一度だけ再生したい場合に限り
            plt.close(self.fig)      # 変更点: ウィンドウを自動的に閉じる

    # 描画準備
    def animate(self, data):
        t, x1, y1, x2, y2, x3, y3 = data
        self.xlocus.append(x3)
        self.ylocus.append(y3)
        self.locus.set_data(self.xlocus, self.ylocus)  # 荷物の軌跡
        self.line.set_data([x1, x2, x3], [y1, y2, y3])  # 台車、フック、荷物の接続線
        self.time_text.set_text(self.time_template % t)

    def show_animation(self, result_log, one_time=False):
        self.one_time = one_time
        self.result_log = result_log
        self.n_frames = min(result_log.shape[1], len(self.t))
        self.fig, ax = plt.subplots()
        ax.set_xlim(min(result_log[0, :]) - 0.5, max(result_log[0, :]) + 0.5)  # 台車の位置範囲
        ax.set_ylim(-self.l1 - self.l2 - 0.1, 0.5)  # 縦方向の描画範囲
        ax.set_aspect("equal")
        ax.grid()

        self.locus, = ax.plot([], [], "r-", linewidth=2)  # 荷物の軌跡
        self.line, = ax.plot([], [], linewidth=2)  # 台車、フック、荷物を結ぶ線
        self.time_template = "time = %.1fs"
        self.time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        self.xlocus, self.ylocus = [], []

        

        self.ani = FuncAnimation(self.fig, self.animate, self.gen, interval=20, repeat=False, cache_frame_data=False)
        

        plt.show(block=True)
       

    # アニメーション再生と終了処理
    def on_close(self):
        plt.close(self.fig)

    def show_alpha(self, result_log):
        plt.figure(figsize=(10, 6))
        plt.plot(self.t, result_log[2, :len(self.t)], label=r'$\alpha$ (rad)', color='blue')  # αデータを描画
        plt.xlabel('Time (seconds)')
        plt.ylabel(r'$\alpha$ (rad)')
        plt.title('Time vs. Alpha (Pendulum Angle)')
        plt.grid(True)
        plt.legend()
        plt.show()

    

  