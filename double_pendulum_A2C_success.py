# パッケージのimport
import numpy as np
import matplotlib.pyplot as plt
import gym
from numpy import sin, cos
from matplotlib.animation import FuncAnimation,ArtistAnimation
import pandas as pd
import seaborn as sns
sns.set()



# 定数の設定
ENV = 'MyEnv-v0'  # 使用する課題名
GAMMA = 0.9  # 時間割引率
MAX_STEPS = 80  # 1試行のstep数
NUM_EPISODES = 1000000  # 最大試行回数

NUM_PROCESSES = 30  # 同時に実行する環境
NUM_ADVANCED_STEP = 4  # 何ステップ進めて報酬和を計算するのか設定

# A2Cの損失関数の計算のための定数設定
value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5

# メモリクラスの定義
class RolloutStorage(object):
    '''Advantage学習するためのメモリクラスです'''

    def __init__(self, num_steps, num_processes, obs_shape):

        self.observations = torch.zeros(num_steps + 1, num_processes, 4)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        # 割引報酬和を格納
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.index = 0  # insertするインデックス

    def insert(self, current_obs, action, reward, mask):
        '''次のindexにtransitionを格納する'''
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # インデックスの更新

    def after_update(self):
        '''Advantageするstep数が完了したら、最新のものをindex0に格納'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''Advantageするステップ中の各ステップの割引報酬和を計算する'''

        # 注意：5step目から逆向きに計算しています
        # 注意：5step目はAdvantage1となる。4ステップ目はAdvantage2となる。・・・
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * \
                GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


     

# A2Cのディープ・ニューラルネットワークの構築
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc_in = nn.Linear(n_in, n_mid)
        self.fc_mid = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out)  # 行動を決めるので出力は行動の種類数
        self.critic = nn.Linear(n_mid, 1)  # 状態価値なので出力は1つ
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        '''ネットワークのフォワード計算を定義します'''
        x = F.relu(self.fc_in(x))
        x = self.fc_mid(x)
        x = self.fc_mid(x)
        #x = self.dropout(x)
        x = self.fc_mid(x)
        critic_output = self.critic(x)  # 状態価値の計算
        actor_output = self.actor(x)  # 行動の計算

        return critic_output, actor_output

    def act(self, x):
        '''状態xから行動を確率的に求めます'''
        value, actor_output = self(x)
        # dim=1で行動の種類方向にsoftmaxを計算
        action_probs = F.softmax(actor_output, dim=1)
        #print(action_probs)
        action = action_probs.multinomial(num_samples=1)  # dim=1で行動の種類方向に確率計算
        return action

    def get_value(self, x):
        '''状態xから状態価値を求めます'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''状態xから状態価値、実際の行動actionsのlog確率とエントロピーを求めます'''
        """valueはrolloutから吐き出されたエージェント数＊advantageのステップ数の分だけ推定する"""
        
        
        value, actor_output = self(x)
        
        #print("value, actor_output")
        #print(value.shape, actor_output.shape)
        print("action", actions)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1で行動の種類方向に計算
        action_log_probs = log_probs.gather(1, actions)  # 実際の行動のlog_probsを求める
        #print("log_probs, action_log_probs")
        #print(log_probs.shape, action_log_probs)

        probs = F.softmax(actor_output, dim=1)  # dim=1で行動の種類方向に計算
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


     

# エージェントが持つ頭脳となるクラスを定義、全エージェントで共有する
import torch
from torch import optim


class Brain(object):
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic  # actor_criticはクラスNetのディープ・ニューラルネットワーク
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.0001)
        
        load_file = True
        self.file_name = "Double_Pendulum_A2C_Default.pth"
        self.i = 4
        
        #学習状況を読み込む
        if load_file:
            checkpoint_main = torch.load(self.file_name)
            self.actor_critic.load_state_dict(checkpoint_main["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint_main["optimizer_state_dict"])
            self.episode = checkpoint_main["episode"]
            
            
        else:
            self.episode = 0
        
    def model_save(self, episode): #最後にモデルを保存する関数
        torch.save({
            "model_state_dict":self.actor_critic.state_dict(),
            "episode":episode,
            "optimizer_state_dict":self.optimizer.state_dict()
            },
            self.file_name)
        

    def update(self, rollouts):
        '''Advantageで計算した5つのstepの全てを使って更新します'''
        obs_shape = rollouts.observations.size()[2:]  # torch.Size([4, 84, 84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 4),
            rollouts.actions.view(-1, 1))

        # 注意：各変数のサイズ
        # rollouts.observations[:-1].view(-1, 4) torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(num_steps, num_processes,
                             1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # advantage（行動価値-状態価値）の計算
        advantages = rollouts.returns[:-1] - values  # torch.Size([5, 16, 1])

        # Criticのlossを計算
        value_loss = advantages.pow(2).mean()

        # Actorのgainを計算、あとでマイナスをかけてlossにする
        action_gain = (action_log_probs*advantages.detach()).mean()
        # detachしてadvantagesを定数として扱う

        # 誤差関数の総和
        total_loss = (value_loss * value_loss_coef -
                      action_gain - entropy * entropy_coef)

        # 結合パラメータを更新
        self.actor_critic.train()  # 訓練モードに
        self.optimizer.zero_grad()  # 勾配をリセット
        total_loss.backward()  # バックプロパゲーションを計算
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        #  一気に結合パラメータが変化しすぎないように、勾配の大きさは最大0.5までにする

        self.optimizer.step()  # 結合パラメータを更新


    

     

# 実行する環境のクラスです
class Environment:
    def run(self):
        '''メインの実行'''

        # 同時実行する環境数分、envを生成
        envs = [gym.make(ENV) for i in range(NUM_PROCESSES)]

        # 全エージェントが共有して持つ頭脳Brainを生成
        n_in = envs[0].observation_space.shape[0]  # 状態は4
        n_out = envs[0].action_space.n  # 行動は11
        n_mid = 128
        actor_critic = Net(n_in, n_mid, n_out)  # ディープ・ニューラルネットワークの生成
        global_brain = Brain(actor_critic)
        episode = global_brain.episode
        

        # 格納用変数の生成
        obs_shape = n_in
        current_obs = torch.zeros(
            NUM_PROCESSES, obs_shape)  # torch.Size([16, 4])
        rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)  # rolloutsのオブジェクト
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])  # 現在の試行の報酬を保持
        final_rewards = torch.zeros([NUM_PROCESSES, 1])  # 最後の試行の報酬を保持
        obs_np = np.zeros([NUM_PROCESSES, obs_shape])  # Numpy配列
        reward_np = np.zeros([NUM_PROCESSES, 1])  # Numpy配列
        done_np = np.zeros([NUM_PROCESSES, 1])  # Numpy配列
        each_step = np.zeros(NUM_PROCESSES)  # 各環境のstep数を記録
        
        

        # 初期状態の開始
        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()  # torch.Size([16, 4])
        current_obs = obs  # 最新のobsを格納

        # advanced学習用のオブジェクトrolloutsの状態の1つ目に、現在の状態を保存
        rollouts.observations[0].copy_(current_obs)
        
        #実行結果の表示のために変数の用意
        tmp_y = envs[0].reset()
        frames = np.array([[tmp_y[0]], [tmp_y[1]], [tmp_y[2]], [tmp_y[3]]]) #最終試行の結果を保存する
        episode_final = False

        # 実行ループ
        for j in range(episode+1, episode+NUM_EPISODES+1):  # 全体のforループ
            if episode_final:
                break
            
        
            # advanced学習するstep数ごとに計算
            #複数ステップ実行して結果を回収する
            for step in range(NUM_ADVANCED_STEP):
                if episode_final:
                    break

                # 行動を求める
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                # (16,1)→(16,)→tensorをNumPyに
                actions = action.squeeze(1).numpy()

                # 1stepの実行
                for i in range(NUM_PROCESSES): #複数のエージェントでの試行
                    if episode_final:
                        break
                    
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(
                        actions[i])
                    

                    
                    
                    reward_np[i] = torch.FloatTensor([obs_np[i, 1]])
                    #reward_np[i] = torch.FloatTensor([0.0])
                    
                    if each_step[i] == MAX_STEPS-1:
                        done_np[i] = True

                    # episodeの終了評価と、state_nextを設定
                    if done_np[i]:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる

                        reward_np[i] = torch.FloatTensor([obs_np[i, 1]])
                        
                        if each_step[i] != MAX_STEPS-1:
                            reward_np[i] = torch.FloatTensor([-1.0]) 

                        # 環境0のときのみ出力
                        if i == 0:
                            print('%d Episode: Finished after %d steps' % (
                                episode, each_step[i]+1), np.degrees(obs_np[i, 0]))
                            
                            if np.degrees(obs_np[0, 0]) >= 2250 and each_step[0]+1 == MAX_STEPS:
                                frames = np.concatenate([frames, envs[0].render(mode="rgb_array")], 1)
                                episode_final = True
                            else:
                                episode += 1
                                tmp_y = envs[0].reset()
                                frames = np.array([[tmp_y[0]], [tmp_y[1]], [tmp_y[2]], [tmp_y[3]]]) #最終試行の結果を保存する

                        each_step[i] = 0  # step数のリセット
                        obs_np[i] = envs[i].reset()  # 実行環境のリセット

                    else:
                        #環境０のときのみ出力
                        if i == 0:
                            frames = np.concatenate([frames, envs[0].render(mode="rgb_array")], 1)
                        each_step[i] += 1
                        

                # 報酬をtensorに変換し、試行の総報酬に足す
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # 各実行環境それぞれについて、doneならmaskは0に、継続中ならmaskは1にする
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done_np])

                # 最後の試行の総報酬を更新する
                final_rewards *= masks  # 継続中の場合は1をかけ算してそのまま、done時には0を掛けてリセット
                # 継続中は0を足す、done時にはepisode_rewardsを足す
                final_rewards += (1 - masks) * episode_rewards

                # 試行の総報酬を更新する
                episode_rewards *= masks  # 継続中のmaskは1なのでそのまま、doneの場合は0に

                # 現在の状態をdone時には全部0にする
                current_obs *= masks

                # current_obsを更新
                obs = torch.from_numpy(obs_np).float()  # torch.Size([16, 4])
                current_obs = obs  # 最新のobsを格納
                

                # メモリオブジェクトに今stepのtransitionを挿入
                rollouts.insert(current_obs, action.data, reward, masks)

            # advancedのfor loop終了

            # advancedした最終stepの状態から予想する状態価値を計算

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()
                # rollouts.observationsのサイズはtorch.Size([6, 16, 4])

            # 全stepの割引報酬和を計算して、rolloutsの変数returnsを更新
            rollouts.compute_returns(next_value)

            # ネットワークとrolloutの更新
            global_brain.update(rollouts)
            rollouts.after_update()

            if episode % 50 == 0:
                global_brain.model_save(episode)

            # 全部のNUM_PROCESSESが200step経ち続けたら成功
            #if final_rewards.sum().numpy() >= NUM_PROCESSES:
            if episode >= 300000:
                print('連続成功')
                #最後に描画のために１エピソードだけ実行する
                tmp_y = envs[0].reset()
                frames = np.array([[tmp_y[0]], [tmp_y[1]], [tmp_y[2]], [tmp_y[3]]]) #最終試行の結果を保存する

            
                for step in range(MAX_STEPS):
                    # 行動を求める
                    with torch.no_grad():
                        action = actor_critic.act(rollouts.observations[0])
                    actions = action.squeeze(1).numpy()
                        
                    obs_np[0], reward_np[0], done_np[0], _ = envs[0].step(actions[0])
                    
                    
                    frames = np.concatenate([frames, envs[0].render(mode="rgb_array")], 1)
                    
                    if done_np[0]:
                        break
                break
            
            
        
            
            
        #各エージェントを閉じる    
        for i in range(NUM_PROCESSES):
            envs[i].close()
            
        return frames


     

# main学習
cartpole_env = Environment()
y = cartpole_env.run()


y = y[:, 1:]
L1 = 10.0
L2 = 10.0
print(y.shape)

if True: #ｃｓｖファイルで結果を出力したい場合はＴｒｕｅにすること
    y_df = pd.DataFrame({"theta1":np.degrees(y[0]), "omega1":np.degrees(y[1]), "theta2":np.degrees(y[2]), "omega2":np.degrees(y[3])})
    print(y_df)
    y_df.to_csv("over2250_result.csv")

    sns.jointplot(x="omega1", y="omega2", data=y_df[["omega1", "omega2"]])

if False: #アニメーションを表示・保存したいときはTrueに変えること
    t = np.arange(0, y.shape[1]*0.05, 0.05)
    #ジェネレーター関数
    def gen():
        for tt,th1,th2 in zip(t,y[0,:],y[2,:]):
            x1=L1*sin(th1)
            y1=-L1*cos(th1)
            x2=L2*sin(th2)+x1
            y2=-L2*cos(th2)+y1
            yield tt,x1,y1,x2,y2
            
    #描画準備
    fig,ax=plt.subplots()
    ax.set_xlim(-(L1+L2),L1+L2)
    ax.set_ylim(-(L1+L2),L1+L2)
    ax.set_aspect("equal")
    ax.grid()
    
    locus, =ax.plot([],[],"r-",linewidth=2)
    line, =ax.plot([],[],linewidth=2)
    time_template="time=%.1fs"
    time_text=ax.text(0.05,0.9,"",transform=ax.transAxes)
    
    xlocus,ylocus=[],[]
    
    
    #描画
    def animate(data):
        t,x1,y1,x2,y2=data
        xlocus.append(x2)
        ylocus.append(y2)
        locus.set_data(xlocus,ylocus)
        line.set_data([0,x1,x2],[0,y1,y2])
        time_text.set_text(time_template % (t))
    
    ani=FuncAnimation(fig, animate, gen,interval=50,repeat=True, save_count=2000)
    
    #保存するときだけコメントを外す
    #filename = "th1;{0},w1;{1},th2;{2},w2;{3},L1;{4},L2;{5}_ani.gif".format(th1,w1,th2,w2,L1,L2)
    ani.save("over2350.gif", writer= "imagemagick")
    
    
    
    
    #----------------thetaのアニメーション作成-------------------------
    # fig, axオブジェクトを作成
    fig = plt.figure(figsize = (7.5,7.5))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 2)
    ax4 = fig.add_subplot(2, 2, 4)
    
    
    
    # グラフのリスト作成
    ims=[]
    t_nows=np.array([])
    theta1_nows = np.array([])
    theta2_nows = np.array([])
    omega1_nows = np.array([])
    omega2_nows = np.array([])
    
    for theta1, theta2, omega1, omega2, t_now in zip(y[0].tolist(), y[2].tolist(), y[1].tolist(), y[3].tolist(), t.tolist()): 
        t_nows=np.append(t_nows,[t_now])
        theta1_nows = np.append(theta1_nows, np.degrees(theta1))
        theta2_nows = np.append(theta2_nows, np.degrees(theta2))
        omega1_nows = np.append(omega1_nows, np.degrees(omega1))
        omega2_nows = np.append(omega2_nows, np.degrees(omega2))
        
        im3 = ax1.plot(t, np.degrees(y[0]), color="lightgray")
        im4 = ax2.plot(t, np.degrees(y[2]), color="lightgray")
        im1 = ax1.plot(t_nows, theta1_nows, color='r',label="theta1", lw=2.5)
        im2 = ax2.plot(t_nows, theta2_nows, color="b",label="theta2", lw=2.5)
        
        im7 = ax3.plot(t, np.degrees(y[1]), color="lightgray")
        im8 = ax4.plot(t, np.degrees(y[3]), color="lightgray")
        im5 = ax3.plot(t_nows, omega1_nows, color='r',label="omega1", lw=2.5)
        im6 = ax4.plot(t_nows, omega2_nows, color="b",label="omega2", lw=2.5)
        
        ims.append(im1+im2+im3+im4+im5+im6+im7+im8)
       
       
    
    # 各軸のラベル
    ax.set_xlabel(r"$t$", fontsize=15)
    ax.set_ylabel("theta", fontsize=15)
    plt.legend()
    # グラフの範囲を設定
    ax1.set_xlim([0, max(t)])
    ax1.set_ylim([min(np.degrees(y[0]))-10, max(np.degrees(y[0]))+10]) 
    ax2.set_xlim([0, max(t)])
    ax2.set_ylim([min(np.degrees(y[2]))-10, max(np.degrees(y[2]))+10])
    
    ax3.set_xlim([0, max(t)])
    ax3.set_ylim([min(np.degrees(y[1]))-10, max(np.degrees(y[1]))+10]) 
    ax4.set_xlim([0, max(t)])
    ax4.set_ylim([min(np.degrees(y[3]))-10, max(np.degrees(y[3]))+10])  
    
    # ArtistAnimationにfigオブジェクトとimsを代入してアニメーションを作成
    ax1.legend(['theta1'], loc='upper right')
    ax2.legend(['theta2'], loc='upper right')
    ax3.legend(['omega1'], loc='upper right')
    ax4.legend(['omega2'], loc='upper right')
    anim = ArtistAnimation(fig, ims, interval=50)
    
    #保存するときだけコメントを外す
    #filename2 = "th1;{0},w1;{1},th2;{2},w2;{3},L1;{4},L2;{5}_th.gif".format(th1,w1,th2,w2,L1,L2)
    anim.save("over2350th.gif", writer= "imagemagick")
    
    
    
    plt.plot()