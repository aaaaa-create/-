from CraneSystem2DOF import Crane2DModel
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random


env = Crane2DModel()
done = False

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SACの実装

class ClippedCriticNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size):

        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_dim)

        self.linear4 = nn.Linear(input_dim, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, output_dim)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class SoftActorNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size, action_scale):

        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, output_dim)
        self.log_std_linear = nn.Linear(hidden_size, output_dim)

        self.action_scale = torch.tensor(action_scale).to(device)
        self.action_bias = torch.tensor(0.).to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SoftActorCriticModel(object):

    def __init__(self, state_dim, action_dim, action_scale, args, device):

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.device = device
        self.target_update_interval = args['target_update_interval']

        self.actor_net = SoftActorNet(
            input_dim=state_dim, output_dim=action_dim, hidden_size=args['hidden_size'], action_scale=action_scale
        ).to(self.device)
        self.critic_net = ClippedCriticNet(input_dim=state_dim + action_dim, output_dim=1, hidden_size=args['hidden_size']).to(device=self.device)
        self.critic_net_target = ClippedCriticNet(input_dim=state_dim + action_dim, output_dim=1, hidden_size=args['hidden_size']).to(self.device)

        hard_update(self.critic_net_target, self.critic_net)
        convert_network_grad_to_false(self.critic_net_target)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=0.001)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=0.001)

        self.target_entropy = -action_dim  # 修正

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha])

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not evaluate:
            action, _, _ = self.actor_net.sample(state)
        else:
            _, _, action = self.actor_net.sample(state)
        return action.cpu().detach().numpy().reshape(-1)

    def update_parameters(self, memory, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor_net.sample(next_state_batch)
            next_q1_values_target, next_q2_values_target = self.critic_net_target(next_state_batch, next_action)
            next_q_values_target = torch.min(next_q1_values_target, next_q2_values_target) - self.alpha * next_log_pi
            next_q_values = reward_batch + mask_batch * self.gamma * next_q_values_target

        q1_values, q2_values = self.critic_net(state_batch, action_batch)
        critic1_loss = F.mse_loss(q1_values, next_q_values)
        critic2_loss = F.mse_loss(q2_values, next_q_values)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action, log_pi, _ = self.actor_net.sample(state_batch)

        q1_values, q2_values = self.critic_net(state_batch, action)
        q_values = torch.min(q1_values, q2_values)

        actor_loss = ((self.alpha * log_pi) - q_values).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_net_target, self.critic_net, self.tau)

        return critic_loss.item(), actor_loss.item()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def convert_network_grad_to_false(network):
    for param in network.parameters():
        param.requires_grad = False

class ReplayMemory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)




# 状態と行動の次元数
state_dim = 6  # ここでは[alpha, alpha_dot, beta, beta_dot, x_dot_dot, u]のみを制御させるため5に設定
action_dim = env.action_space.shape[0]
action_scale = env.action_space.high[0]

# エージェントの作成
args = {
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.2,
    'seed': 12345,
    'batch_size': 256,
    'hidden_size': 512,
    'start_steps': 15000,
    'updates_per_step': 1,
    'target_update_interval': 1,
    'memory_size': 10000,
    'epochs': 10000,
    'eval_interval': 50
}

n_steps = 0
n_update = 0

agent = SoftActorCriticModel(
    state_dim=state_dim, action_dim=action_dim,
    action_scale=action_scale, args=args, device=device
)
memory = ReplayMemory(args['memory_size'])
max_score = 0

for i_episode in range(1, args["epochs"] + 1):
    # 状態データを保存するリストを初期化
    result_log = env.reset().reshape(6, 1)
    done = False
    episode_reward = 0
    while not done:
        state = env.state.copy()
        state[0] -= env.x_r
        u_position = -env.f_position @ state[:2]
        
       

        state6 = np.concatenate((state[2:], np.array([env.x_dot_dot]), np.ravel([u_position])))
        if args['start_steps'] > n_steps:
            u_theta = env.action_space.sample()
        else:
            u_theta = agent.select_action(state6)
        if len(memory) > args['batch_size']:
            for _ in range(args['updates_per_step']):
                agent.update_parameters(memory, args['batch_size'], n_update)
                n_update += 1
        n_steps += 1

     
        action = u_position  + u_theta 
        next_state, reward, done, info = env.step(action, u_theta)
        episode_reward += reward
        result_log = np.concatenate([result_log, next_state.reshape(6, 1)], 1)


        next_state = env.state.copy()
        next_state[0] -= env.x_r 
        u_position_next = -env.f_position @ next_state[:2]
        state6_next = np.concatenate((next_state[2:], np.array([env.x_dot_dot]), np.ravel([u_position_next])))

        mask = 0 if done else 1
        memory.push(state=state6, action=u_theta, reward=reward, next_state=state6_next, mask=mask)

    torch.save(agent.actor_net.to('cpu').state_dict(), 'LQR_SAC.pth')

    if state[-1] < 10 ** (-2) and episode_reward >= max_score:
        max_score = episode_reward + 0.0001
        np.savetxt("LQR_SAC2.csv", result_log.T, delimiter=",")


    print(i_episode, env.time, env.state, episode_reward)


