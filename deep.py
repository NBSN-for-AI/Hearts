'''
Author: Zhuiy
'''

from game import game, Card, Suit
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

meanlog = []

class Policy_net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(Policy_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.to(device)
    def forward(self, state):
        return F.softmax(self.fc2(F.relu(self.fc1(state))), dim=-1)
    
class RF:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, device):
        self.policy_net = Policy_net(state_dim, hidden_dim, action_dim, device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device
        self.lr = lr
        self.transition_dict = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        self.log = []

    def take_action(self, state, mask=None):
        # 预转换为tensor并移到GPU
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
            
        probs = self.policy_net(state)
        if mask is not None:
            probs = probs.masked_fill(~mask, 0)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                # 如果所有动作都被mask，随机选择一个可用动作
                probs = mask.float() / mask.float().sum()
                
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)

        G = 0
        returns = []
        # 计算折扣回报
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float).view(-1, 1).to(self.device)
        
        # 标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        self.optimizer.zero_grad()
        loss = 0
        for i in range(len(returns)):
            state = states[i].unsqueeze(0)
            action = actions[i]
            probs = self.policy_net(state)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            loss += -log_prob * returns[i]
            
        loss = loss / len(returns)  # 平均损失
        loss.backward()
        self.optimizer.step()

def info_to_tensor(info):
    hand = info['hand']
    hand_tensor = torch.zeros(52)
    for card in hand:
        index = (card.suit.value) * 13 + card.rank - 1
        hand_tensor[index] = 1
    points_tensor = torch.tensor([info['points']], dtype=torch.float)
    table_tensor = torch.zeros(52)
    for card, _ in info['table']:
        index = (card.suit.value) * 13 + card.rank - 1
        table_tensor[index] = 1
    current_suit_tensor = torch.zeros(4)
    if info['current_suit'] is not None:
        current_suit_tensor[info['current_suit'].value] = 1
    else:
        current_suit_tensor = torch.zeros(4)
    current_table_tensor = torch.zeros(52)
    for card, _ in info['current_table']:
        index = (card.suit.value) * 13 + card.rank - 1
        current_table_tensor[index] = 1
    hearts_broken_tensor = torch.tensor([1.0 if info['hearts_broken'] else 0.0], dtype=torch.float)
    piggy_pulled_tensor = torch.tensor([1.0 if info['piggy_pulled'] else 0.0], dtype=torch.float)
    state_tensor = torch.cat([hand_tensor, points_tensor, table_tensor, current_suit_tensor, current_table_tensor, hearts_broken_tensor, piggy_pulled_tensor], dim=0)
    return state_tensor.numpy()

def actions_to_mask(actions):
    mask = torch.zeros(52, dtype=torch.bool)
    for card in actions:
        index = (card.suit.value) * 13 + card.rank - 1
        mask[index] = True
    return mask.numpy()

def action_to_card(action):
    suit = action // 13
    rank = action % 13
    return Card(Suit(suit), rank + 1)

model = RF(163, 128, 52, 1e-3, 0.99, 'cuda')
mirror_model = copy.deepcopy(model)

def rl_policy(player, player_info, actions, order):
    global model
    state = info_to_tensor(player_info)
    mask = actions_to_mask(actions)
    action = model.take_action(state, mask)
    
    # 简化的奖励：主要依赖最终结果
    model.transition_dict['rewards'].append(0)  # 中间步骤奖励为0
    model.transition_dict['actions'].append(action)
    model.transition_dict['states'].append(state)
    
    return action_to_card(action)

def mirror_policy(player, player_info, actions, order):
    global mirror_model
    action = mirror_model.take_action(info_to_tensor(player_info), actions_to_mask(actions))
    return action_to_card(action)

def train(model):
    global mirror_model
    for episode in range(200):
        print('----------------------------------------------------')
        print(f'episode: {episode}')
        model.transition_dict = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        
        # 运行游戏
        scores = game([rl_policy, mirror_policy, mirror_policy, mirror_policy])
        final_reward = -scores[0]  # 负得分作为奖励
        
        # 关键修正：给所有transition分配最终奖励
        num_transitions = len(model.transition_dict['actions'])
        if num_transitions > 0:
            # 给所有动作相同的最终奖励
            model.transition_dict['rewards'] = [final_reward] * num_transitions
            model.update(model.transition_dict)
        
        # 定期更新镜像模型
        if episode % 20 == 0 and episode > 0:
            mirror_model = copy.deepcopy(model)
            recent_rewards = model.log[-20:] if len(model.log) >= 20 else model.log
            mean_reward = np.mean(recent_rewards)
            meanlog.append(mean_reward)
            print(f"Episode {episode}, Mean Reward: {mean_reward:.2f}")
        
        model.log.append(final_reward)
        print(f'Episode {episode}, Final Reward: {final_reward}')

if __name__ == '__main__':
    print("Start Training")
    print('----------------------------------------------------')
    train(model)
    print("Training Finished")
    
    # 保存结果
    pd.DataFrame(model.log).to_csv('./data/Zhuiy_rl_log.csv', index=False)
    pd.DataFrame(meanlog).to_csv('./data/Zhuiy_rl_meanlog.csv', index=False)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model.log)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(meanlog)
    plt.title('Mean Reward (every 20 episodes)')
    plt.xlabel('Evaluation Point')
    plt.ylabel('Mean Reward')
    
    plt.tight_layout()
    plt.savefig('./data/training_curve.png')
    plt.show()