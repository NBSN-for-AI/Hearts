'''
Author: Zhuiy
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.dirname(os.path.abspath(__file__)) + '/data'
from game import Game, Card, Suit
from Zhuiy_sample.Zhuiy_sample_policy import sample_policy


def data_save(data, name):
    pd.DataFrame(data).to_csv(data_path +'/' + name + '.csv', index=False)

def info_to_tensor(info) -> np.ndarray:
    hand_array = np.zeros(52, dtype=np.float32)
    table_array = np.zeros(52, dtype=np.float32)
    current_table_array = np.zeros(52, dtype=np.float32)
    
    if info['hand']:
        hand_indices = [(card.suit.value * 13 + card.rank - 1) for card in info['hand']]
        hand_array[hand_indices] = 1
    
    if info['table']:
        table_indices = [(card.suit.value * 13 + card.rank - 1) for card, _ in info['table']]
        table_array[table_indices] = 1
    
    if info['current_table']:
        current_table_indices = [(card.suit.value * 13 + card.rank - 1) for card, _ in info['current_table']]
        current_table_array[current_table_indices] = 1
    
    current_suit_array = np.zeros(4, dtype=np.float32)
    if info['current_suit'] is not None:
        current_suit_array[info['current_suit'].value] = 1
    
    points_array = np.array([info['points']], dtype=np.float32)
    hearts_broken_array = np.array([1.0 if info['hearts_broken'] else 0.0], dtype=np.float32)
    piggy_pulled_array = np.array([1.0 if info['piggy_pulled'] else 0.0], dtype=np.float32)
    
    state_array = np.concatenate([
        hand_array, points_array, table_array, 
        current_suit_array, current_table_array, 
        hearts_broken_array, piggy_pulled_array
    ])
    
    return state_array

def actions_to_mask(actions) -> np.ndarray:
    mask_array = np.zeros(52, dtype=np.bool_)
    if actions:
        action_indices = [(card.suit.value * 13 + card.rank - 1) for card in actions]
        mask_array[action_indices] = True
    return mask_array

def actions_to_tensor(actions) -> np.ndarray:
    action_array = np.zeros(52, dtype=np.float32)
    if actions:
        action_indices = [(card.suit.value * 13 + card.rank - 1) for card in actions]
        action_array[action_indices] = 1
    return action_array

def action_to_card(action) -> Card:
    suit = action // 13
    rank = action % 13 + 1
    return Card(Suit(suit), rank)

def card_to_action(card) -> int:
    return card.suit * 13 + card.rank - 1

class Value_net(nn.Module):
    def __init__(self, state_dim, hidden_dim, device):
        super(Value_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.to(device)
    
    def forward(self, state):
        return self.fc2(F.relu(self.fc1(state)))

class Policy_net(nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2,  action_dim, device):
        super(Policy_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
        self.to(device)
    def forward(self, state):
        return F.softmax(self.fc3(F.relu(self.fc2(F.relu(self.fc1(state))))), dim=-1)
    
class RF:
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, action_dim, lr, gamma, device):
        self.policy_net = Policy_net(state_dim, hidden_dim1, hidden_dim2, action_dim, device)
        self.value_net = Value_net(state_dim, (hidden_dim1 + hidden_dim2) // 2, device)
        self.gamma = gamma
        self.device = device
        self.lr = lr
        self.transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'masks': []
            }
        self.log = []
        self.loss_log = []
        self.p_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.v_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.action_log_probs = []

    def take_action(self, state, mask=None) -> int:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(self.device)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
    
        probs = self.policy_net(state)
        action_dist_raw = torch.distributions.Categorical(probs)
        if mask is not None:
            probs = probs.masked_fill(~mask, 0)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = mask.float() / mask.float().sum()
        
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()

        
        log_prob = action_dist_raw.log_prob(action)
        
        return action.item(), log_prob

    def reward_design(self, ai_score_delta, ai_actions, ai_info, shot) -> list[int]:
        if not shot:
            reward = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(13):
                reward[i] -= ai_score_delta[i] * 3
                if ai_actions[i] == 50:
                    if ai_score_delta[i] > 0:
                        reward[i] -= 20
                    else:
                        reward[i] += 10
                if ai_actions[i] < 13:
                    if ai_score_delta[i] > 0:
                        reward[i] -= 5
                    else:
                        reward[i] += 10
        else:
            reward = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
            for i in range(13):
                reward[i] += ai_score_delta[i]
        return reward

    def update(self, ai_score_delta, ai_actions, ai_log_probs, ai_info, shot):
        states = np.stack(ai_info)
        states = torch.from_numpy(states).float().to(self.device)

        actions = np.array(ai_actions)
        actions = torch.from_numpy(actions).long().view(-1, 1).to(self.device)

        rewards = np.array(self.reward_design(ai_score_delta, ai_actions, ai_info, shot))
        rewards = torch.from_numpy(rewards).float().view(-1, 1).to(self.device)

        value_predicted = self.value_net(states).float().view(-1, 1).to(self.device)

        log_probs = torch.stack(ai_log_probs).view(-1, 1).to(self.device)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.cat(returns).view(-1, 1)

        advantages = returns - value_predicted.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)


        self.p_optimizer.zero_grad()
        # L_policy = -E[logπ(a|s) * A(s,a)]
        policy_loss = -(log_probs * advantages).mean()
        policy_loss.backward()
        self.p_optimizer.step()


        self.v_optimizer.zero_grad()
        # L_value = (V(s) - G_t)^2
        value_loss = F.mse_loss(value_predicted, returns)
        value_loss.backward()
        self.v_optimizer.step()

        return policy_loss.item(), value_loss.item()
    
    def save(self, path1, path2):
        torch.save(self.policy_net.state_dict(), path1)
        torch.save(self.value_net.state_dict(), path2)

    def load(self, path1, path2):
        self.policy_net.load_state_dict(torch.load(path1))
        self.value_net.load_state_dict(torch.load(path2))
    
    def policy(self, player, player_info, actions, order) -> Card:
        a, b = self.take_action(info_to_tensor(player_info), actions_to_mask(actions))
        self.action_log_probs.append(b)
        return action_to_card(a)
    
    def train(self, game: Game, oppo_policy, episodes):
        p_loss = []
        v_loss = []
        points = []
        for i in tqdm(range(episodes)):
            self.action_log_probs = []
            score, shot, ai_score_delta, ai_actions, ai_masks, ai_info = game.fight([self.policy] + oppo_policy, True, False, False)
            ai_actions = np.array([card_to_action(action) for action in ai_actions])
            ai_masks = np.array([actions_to_mask(mask) for mask in ai_masks])
            ai_info = np.array([info_to_tensor(info) for info in ai_info])
            p, v = self.update(ai_score_delta, ai_actions, self.action_log_probs, ai_info, shot)
            p_loss.append(p)
            v_loss.append(v)
            points.append(score[0])
        self.save(data_path + '/policy_net', data_path+'/value_net')
        data_save(p_loss, 'p_loss')
        data_save(v_loss, 'v_loss')
        data_save(points, 'score')
        print(f'training finished after {episodes} episodes, mean_score: {sum(points) / episodes}, mean_p_loss: {sum(p_loss) / episodes}, mean_v_loss: {sum(v_loss) / episodes}')

    def evaluation(self, game: Game, oppo_policy, episodes):
        s = np.array([0, 0, 0, 0])
        print('evaluating')
        for i in tqdm(range(episodes)):
            score, a, b, c, d, e = game.fight([self.policy] + oppo_policy, True, False, False)
            s += score
        print(s / episodes)

if __name__ == '__main__':
    model = RF(163, 128, 64, 52, 1e-3, 0.99, 'cuda')
    Hearts = Game()
    print(card_to_action(Card(0, 1)))

    while True:
        to_train = input("Train? (y/n): ")
        if to_train in ['y', 'n']:
            break
    
    while True:
        to_load = input("Load existing model? (y/n): ")
        if to_load in ['y', 'n']:
            break
    
    if to_train == 'n':
        if to_load == 'y':
            model.load(data_path + '/policy_net', data_path + '/value_net')
            model.evaluation(Hearts, [sample_policy, sample_policy, sample_policy], 300)
        else:
            print('kicking you off')
            time.sleep(2)
            exit(0)
    else:
        mirror_model = copy.deepcopy(model)
        if to_load == 'y':
            model.load(data_path + '/policy_net', data_path + '/value_net')
            model.train(Hearts, [sample_policy, sample_policy, sample_policy], 200)
            model.evaluation(Hearts, [sample_policy, sample_policy, sample_policy], 300)
        else:
            model.train(Hearts, [sample_policy, sample_policy, sample_policy], 200)
            model.evaluation(Hearts, [sample_policy, sample_policy, sample_policy], 300)
    