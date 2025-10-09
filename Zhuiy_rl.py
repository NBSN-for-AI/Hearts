'''
Author: Zhuiy
'''

from game import Game, Card, Suit
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
import os
import 

from Zhuiy_sample_policy import sample_policy

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
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

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
        if mask is not None:
            probs = probs.masked_fill(~mask, 0)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = mask.float() / mask.float().sum()
        
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()  

    def update_withoutmask(self, ai_score_delta, ai_actions, ai_masks, ai_info):
        """Update without mask - NOT RECOMMENDED. Use update_withmask instead."""
        states_np = np.stack(ai_info)
        states = torch.from_numpy(states_np).float().to(self.device)

        actions_np = np.array(ai_actions)
        actions = torch.from_numpy(actions_np).long().view(-1, 1).to(self.device)

        rewards_np = np.array(-ai_score_delta + 1)
        rewards = torch.from_numpy(rewards_np).float().view(-1, 1).to(self.device)

        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float).view(-1, 1).to(self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        self.optimizer.zero_grad()
        total_loss = 0

        for i in range(len(returns)):
            state = states[i].unsqueeze(0)
            action = actions[i]
            return_ = returns[i]
            action_probs = self.policy_net(state)
            action_probs = torch.clamp(action_probs, min=1e-6, max=1.0)

            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action.squeeze(-1))

            loss = -log_prob * return_.detach()
            total_loss += loss

        average_loss = total_loss / 13
        average_loss.backward()
        self.optimizer.step()

        loss_value = average_loss.item()
        self.loss_log.append(loss_value)

        return loss_value

    def update_withmask(self, transition_dict):
        states_np = np.stack(transition_dict['states'])
        states = torch.from_numpy(states_np).float().to(self.device)

        actions_np = np.array(transition_dict['actions'])
        actions = torch.from_numpy(actions_np).long().view(-1, 1).to(self.device)

        rewards_np = np.array(transition_dict['rewards'])
        rewards = torch.from_numpy(rewards_np).float().view(-1, 1).to(self.device)

        masks_np = np.stack(transition_dict['masks'])
        masks = torch.from_numpy(masks_np).bool().to(self.device)

        # Compute discounted returns (Monte Carlo)
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float).view(-1, 1).to(self.device)

        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        self.optimizer.zero_grad()
        total_loss = 0

        for i in range(len(returns)):
            state = states[i].unsqueeze(0)
            action = actions[i]
            return_ = returns[i]
            mask = masks[i].unsqueeze(0)

            # Get raw policy output
            action_probs = self.policy_net(state)

            # Apply mask to zero out illegal actions
            action_probs = action_probs.masked_fill(~mask, 0)

            # Renormalize over legal actions only
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
            else:
                # Fallback: uniform over legal actions
                action_probs = mask.float() / mask.float().sum()

            # Clamp for numerical stability
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)

            # Create distribution and compute log probability
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action.squeeze(-1))

            # REINFORCE loss: negative log probability weighted by return
            loss = log_prob * return_.detach()
            total_loss += loss

        average_loss = total_loss / len(returns)
        average_loss.backward()
        self.optimizer.step()

        loss_value = average_loss.item()
        self.loss_log.append(loss_value)

        return loss_value

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
    
    def policy(self, player, player_info, actions, order) -> Card:
        return action_to_card(self.take_action(self, info_to_tensor(player_info), actions_to_mask(actions)))
    
    def train(self, game: Game, oppo_policy, episodes):
        for i in episodes:
            score, shot, ai_score_delta, ai_actions, ai_masks, ai_info = game.fight([self.policy] + oppo_policy, True, False, False)
            ai_actions = np.vectorize(card_to_action)(ai_actions)
            ai_masks = np.vecorize(actions_to_mask)(ai_masks)
            if shot:
                if score[0] == 0:
                    ai_score_delta = np.full(13, -3)
                else:
                    ai_score_delta = np.full(13, 2)
            else:
                ai_score_delta = np.ndarray(ai_score_delta)
            ai_info = np.vectorize(info_to_tensor)(ai_info)
            self.update_withoutmask(ai_score_delta, ai_actions, ai_masks, ai_info)

model = RF(163, 128, 52, 1e-5, 0.99, 'cuda')

def training_rl_policy(player, player_info, actions, order):
    global model
    action = model.take_action(info_to_tensor(player_info), actions_to_mask(actions))
    if player_info['points'] > 0:
        model.transition_dict['rewards'].append((0 - player_info['points'])/10)
    else:
        model.transition_dict['rewards'].append(1)
    model.transition_dict['actions'].append(action)
    model.transition_dict['states'].append(info_to_tensor(player_info))
    model.transition_dict['masks'].append(actions_to_mask(actions))
    return action_to_card(action)

def mirror_policy(player, player_info, actions, order):
    global mirror_model
    action = mirror_model.take_action(info_to_tensor(player_info), actions_to_mask(actions))
    return action_to_card(action)


def train(model):
    global mirror_model, meanlog
    for episode in range(500):
        if episode > 160:
            model.lr *= 0.98
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = model.lr
        print('----------------------------------------------------')
        print(f'episode: {episode}')
        model.transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'masks': []
        }
        if episode < 210 and episode % 50 == 0:
            mirror_model = copy.deepcopy(model)
            meanlog.append(np.mean(model.log[-20:]))
        points = game([rl_policy, mirror_policy, mirror_policy, sample_policy], True, False)
        model.transition_dict['rewards'][-1] += -points[0]
        loss = model.update_withoutmask(model.transition_dict)
        model.log.append(model.transition_dict['rewards'][-1])
        print(f'episode: {episode}, last_reward: {model.transition_dict["rewards"][-1]}, loss: {loss}')

def train_and_save():
    print("Start Training")
    train(model)
    print("Training Finished")
    pd.DataFrame(model.log).to_csv('./data/Zhuiy_rl_log.csv', index=False)
    pd.DataFrame(meanlog).to_csv('./data/Zhuiy_rl_meanlog.csv', index=False)
    pd.DataFrame(model.loss_log).to_csv('./data/Zhuiy_rl_loss.csv', index=False)

    model.save('./data/Zhuiy_rl_model.pth')

def evaluation(model, n=100):
    score = [0, 0, 0, 0]
    for i in range(n):
        points = game([rl_policy, sample_policy, sample_policy, sample_policy], True, False)
        score[0] += points[0]
        score[1] += points[1]
        score[2] += points[2]
        score[3] += points[3]
    return [s / n for s in score]

if __name__ == '__main__':
    model = RF(163, 128, 52, 1e-5, 0.99, 'cuda')
    Hearts = Game()

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
            model.load('./data/Zhuiy_rl_model.pth')
            print(evaluation(model, 500))
        else:
            print('kicking you off')
            exit(0)
    else:
        mirror_model = copy.deepcopy(model)
        if to_load == 'y':
            model.load('./data/Zhuiy_rl_model.pth')
            train_and_save()
            print(evaluation(model, 500))
        else:
            train_and_save()
            print(evaluation(model, 500))
    