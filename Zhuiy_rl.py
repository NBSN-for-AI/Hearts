'''
Author: Zhuiy
'''

from game import game, Card, Suit
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy_net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Policy_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        return F.softmax(self.fc2(F.relu(self.fc1(state))), dim=-1)
    
class RF:
    transition_dict = {
    'states': [],
    'actions': [],
    'rewards': []
    }
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, device):
        self.policy_net = Policy_net(state_dim, hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.gamma = gamma
        self.device = device
        self.lr = lr

    def take_action(self, state, mask=None):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
            probs = probs.masked_fill(~mask, 1e-8)
            probs = probs / probs.sum()
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G
            state = states[i].view(1, -1)
            action = actions[i].item()
            probs = self.policy_net.forward(state)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(action).to(self.device))
            reward = -log_prob * G
            reward.backward()
        self.optimizer.step()

def info_to_tensor(info):
    hand = info['hand']
    hand_tensor = torch.zeros(52)
    for card in hand:
        index = (card.suit.value) * 13 + card.rank
        hand_tensor[index] = 1
    points_tensor = torch.tensor([info['points']], dtype=torch.float)
    table_tensor = torch.zeros(52)
    for card, _ in info['table']:
        index = (card.suit.value) * 13 + card.rank
        table_tensor[index] = 1
    current_suit_tensor = torch.zeros(4)
    if info['current_suit'] is not None:
        current_suit_tensor[info['current_suit'].value] = 1
    else:
        current_suit_tensor = torch.zeros(4)
    current_table_tensor = torch.zeros(52)
    for card, _ in info['current_table']:
        index = (card.suit.value) * 13 + card.rank
        current_table_tensor[index] = 1
    hearts_broken_tensor = torch.tensor([1.0 if info['hearts_broken'] else 0.0], dtype=torch.float)
    piggy_pulled_tensor = torch.tensor([1.0 if info['piggy_pulled'] else 0.0], dtype=torch.float)
    state_tensor = torch.cat([hand_tensor, points_tensor, table_tensor, current_suit_tensor, current_table_tensor, hearts_broken_tensor, piggy_pulled_tensor], dim=0)
    return state_tensor.numpy()

def actions_to_mask(actions):
    mask = torch.zeros(52, dtype=torch.bool)
    for card in actions:
        index = (card.suit.value) * 13 + card.rank
        mask[index] = True
    return mask.numpy()

def actions_to_tensor(actions):
    action_tensor = torch.zeros(52)
    for card in actions:
        index = (card.suit.value) * 13 + card.rank
        action_tensor[index] = 1
    return action_tensor.numpy()

def action_to_card(action):
    suit = action // 13
    rank = action % 13
    return Card(Suit(suit), rank)

model = RF(165, 128, 52, 1e-3, 0.99, 'gpu')
mirror_model = model

def rl_policy(player, player_info, actions, order):
    action = model.take_action(player_info, actions_to_tensor(actions), actions_to_mask(actions))
    model.transition_dict['rewards'].append(0)
    model.transition_dict['actions'].append(action)
    model.transition_dict['states'].append(info_to_tensor(player_info))
    return action_to_card(action)

def mirror_policy(player, player_info, actions, order):
    action = mirror_model.take_action(player_info, actions_to_tensor(actions), actions_to_mask(actions))
    return action_to_card(action)

def train(model):
    global mirror_model
    for episode in range(100):
        model.transition_dict = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        model.transition_dict['rewards'][-1] = game([rl_policy, mirror_policy, mirror_policy, mirror_policy])[0]
        model.update(model.transition_dict)
        mirror_model = model
        print(f'episode: {episode}, reward: {model.transition_dict["rewards"][-1]}')

