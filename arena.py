from game import Game
from Zhuiy_approach.Zhuiy_rl import RF
from Zhuiy_sample.Zhuiy_sample_policy import sample_policy
import time

def real_player(player, player_info, actions, order):
    actions = sorted(actions)
    print(f"Your turn. Available actions: {actions}")
    while True:
        try:
            choice = int(input(f"Choose your action (0-{len(actions)-1}): "))
            if 0 <= choice < len(actions):
                return actions[choice]
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid number.")

Zhuiy = RF(163, 128, 64, 52, 1e-5, 0.99, 'cuda')
Zhuiy.load('./Zhuiy_approach/data/policy_net', './Zhuiy_approach/data/value_net')

Hearts = Game()

policy_list = ['sample_policy', 'Zhuiy.policy']
policies = [sample_policy, Zhuiy.policy]

while True:
    a = input('Is there a REAL PERSON? (Y/n)')
    if not a or a in ['Y', 'y', 'n']:
        break
    else:
        print('kicking you off')
        time.sleep(2)
        exit(0)

if a == 'n':
    choice = []
    name = []
    for i in range(4):
        print(f'Select your policy: {policy_list} (0~{len(policy_list) - 1})')
        while True:
            k = input()
            if k in range(len(policy_list)):
                break
            else:
                print('WRONG input')
        choice.append(policies[k])
        name.append(policy_list[k])
    print(f'your Hearters: {name}')
    Hearts.fight(choice)
    print(f'thank you Hearters: {name}')
else:
    print('You are player_0......')
    time.sleep(1)
    choice = []
    name = []
    for i in range(3):
        print(f'Select your policy: {policy_list} (0~{len(policy_list) - 1})')
        while True:
            k = int(input())
            if k in range(len(policy_list)):
                break
            else:
                print('WRONG input')
        choice.append(policies[k])
        name.append(policy_list[k])
    print(f'the Hearters: You and {name}')

    while True:
        a = input('Want some challenge? (y/N)')
        if not a or a in ['y', 'N', 'n']:
            break
        else:
            print('WRONG input')

    trick = False
    if a == 'y':
        trick = True
    
    Hearts.fight([real_player] + choice, False, True, trick)

    print(f'thank you Hearters: You and {name}')



