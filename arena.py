from game import Game
from Zhuiy_approach.Zhuiy_rl import RF
from Zhuiy_sample.Zhuiy_sample_policy import sample_policy

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

Zhuiy = RF(163, 128, 32, 52, 1e-5, 0.99, 'cuda')
Zhuiy.load('./Zhuiy_approach/data/policy_net')

Hearts = Game()
#game([sample_policy, sample_policy, sample_policy, sample_policy])
Hearts.fight([real_player, Zhuiy.policy, sample_policy, sample_policy], False, True, False)