from game import game
from Zhuiy_sample_policy import sample_policy
from Zhuiy_rl import rl_policy

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

#game([sample_policy, sample_policy, sample_policy, sample_policy])
game([real_player, rl_policy, sample_policy, sample_policy], False)