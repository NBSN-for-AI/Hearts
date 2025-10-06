from game import game
from Zhuiy_sample_policy import sample_policy

def real_player(player, player_info, actions, order):
    print(f"Your turn. Available actions: {sorted(actions)}")
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
game([real_player, sample_policy, sample_policy, sample_policy], False)