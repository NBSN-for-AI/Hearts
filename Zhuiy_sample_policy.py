from game import game
def sample_policy(player, player_info, actions, order):
    # Example policy : always play the first card available
    return sorted(actions)[0]

if __name__ == '__main__':
    game([sample_policy, sample_policy, sample_policy, sample_policy])