from game import Game
import random
def sample_policy(player, player_info, actions, order):
    return random.choice(actions)

if __name__ == '__main__':
    sample = Game()
    sample.fight([sample_policy, sample_policy, sample_policy, sample_policy])