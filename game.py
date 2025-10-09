'''
authour : Zhuiy & Copilot
'''

from dataclasses import dataclass, field
from enum import IntEnum
import random
from typing import List, Callable, Optional, Tuple

class Suit(IntEnum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: int  # 1-13
    def __repr__(self):
        suit_symbols = {Suit.HEARTS: 'H', Suit.DIAMONDS: 'D', Suit.CLUBS: 'C', Suit.SPADES: 'S'}
        return f"{suit_symbols[self.suit]}{self.rank}"
    
    def __lt__(self, other):
        if self.suit == other.suit:
            return self.rank < other.rank
        return self.suit < other.suit

@dataclass
class Player:
    hand: List[Card] = field(default_factory=list)
    points: int = 0
    table: List[Card] = field(default_factory=list)

@dataclass
class GameState:
    players: List[Player] = field(default_factory=lambda: [Player() for _ in range(4)])
    deck: List[Card] = field(default_factory=list)
    table: List[Tuple[Card, int]] = field(default_factory=list)
    current_table: List[Tuple[Card, int]] = field(default_factory=list)
    current_suit: Suit = None
    hearts_broken: bool = False
    piggy_pulled: bool = False

    def reset(self):
        current_suits = None
        hearts_broken = False
        piggy_pulled = False
        self.deck = [Card(suit, rank) for suit in Suit for rank in range(1, 14)]
        random.shuffle(self.deck)
        for player in self.players:
            player.hand = []
            player.points = 0
            player.table = []
        for i, card in enumerate(self.deck):
            self.players[i % 4].hand.append(card)
        self.rounds = 1

    def get_first_player(self) -> int:
        for i, player in enumerate(self.players):
            for card in player.hand:
                if card.suit == Suit.CLUBS and card.rank == 2:
                    return i
        return 0

def card_value(card: Card) -> int:
    if card.suit == Suit.HEARTS:
        return 1
    elif card.suit == Suit.SPADES and card.rank == 12:
        return 13
    return 0

def available_actions(player: Player, suit: Optional[Suit], is_first_round: bool, scored: bool) -> List[Card]:
    """
    :param player: 当前玩家
    :param suit: 本轮首出花色
    :param is_first_round: 是否第一轮
    :param scored: 是否已经有人得分（只要有一人分数>0）
    """
    if is_first_round:
        if suit is None:
            return [Card(Suit.CLUBS, 2)]
        else:
            suited = [c for c in player.hand if c.suit == suit]
            if suited:
                return sorted(suited)
            non_hearts = [c for c in player.hand if c.suit != Suit.HEARTS]
            if not scored and non_hearts:
                return sorted([c for c in non_hearts if not (c.suit == Suit.SPADES and c.rank == 12)] or non_hearts)
            return sorted(player.hand)
    else:
        if suit is None:
            if not scored:
                non_hearts = [c for c in player.hand if c.suit != Suit.HEARTS]
                if non_hearts:
                    return sorted(non_hearts)
                else:
                    return sorted(player.hand)
            else:
                return sorted(player.hand)
        else:
            suited = [c for c in player.hand if c.suit == suit]
            if suited:
                return sorted(suited)
            non_hearts = [c for c in player.hand if c.suit != Suit.HEARTS]
            if not scored and non_hearts:
                return sorted(non_hearts)
            return sorted(player.hand)
# availavle_actions are sorted

def end_game(state: GameState):
    scores = [p.points for p in state.players]
    if 26 in scores:
        for p in state.players:
            if p.points != 26:
                p.points = 26
            else:
                p.points = 0
    return [p.points for p in state.players]

def fight(policies: List[Callable], training:bool=True, display:bool=True) -> List[int]:
    state = GameState()
    state.reset()
    first_player = state.get_first_player()
    while state.rounds <= 13:
        first_player = play_round(state, first_player, policies, training, display)
        if display:
            print('points:', [p.points for p in state.players])
            print('===================================')
    print('final points:', end_game(state))
    return [p.points for p in state.players]

class game:
    def __init__(self):
        self.gamestate = GameState()
        self.rounds = 0

    def reset(self, trick=False):
        self.gamestate.reset()
        if trick:
            # pull the pig into player_0's hand!
            for i in range(4):
                if Card(3, 12) in self.gamestate.players[i].hand:
                    self.gamestate.players[i].hand.remove(Card(3, 12))
                    self.gamestate.players[i].hand.append(self.gamestate.players[0].hand.pop())
                    self.gamestate.players[0].hand.append(Card(3, 12))
                    break
                
    def play_round(self, first_player: int, policies: List[Callable]) -> int:
        self.gamestate.current_table = []
        self.current_suit = None
        print(f'round {self.gamestate.rounds}:')
        print()
        for i, player in enumerate(self.gamestate.players):
            print(f'player {i} hand:', sorted(player.hand))
        print()
        print('current points:', [p.points for p in self.gamestate.players])
        print()
        table: List[(Card, int)] = []
        scored = any(p.points > 0 for p in self.gamestate.players)
        for i in range(4):
            player_idx = (first_player + i) % 4
            player = self.gamestate.players[player_idx]
            is_first = (self.gamestate.rounds == 1)
            print(f'player {player_idx}\'s avilable actions: {sorted(available_actions(player, self.gamestate.current_suit, is_first, scored))}')
            actions = available_actions(player, self.gamestate.current_suit, is_first, scored)
            card = policies[player_idx](player, player_info(player, self.gamestate), actions, i)
            if not self.gamestate.piggy_pulled and (card.suit == Suit.SPADES and card.rank == 12):
                self.gamestate.piggy_pulled = True
                self.gamestate.hearts_broken = True
            if card.suit == Suit.HEARTS and not self.gamestate.hearts_broken:
                self.gamestate.hearts_broken = True
            if card not in actions:
                raise ValueError(f'player {player_idx} played invalid card {card}, available actions: {actions}')
            player.hand.remove(card)
            player.table.append(card)
            table.append((card, player_idx))
            self.gamestate.table.append((card, player_idx))
            self.gamestate.current_table.append((card, player_idx))
            print(f'player {player_idx} plays {card}')
            if i == 0:
                self.gamestate.current_suit = card.suit

        lead_cards = [(c, idx) for c, idx in table if c.suit == self.gamestate.current_suit]
        winner_card, winner_idx = max(lead_cards, key=lambda x: (x[0].rank - 2) % 13)
        value = sum(card_value(c) for c, _ in table)
        self.gamestate.players[winner_idx].points += value
        self.gamestate.rounds += 1
        print(f'player {winner_idx} wins the round and gets {value} points')
        print('-----------------------------------')
        print()
        return winner_idx

    def play_round_training(self, first_player: int, policies: List[Callable]) -> tuple[int, int, Card, List[Card]]:
        self.gamestate.current_table = []
        self.gamestate.current_suit = None
        table: List[(Card, int)] = []
        scored = any(p.points > 0 for p in self.gamestate.players)
        for i in range(4):
            player_idx = (first_player + i) % 4
            player = self.gamestate.players[player_idx]
            is_first = (self.gamestate.rounds == 1)
            actions = available_actions(player, self.gamestate.current_suit, is_first, scored)
            card = policies[player_idx](player, player_info(player, self.gamestate), actions, i)
            if player_idx == 0:
                ai_action = card
            if not self.gamestate.piggy_pulled and (card.suit == Suit.SPADES and card.rank == 12):
                self.gamestate.piggy_pulled = True
                self.gamestate.hearts_broken = True
            if card.suit == Suit.HEARTS and not self.gamestate.hearts_broken:
                self.gamestate.hearts_broken = True
            if card not in actions:
                raise ValueError(f'player {player_idx} played invalid card {card}, available actions: {actions}')
            player.hand.remove(card)
            player.table.append(card)
            table.append((card, player_idx))
            self.gamestate.table.append((card, player_idx))
            self.gamestate.current_table.append((card, player_idx))
            if i == 0:
                self.gamestate.current_suit = card.suit
        
        lead_cards = [(c, idx) for c, idx in table if c.suit == self.gamestate.current_suit]
        winner_card, winner_idx = max(lead_cards, key=lambda x: (x[0].rank - 2) % 13)
        value = sum(card_value(c) for c, _ in table)
        ai_socre = 0
        if winner_idx == 0:
            ai_score = value
        self.gamestate.players[winner_idx].points += value
        self.gamestate.rounds += 1
        return winner_idx, ai_score, ai_action, actions

    def play_round_human_0(self, first_player: int, policies: List[Callable]) -> int:
        self.gamestate.current_table = []
        self.gamestate.current_suit = None
        print(f'round {self.gamestate.rounds}:')
        print()
        print('your hand:', sorted(self.gamestate.players[0].hand))
        print()
        print('current points:', [p.points for p in self.gamestate.players])
        print()
        table: List[(Card, int)] = []
        scored = any(p.points > 0 for p in self.gamestate.players)
        for i in range(4):
            player_idx = (first_player + i) % 4
            player = self.gamestate.players[player_idx]
            is_first = (self.gamestate.rounds == 1)
            actions = available_actions(player, self.gamestate.current_suit, is_first, scored)
            card = policies[player_idx](player, player_info(player, self.gamestate), actions, i)
            if not self.gamestate.piggy_pulled and (card.suit == Suit.SPADES and card.rank == 12):
                self.gamestate.piggy_pulled = True
                self.gamestate.hearts_broken = True
            if card.suit == Suit.HEARTS and not self.gamestate.hearts_broken:
                self.gamestate.hearts_broken = True
            if card not in actions:
                raise ValueError(f'player {player_idx} played invalid card {card}, available actions: {actions}')
            player.hand.remove(card)
            player.table.append(card)
            table.append((card, player_idx))
            self.gamestate.table.append((card, player_idx))
            self.gamestate.current_table.append((card, player_idx))
            print(f'player {player_idx} plays {card}')
            if i == 0:
                self.gamestate.current_suit = card.suit
        
        lead_cards = [(c, idx) for c, idx in table if c.suit == self.gamestate.current_suit]
        winner_card, winner_idx = max(lead_cards, key=lambda x: (x[0].rank - 2) % 13)
        value = sum(card_value(c) for c, _ in table)
        self.gamestate.players[winner_idx].points += value
        self.gamestate.rounds += 1
        print(f'player {winner_idx} wins the round and gets {value} points')
        print('-----------------------------------')
        print()
        return winner_idx

    def end_game(self):
        scores = [p.points for p in self.gamestate.players]
        if 26 in scores:
            for p in self.gamestate.players:
                if p.points != 26:
                    p.points = 26
                else:
                    p.points = 0
            return [p.points for p in self.gamestate.players], True
        return scores, False
    
    def end_game_error(self, text=None):
        print(f'error: text')
        return [p.points for p in self.gamestate.players]
    
    def fight(self, policies: List[Callable], training:bool=False, human_0:bool=False, trick=False) -> List[int]:
        self.reset(self, trick)
        try:
            first_player = self.gamestate.get_first_player()
            if training:
                ai_score_delta = []
                ai_actions = []
                ai_masks = []
                while self.rounds <= 13:
                    first_player = self.play_round_training(self.gamestate, first_player, policies)
                score, shot = self.end_game(self)
                print('final points:', score)
                return score, shot, ai_score_delta, ai_actions, ai_masks


def player_info(player: Player, state: GameState) -> dict:
    return {
        'hand': player.hand,
        'points': player.points,
        'table': state.table,
        'current_table': state.current_table,
        'current_suit': state.current_suit,
        'hearts_broken': state.hearts_broken,
        'piggy_pulled': state.piggy_pulled,
        'rounds': state.rounds,
        'current_order': len(state.current_table)
    }

if __name__ == '__main__':
    def random_policy(player: Player, player_info: Callable, actions: List[Card], order: int) -> Card:
        return random.choice(actions)
    game([random_policy]*4)
