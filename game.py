
try:
    from state import State
    from utils import *
except ImportError:
    from .state import State
    from .utils import *


class Game:

    def __init__(self, state=None, board_size=6, stones=4, komi=.5, show=False, do_record=False):

        # data
        self.state = state
        self.outcome = None
        self.kifu = []
        self.policy_record = []     # record of the actual visits of the first nodes
        self.values = []  # record of subjective expectation of result
        self.players = []  # record of who was on duty

        # settings
        self.board_size = board_size
        self.stones = stones
        self.komi = komi
        self.show = show
        self.do_record = do_record

    def save_info(self, policy, value):
        if self.do_record:
            self.players += [self.state.player]
            self.kifu += [self.state]
            self.policy_record += [policy]
            self.values += [value]

    def reset(self):
        """prepare the class instance for a new game"""
        self.state = None
        self.outcome = None
        self.kifu = []
        self.values = []
        self.policy_record = []
        self.players = []

    def game_loop(self, players):
        while not self.state.is_game_over():
            if self.show:
                print(self.state)

            # get move from current player
            policy, value = players[self.state.player].get_move(self.state)
            policy *= self.state.legal_moves()  # filter out illegal moves
            move = policy.choose_move()

            # distribute move to all players
            for p in players:
                p.set_move(move)

            if self.do_record:
                self.save_info(policy, value)   # before performing the move

            self.state = self.state.make_move(move)

    def _print_title(self, players):
        if self.show:
            print('\n')
            print(f' < {players[0]}')
            print(f' > {players[1]}')
            print()

    def _print_ending(self):
        if self.show:
            print(self.state)
            print()
            if self.outcome == 1:
                print(' <  wins')
            elif self.outcome == 0:
                print('  > wins')
            print('\n')

    def _init_state(self):
        if self.state is None:
            self.state = State(board_size=self.board_size, stones=self.stones, komi=self.komi)

    def _activate_bots(self, players):
        for i, p in enumerate(players):
            p.open(self.state, player_id=i)

    def _close_bots(self, players):
        for p in players:
            p.close(self.state, self.outcome)

    def play(self, players):
        assert len(players) == 2
        self._print_title(players)
        self._init_state()
        self._activate_bots(players)
        self.game_loop(players)
        self.outcome = self.state.get_result()
        self._close_bots(players)
        # self.save_info(self.state.get_subjective_result())    # todo last state?
        self._print_ending()
        return self.outcome

    def congrats(self):
        if self.outcome == 1:
            return '\nSente won!\n'
        elif self.outcome == 0:
            return '\nGote won!\n'

    def __repr__(self):
        out = ''
        for s in self.kifu:
            out += f'{s.n_moves:4})' + ' \t\t'
            out += f'{s}\n'
        return out + self.congrats()

    def __len__(self):
        return len(self.kifu)

    def get_player_data(self, player):
        """collect subjective data relevant to one player, assuming game over or resignation"""
        if len(self.kifu) == 0:
            raise ValueError('No data collected!')

        kifu = [s for s in self.kifu if s.player == player]

        n_moves = [i.n_moves for i in kifu]
        players = [i.player for i in kifu]
        values = [self.values[i] for i in n_moves]

        outcome = self.outcome
        if self.outcome is not None:
            if player == 1:
                outcome = 1 - outcome

        state_repr = [i.representation() for i in kifu]

        policy_record = [[i for i in self.policy_record[i].normalize()] for i in n_moves]

        return {'kifu': kifu, 'values': values, 'policy_record': policy_record, 'players': players,
                'n_moves': n_moves, 'state_repr': state_repr, 'outcome': outcome}

    def get_edited_player_data(self, player, k):
        """list of (board_representation, adjusted_value)
        adjusted_value predicts better the actual outcome of that game
        """
        data = self.get_player_data(player)
        assert data['outcome'] is not None
        values = adjust_values(data['values'], target=data['outcome'], k=k)
        return [(data['state_repr'][i], data['policy_record'][i], values[i]) for i in range(len(values))]

    def get_training_data(self, k=3.):
        """data ready for training"""
        return self.get_edited_player_data(0, k) + self.get_edited_player_data(1, k)
