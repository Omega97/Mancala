from action_distribution import ActionDistribution
from utils import *
from copy import copy
from math import log

# todo if resign -> update state outcome


class State:

    def __init__(self, stones, board_size=None, komi=.5, board=None, player=0, n_moves=0):
        self.board_size = board_size
        self.stones = stones
        self.komi = komi    # komi is subtracted to sente
        self.board = board  # 2 2 2 0 2 2 2 0
        self.player = player
        self.n_moves = n_moves
        self.outcome = None

        if self.board is None:
            self.set_initial_state()
        else:
            self.board_size = len(self.board) // 2 - 1

    def __copy__(self):
        return State(board_size=self.board_size, stones=self.stones,
                     komi=self.komi, board=copy(self.board), player=self.player)

    def set_initial_state(self):
        self.board = ([self.stones for _ in range(self.board_size)] + [0]) * 2

    def legal_moves(self):
        n = self.board_size + 1
        v = [1 if self.board[i + n * self.player] else 0 for i in range(self.board_size)]
        return ActionDistribution(v)

    def make_move(self, move: ActionDistribution):
        new_state = copy(self)
        new_state.n_moves = self.n_moves + 1

        n = move.get_move_index()

        if new_state.player:
            n += new_state.board_size + 1

        if new_state.board[n] == 0:
            raise ValueError(f'Illegal move, square {n} is empty!')

        last = None
        for i in mancala_gen(start=n, n_stones=new_state.board[n], board_size=new_state.board_size):
            new_state.board[i] += 1
            last = i
        if not last == self.board_size + (new_state.board_size + 1) * new_state.player:
            new_state.player = 1 - new_state.player
        new_state.board[n] = 0
        return new_state

    def is_game_over(self):
        return self.legal_moves().norm() == 0

    def result(self):
        if self.outcome is None:
            if self.is_game_over():
                self.outcome = step(self.board[self.board_size] - self.board[self.board_size * 2 + 1] - self.komi)
        return self.outcome

    def representation(self) -> list:
        """representation is how bots should view the game state
        It is subjective, i.e. the player on duty always faces his side of the board
        """
        if self.player == 0:
            v = self.board
        else:
            n = self.board_size + 1
            v = self.board[n:] + self.board[:n]
        return v + [self.player]

    def __repr__(self, space_1=1, space_2=2):
        s = ''
        s += f'{self.n_moves:3})'
        s += ' ' * 5
        s += ' >' if self.player else '< '
        s += '   '
        n = self.board_size
        for i in self.board[:n]:
            s += f'{i:{space_1+1}}' + ' ' * space_1
        s += f'{self.board[n]:{space_2+1}}' + ' ' * (space_2+2)
        for i in self.board[n+1:2*n+1]:
            s += f'{i:{space_1+1}}' + ' ' * space_1
        s += f'{self.board[2*n+1]:{space_2+1}}'
        return s


class Game:

    def __init__(self, state=None, board_size=6, stones=4, komi=.5, show=False, do_record=False):

        # data
        self.state = state
        self.outcome = None
        self.kifu = []
        self.values = []  # record of subjective expectation of result
        self.players = []  # record of who was on duty

        # settings
        self.board_size = board_size
        self.stones = stones
        self.komi = komi
        self.show = show
        self.do_record = do_record

    def save_info(self, value):
        if self.do_record:
            self.players += [self.state.player]
            self.kifu += [self.state]
            self.values += [value]

    def reset(self):
        self.state = None
        self.outcome = None
        self.kifu = []
        self.values = []
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
                self.save_info(value)   # before performing the move

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
        self.outcome = self.state.result()
        self._close_bots(players)
        self.save_info(self.outcome)
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
        """collect data relevant to a player, assuming game over or resignation"""
        if len(self.kifu) == 0:
            raise ValueError('No data collected!')

        kifu = [s for s in self.kifu if (s.player == player or s.outcome is not None)]

        n_moves = [i.n_moves for i in kifu]
        players = [i.player for i in kifu]
        values = [self.values[i] for i in n_moves]

        outcome = kifu[-1].result()
        if outcome is not None:
            if player == 1:
                outcome = 1 - outcome     # <<<
            values[-1] = outcome

        return {'kifu': kifu, 'values': values, 'players': players, 'n_moves': n_moves}

    def get_edited_player_data(self, player, k):
        data = self.get_player_data(player)
        data['values'] = adjust_values(data['values'], k=k)
        return data


def compute_elo(agents, board_size, stones, komi, show=True,
                n_won_sente=0, n_lost_sente=0, n_won_gote=0, n_lost_gote=0):

    assert len(agents) == 2

    def elo(n_won, n_lost, c_elo=1/400):
        score = (n_won + 1) / (n_won + n_lost + 2)
        return -log(1 / score - 1) / c_elo

    def gen(w_0, l_0, w_1, l_1):
        while True:
            game = Game(board_size=board_size, stones=stones, komi=komi, show=show)
            game.play(*agents)

            if game.outcome == 1:
                w_0 += 1
            elif game.outcome == 0:
                l_0 += 1

            yield w_0, l_0, w_1, l_1

            game = Game(board_size=board_size, stones=stones, komi=komi, show=show)
            game.play(*reversed(agents))

            if game.outcome == 1:
                l_1 += 1
            elif game.outcome == 0:
                w_1 += 1

            yield w_0, l_0, w_1, l_1

    for w0, l0, w1, l1 in gen(n_won_sente, n_lost_sente, n_won_gote, n_lost_gote):

        yield {'n_won_sente': w0,
               'n_lost_sente': l0,
               'elo_sente': elo(w0, l0),
               'n_won_gote': w1,
               'n_lost_gote': l1,
               'elo_gote': elo(w1, l1),
               'n_won': w0 + w1,
               'n_lost': l0 + l1,
               'elo': elo(w0 + w1, l0 + l1),
               }
