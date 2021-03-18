from copy import copy

try:
    from action_distribution import ActionDistribution
    from utils import *
except ImportError:
    from .action_distribution import ActionDistribution
    from .utils import *


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

    def get_result(self):
        if self.outcome is None:
            if self.is_game_over():
                self.outcome = step(self.board[self.board_size] - self.board[self.board_size * 2 + 1] - self.komi)
        return self.outcome

    def get_subjective_result(self):
        return self.outcome if self.player == 0 else 1 - self.outcome

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
            s += f'{i:{space_1+1}}' if i else ' ' * space_1 + '.'
            s += ' ' * space_1
        s += f'{self.board[n]:{space_2+1}}' + ' ' * (space_2+2)
        for i in self.board[n+1:2*n+1]:
            s += f'{i:{space_1+1}}' if i else ' ' * space_1 + '.'
            s += ' ' * space_1
        s += f'{self.board[2*n+1]:{space_2+1}}'
        return s
