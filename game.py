from action_distribution import ActionDistribution
from utils import *
from copy import copy
from math import log


class State:

    def __init__(self, board_size, stones, komi, board=None, player=0, n_moves=0):
        self.board_size = board_size
        self.stones = stones
        self.komi = komi    # komi is subtracted to sente
        self.board = board  # 2 2 2 0 2 2 2 0
        self.player = player
        self.n_moves = n_moves

        if self.board is None:
            self.set_initial_state()

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
        if self.is_game_over():
            return step(self.board[self.board_size] - self.board[self.board_size * 2 + 1] - self.komi)

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
        s = '>' if self.player else '<'
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

    def __init__(self, state=None, board_size=6, stones=4, komi=.5):
        self.state = state
        self.board_size = board_size
        self.stones = stones
        self.komi = komi
        self.kifu = []
        self.outcome = None
        self.values = []    # record of subjective expectation of result
        self.players = []   # record of who was on duty

    def save_info(self, value):
        self.players += [self.state.player]
        self.kifu += [self.state]
        self.values += [value]

    def play(self, *players):

        # initialize state
        if self.state is None:
            self.state = State(board_size=self.board_size, stones=self.stones, komi=self.komi)

        # activate bots
        for i, p in enumerate(players):
            p.open(self.state, player_id=i)

        # game loop
        while not self.state.is_game_over():
            # get move from current player
            policy, value = players[self.state.player].get_move(self.state)
            policy *= self.state.legal_moves()  # filter out illegal moves
            move = policy.choose_move()

            # distribute move to all players
            for p in players:
                p.set_move(move)

            self.save_info(value)   # before performing the move
            self.state = self.state.make_move(move)

        self.outcome = self.state.result()

        for p in players:
            p.close(self.state, self.outcome)

        self.save_info(self.outcome)
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
        kifu = [self.kifu[i] for i in range(len(self) - 1) if self.players[i] == player]
        kifu += [self.kifu[len(self) - 1]]
        values = [self.values[i] for i in range(len(self) - 1) if self.players[i] == player]
        values += [self.values[len(self) - 1]]
        if player == 1:
            values = [1-i for i in values]
        return {'kifu': kifu, 'values': values}

    def get_edited_player_data(self, player, k):
        data = self.get_player_data(player)
        data['values'] = adjust_values(data['values'], k=k)
        return data


def compute_elo(agent_1, agent_2, board_size, stones, komi):

    def elo(n_games, n_wins, c_elo=1/400):
        score = (n_wins + 1) / (n_games + 2)
        return -log(1 / score - 1) / c_elo

    n0 = w0 = n1 = w1 = 0

    while True:
        game = Game(board_size=board_size, stones=stones, komi=komi)
        game.play(agent_1, agent_2)
        n0 += 1
        w0 += game.outcome

        game = Game(board_size=board_size, stones=stones, komi=komi)
        game.play(agent_2, agent_1)
        n1 += 1
        w1 += 1 - game.outcome

        yield elo(n0, w0), elo(n1, w1), elo(n0 + n1, w0+w1)
