from action_distribution import ActionDistribution
from math import tanh
from tree import Tree


class Agent:    # todo filter illegal moves here?
    """
    Agent(state: State) -> (policy, value)
    illegal moves are automatically filtered
    """
    def open(self, initial_state, player_id):
        pass

    def set_move(self, move):
        pass

    def get_move(self, state):
        raise NotImplementedError

    def close(self, final_state, outcome):
        pass

    def __call__(self, state):
        return self.get_move(state)


# ----- BOTS ----- #


class RandomAgent(Agent):
    """random moves
    Elo = 0(0)"""
    def get_move(self, state):
        v = [1 for _ in range(state.board_size)]
        return ActionDistribution(v), .5


class SimpleAgent(Agent):
    """simple heuristic
        Elo = 1950(50)
        """
    def __init__(self, x1=5., m1=50, x2=.25, m2=1., k_value=6):
        self.x1 = x1
        self.m1 = m1
        self.x2 = x2
        self.m2 = m2
        self.k_value = k_value

    def get_move(self, state):
        v = state.representation()
        n = state.board_size
        value = v[n] - v[n * 2 + 1] - state.komi
        value = ((tanh(value / self.k_value) + 1) / 2)
        v = [(n - i + self.x1) * self.m1 if n - i == v[i] else (i + self.x2) * self.m2 for i in range(n)]
        return ActionDistribution(v), value


class HumanAgent(Agent):
    """human inputs moves"""
    def get_move(self, state):
        t = state.legal_moves().get_non_zero()
        print()
        print(state)
        while True:
            n = input('>>> ')
            try:
                n = int(n)
                if n < 0:
                    n += state.board_size
                else:
                    n -= 1
                if n in t:
                    v = [1 if i == n else 0 for i in range(state.board_size)]
                    return ActionDistribution(v), .5
            except ValueError:
                pass

    def close(self, state, result):
        print()
        print(state)
        print('\n')


class TreeAgent(Agent):
    """"""
    def __init__(self, fast_agent, n_rollouts):
        self.tree = None
        self.fast_agent = fast_agent
        self.n_rollouts = n_rollouts

    def open(self, initial_state, player_id):
        self.tree = Tree(initial_state, self.fast_agent, player_id)

    def set_move(self, move: ActionDistribution):
        n = move.get_non_zero()[0]
        self.tree.re_plant(n)

    def get_move(self, state):
        return self.tree.get_policy_and_value(self.n_rollouts)