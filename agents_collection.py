from math import tanh

try:
    from action_distribution import ActionDistribution
    from agent import Agent
    from tree import Tree
except ImportError:
    from .action_distribution import ActionDistribution
    from .agent import Agent
    from .tree import Tree


class RandomAgent(Agent):
    """random moves
    Elo = 0"""
    def get_move(self, state):
        v = [1 for _ in range(state.board_size)]
        return ActionDistribution(v), .5

    def __repr__(self):
        return 'RandomAgent'


class SimpleAgent(Agent):
    """simple heuristic
        Elo = 850
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

    def __repr__(self):
        return 'SimpleAgent'


class HumanAgent(Agent):
    """human inputs moves"""
    def get_move(self, state):
        t = state.legal_moves().get_non_zero()
        print()
        # print(state)
        while True:
            n = input('>>> ')
            print()
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

    def __repr__(self):
        return 'Human'


class TreeAgent(Agent):
    """ use the Monte-Carlo tree-search
    - use core_agent for the policy
    - use fast agent to do the roll-outs

    TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=100) --> Elo = 1350
    """
    def __init__(self, core_agent, fast_agent, n_rollouts, k_focus_branch=1., k_focus_decision=1., heuristic_par=1.5):
        self.tree = None
        self.core_agent = core_agent
        self.fast_agent = fast_agent
        self.n_rollouts = n_rollouts
        self.k_focus_branch = k_focus_branch
        self.k_focus_decision = k_focus_decision
        self.heuristic_par = heuristic_par

    def open(self, initial_state, player_id):
        self.tree = Tree(initial_state, self.core_agent, self.fast_agent, player_id, k_focus_branch=self.k_focus_branch,
                         k_focus_decision=self.k_focus_decision, heuristic_par=self.heuristic_par)
        self.tree.search(n_rollouts=self.n_rollouts)

    def set_move(self, move: ActionDistribution):
        n = move.get_non_zero()[0]
        self.tree.re_plant(n)

    def get_move(self, state):
        self.tree.search(n_rollouts=self.n_rollouts)
        return self.tree.get_final_policy_and_value()

    def get_raw_policy(self):
        """after the search"""
        return self.tree.raw_policy

    def __repr__(self):
        return f'TreeAgent({self.core_agent}, {self.fast_agent}, n={self.n_rollouts}, ' \
            f'k_fd={self.k_focus_decision:.3f}, k_b={self.k_focus_branch:.3f})'


def custom_agent(fun, k_focus=1., name='custom agent'):
    """ build custom agent class using a function
    :param fun: state representation -> (policy, value)
    :param k_focus: 0 = keep policy unchanged, 1 = keep only max
    :param name: name of the agent
    :return: custom agent class
    """
    class CustomAgent(Agent):
        def __init__(self):
            """
            k = 0 -> don't change
            k = 1 -> keep max
            """
            self.k = k_focus

        def get_move(self, state):
            *policy, value = fun(state.representation())
            policy = ActionDistribution(policy)
            policy *= state.legal_moves()
            policy = policy.focus(self.k)
            return policy, value

        def __repr__(self):
            return name

    return CustomAgent


def neural_net_agent(function, n_rollouts, k_focus_branch=1.,
                     k_focus_move=1., k_focus_rollout=1., k_focus_decision=1.):
    core_agent = custom_agent(function, k_focus_move)()
    fast_agent = custom_agent(function, k_focus_rollout)()

    def repr_(self):
        return f'NeuralNetAgent({self.core_agent}, {self.fast_agent}, n={self.n_rollouts}, ' \
            f'k_fm={self.k_focus_decision:.2f}, k_b={self.k_branch:.2f})'

    agent = TreeAgent(core_agent, fast_agent, n_rollouts,
                      k_focus_branch=k_focus_branch, k_focus_decision=k_focus_decision)

    agent.__repr__ = repr_

    return agent
