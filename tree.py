from game import Game, State
from action_distribution import one_hot, ActionDistribution
from utils import choose
from copy import copy


class Node:

    def __init__(self, state: State, agent, depth=0, parent=None):
        self.state = state
        self.agent = agent
        self.policy, self.value = agent(state)
        self.policy *= self.state.legal_moves()
        self.depth = depth
        self.parent = parent
        self.child = dict()
        self.score = 0
        self.visits = 0

    def add_child(self, move_id):
        """check if state already exists"""
        if move_id in self.child:
            raise ValueError('Move id already used')

        move = one_hot(move_id, self.state.board_size)
        state = self.state.make_move(move)
        new_node = Node(state, agent=self.agent, depth=self.depth + 1, parent=self)
        self.child[move_id] = new_node
        return new_node

    def backtracking(self, score):
        """"""
        self.visits += 1
        self.score += score if self.state.player == 0 else 1-score
        if self.parent is not None:
            self.parent.backtracking(score)

    def expectation(self):
        return (self.score + 1) / (self.visits + 2)

    def is_terminal(self):
        return self.state.is_game_over()

    def choose_move_id(self):
        priorities = copy(self.policy.v)
        for i in self.child:
            x = self.child[i].score
            n = self.child[i].visits
            priorities[i] *= (x+1) / (n+2)**2

        return choose(priorities)

    def outcome(self):
        return self.state.result()

    def __repr__(self, state_space=40, n_bar=10):
        s = ' '
        s += str(self.state)
        s += ' ' * 5
        x = self.outcome()
        if x == 0:
            s += 'X' + ' ' * (n_bar+4) + ' '
        elif x == 1:
            s += 'O' + ' ' * (n_bar+4) + ' '
        else:
            n = int(self.expectation() * n_bar)
            s += ' ' * 4 + '|' + '#' * n + ' ' * (n_bar - n) + '|'
        s += ' ' * 5
        s += '.  ' * self.depth
        s += f'{self.score} / {self.visits}'
        s += '   -   '
        s += f'{self.expectation():.3f}'
        for i in self.child:
            s += '\n' + str(self.child[i])
        return s


class Tree:

    def __init__(self, state, fast_agent, player_id):
        self.root = Node(state, fast_agent)
        self.fast_agent = fast_agent
        self.active_nodes = [self.root]
        self.node_count = 1
        self.game_kwargs = {'board_size': state.board_size,
                            'stones': state.stones,
                            'komi': state.komi}
        self.player_id = player_id

    def __repr__(self):
        return str(self.root)

    def pick_node(self, root=None) -> Node:
        """

        :param root:
        :return:
        """
        if root is None:
            root = self.root
        if root.is_terminal():
            return root
        else:
            move_id = root.choose_move_id()
            if move_id in root.child:           # move already linked to sub-node
                node = root.child[move_id]
                return self.pick_node(root=node)
            else:                               # create sub-node
                return root.add_child(move_id)

    def search(self, n_rollouts):

        for _ in range(n_rollouts):
            # find node
            node = self.pick_node(self.root)

            if node.is_terminal():

                # get game outcome
                outcome = node.outcome()

                # update tree
                node.backtracking(outcome)

            else:
                # do rollout
                rollout = Game(node.state, **self.game_kwargs)
                rollout.play(self.fast_agent, self.fast_agent)  # init

                outcome = rollout.outcome

                # update tree
                node.backtracking(outcome)

    def get_policy_and_value(self, n_rollouts):
        self.search(n_rollouts=n_rollouts)

        if len(self.root.child) == 0:
            raise ValueError('Tree has no nodes!')

        v = [1 for _ in self.root.policy]

        for i in self.root.child:
            v[i] = self.root.child[i].visits

        return ActionDistribution(v), self.root.score

    def re_plant(self, move_id):
        """"""
        if move_id not in self.root.child:
            self.root.add_child(move_id)
        self.root = self.root.child[move_id]
