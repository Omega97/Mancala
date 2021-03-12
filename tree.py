from game import Game, State
from action_distribution import one_hot, ActionDistribution
from utils import *
from copy import copy


class Node:

    def __init__(self, state: State, agent, tree, depth=0, parent=None):
        self.state = state
        self.agent = agent
        self.tree = tree
        self.policy, self.value = agent(state)
        self.policy *= self.state.legal_moves()
        self.depth = depth
        self.parent = parent
        self.child = dict()
        self.wins = 0
        self.losses = 0
        self.n_root_visits = None
        self.base_depth = None

    def add_child(self, move_id):
        """check if state already exists"""
        if move_id in self.child:
            raise ValueError('Move id already used')

        move = one_hot(move_id, self.state.board_size)
        state = self.state.make_move(move)
        new_node = Node(state, agent=self.agent, tree=self.tree, depth=self.depth + 1, parent=self)
        self.child[move_id] = new_node
        return new_node

    def get_player(self):
        return self.state.player

    def get_root_player(self):
        return self.tree.get_root().state.player

    def get_parent(self):
        return self.parent

    def backtracking(self, outcome):
        """
        outcome is absolute
        each node contains:
        - number of wins of the player of the node
        - number of losses of the player of the node
        """
        # todo check

        x = 1 if self.get_player() != outcome else 0
        self.wins += x
        self.losses += 1-x

        if self.parent is not None:
            self.parent.backtracking(outcome)

    def get_visits(self):
        return self.wins + self.losses

    def get_scores_and_visits(self):
        return self.wins, self.losses, self.wins + self.losses

    def expectation(self):
        """subjective expectation value of outcome for the root player"""
        w, l, n = self.get_scores_and_visits()

        player = self.get_player()
        root_player = self.get_root_player()

        outcome = self.outcome()
        if outcome is None:
            if player == root_player:
                return (w + 1) / (n + 2)
            else:
                return (l + 1) / (n + 2)
        else:
            return outcome if root_player == 0 else 1 - outcome

    def priority(self):     # todo better heuristic
        """used to choose the sub-node to visit
        higher if parent node should lead to self
        """
        k_p = self.tree.k_priority

        def f(wins, losses):
            return (wins + 1) / (wins + losses + 2) ** k_p

        w, l, n = self.get_scores_and_visits()
        if self.get_parent() is None:
            return None
        else:
            if self.get_parent().get_player() == self.get_player():
                return f(w, l)
            else:
                return f(l, w)

    def is_terminal(self):
        return self.state.is_game_over()

    def choose_move_id(self):
        priorities = copy(self.policy.v)
        for i in self.child:
            priorities[i] = self.child[i].priority()
        return argmax(priorities)   # todo

    def outcome(self):
        return self.state.result()

    def __repr__(self, bar_length=40):
        w, l, n = self.get_scores_and_visits()
        if self.n_root_visits is None:
            self.n_root_visits = n
        for i in self.child:
            self.child[i].n_root_visits = self.n_root_visits

        if self.base_depth is None:
            self.base_depth = self.depth
        for i in self.child:
            self.child[i].base_depth = self.base_depth

        s = ' '
        s += str(self.state)
        s += ' ' * 4
        x = self.outcome()
        if x == 0:
            s += 'X'
        elif x == 1:
            s += 'O'
        else:
            s += ' '
        s += ' ' * 4

        n_bar = round(bar_length * n / self.n_root_visits)

        p_bar = round(self.expectation() * n_bar)

        c = '|' + '#' * p_bar + ' ' * (n_bar - p_bar) + '|'

        s += f'{c:{6 + bar_length}}'
        s += ' ' * 4
        s += '.  ' * (self.depth - self.base_depth)

        if self.get_player() == self.get_root_player():
            s += f'{w} | {l}'
        else:
            s += f'{l} | {w}'

        s += '   -   '

        s += f'{self.expectation():.3f}'
        if self.priority():
            s += f'  ({self.priority():.3f})'

        for i in self.child:
            s += '\n' + str(self.child[i])

        self.n_root_visits = None
        self.base_depth = None

        return s


class Tree:

    def __init__(self, state, core_agent, fast_agent, player_id, k_focus=1., k_priority=1.5):
        self.root = Node(state, core_agent, self)
        self.fast_agent = fast_agent
        self.active_nodes = [self.root]
        self.node_count = 1
        self.game_kwargs = {'board_size': state.board_size,
                            'stones': state.stones,
                            'komi': state.komi}
        self.player_id = player_id
        self.k_focus = k_focus
        self.k_priority = k_priority

    def __repr__(self):
        return str(self.get_root())

    def get_root(self) -> Node:
        return self.root

    def pick_node(self, root=None) -> Node:
        """

        :param root:
        :return:
        """
        if root is None:
            root = self.get_root()
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
            node = self.pick_node(self.get_root())

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

    def get_policy_and_value(self):
        """use the resulting tree of the search to determine a policy and a value"""
        if len(self.get_root().child) == 0:
            return self.get_root().policy, self.get_root().value

        v = [1 for _ in self.get_root().policy]

        for i in self.get_root().child:
            v[i] = self.get_root().child[i].get_visits()

        a = ActionDistribution(v)
        a *= self.get_root().state.legal_moves()
        a = a.focus(k=self.k_focus)

        return a, self.get_root().expectation()

    def re_plant(self, move_id):
        """select ew root for the tree"""
        if move_id not in self.get_root().child:
            self.get_root().add_child(move_id)
        self.root = self.get_root().child[move_id]
