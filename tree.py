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
        self._n_root_visits = None
        self._base_depth = None

    def add_child(self, move_id):
        """check if state already exists"""
        if move_id in self.child:
            raise ValueError('Move id already used')

        move = one_hot(move_id, self.state.board_size)
        state = self.state.make_move(move)
        new_node = Node(state, agent=self.agent, depth=self.depth + 1, parent=self)
        self.child[move_id] = new_node
        return new_node

    def backtracking(self, outcome, player_id):
        """"""
        self.visits += 1



        # self.score += outcome if self.state.player == 0 else 1 - outcome
        self.score += outcome if player_id == 0 else 1 - outcome



        if self.parent is not None:
            self.parent.backtracking(outcome, player_id)

    def expectation(self, player):
        outcome = self.outcome()
        if outcome is None:
            return (self.score + 1) / (self.visits + 2)
        else:
            return outcome if player == 0 else 1-outcome

    def priority(self):
        x = self.score
        n = self.visits


        return (x+1) / (n+2)**2


    def is_terminal(self):
        return self.state.is_game_over()

    def choose_move_id(self):
        priorities = copy(self.policy.v)
        for i in self.child:
            priorities[i] = self.child[i].priority()    # todo ??
        return choose(priorities)

    def outcome(self):
        return self.state.result()

    def __repr__(self, bar_length=40):
        if self._n_root_visits is None:
            self._n_root_visits = self.visits
        for i in self.child:
            self.child[i]._n_root_visits = self._n_root_visits

        if self._base_depth is None:
            self._base_depth = self.depth
        for i in self.child:
            self.child[i]._base_depth = self._base_depth

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

        n_bar = round(bar_length * self.visits / self._n_root_visits)

        n = round(self.expectation(self.state.player) * n_bar)
        c = '|' + '#' * n + ' ' * (n_bar - n) + '|'

        s += f'{c:{6 + bar_length}}'
        s += ' ' * 4
        s += '.  ' * (self.depth - self._base_depth)
        s += f'{self.score} / {self.visits}'
        s += '   -   '
        s += f'{self.expectation(self.state.player):.3f}'
        for i in self.child:
            s += '\n' + str(self.child[i])

        self._n_root_visits = None
        self._base_depth = None

        return s


class Tree:

    def __init__(self, state, core_agent, fast_agent, player_id):
        self.root = Node(state, core_agent)
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
                node.backtracking(outcome, self.player_id)

            else:
                # do rollout
                rollout = Game(node.state, **self.game_kwargs)
                rollout.play(self.fast_agent, self.fast_agent)  # init

                outcome = rollout.outcome

                # update tree
                node.backtracking(outcome, self.player_id)

    def get_policy_and_value(self):

        if len(self.root.child) == 0:
            raise ValueError('Tree has no nodes, perform search first!')

        v = [1 for _ in self.root.policy]

        for i in self.root.child:
            v[i] = self.root.child[i].visits

        a = ActionDistribution(v)
        a *= self.root.state.legal_moves()
        a = a.focus(k=.95)

        return a, self.root.score / self.root.visits

    def re_plant(self, move_id):
        """"""
        if move_id not in self.root.child:
            self.root.add_child(move_id)
        self.root = self.root.child[move_id]
