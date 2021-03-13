from game import Game, State
from action_distribution import one_hot, ActionDistribution
from copy import copy


class Node:

    def __init__(self, state: State, agent, depth=0, parent=None, token=False):
        self.state = state
        self.agent = agent
        self.depth = depth
        self.parent = parent
        self.token = token

        self.policy = None
        self.value = None
        self.child = dict()
        self.wins = 0
        self.losses = 0
        self.n_root_visits = None
        self.base_depth = None

    def __iter__(self):
        return iter(self.child)

    def get_child(self):
        return self.child

    def compute_policy_and_value(self):
        self.policy, self.value = self.agent(self.state)
        self.policy *= self.state.legal_moves()

    def get_policy(self):
        if self.policy is None:
            self.compute_policy_and_value()
        return self.policy

    def get_value(self):
        if self.value is None:
            self.compute_policy_and_value()
        return self.value

    def add_child(self, move_id):
        """check if state already exists"""
        if move_id in self.get_child():
            raise ValueError('Move id already used')

        move = one_hot(move_id, self.state.board_size)
        state = self.state.make_move(move)
        new_node = Node(state, agent=self.agent, depth=self.depth + 1, parent=self)
        self.get_child()[move_id] = new_node
        return new_node

    def get_player(self):
        return self.state.player

    def get_parent(self):
        return self.parent

    def get_token_node(self):
        if self.token:
            return self
        elif self.get_parent() is not None:
            return self.get_parent().get_root_player()
        else:
            raise ValueError('No token found (among parents)')

    def get_root_player(self):
        return self.get_token_node().get_player()

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

    def priority(self, heuristic_par):     # todo better heuristic
        """used to choose the sub-node to visit
        higher if parent node should lead to self
        """
        def f(wins, losses):
            return (wins + 1) / (wins + losses + 2) ** heuristic_par

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

    def choose_move_id(self, k_focus_branch, heuristic_par):
        priorities = copy(self.get_policy())
        for i in self:
            priorities[i] = self.get_child()[i].priority(heuristic_par)
        return priorities.focus(k_focus_branch).quick_choose()

    def outcome(self):
        return self.state.get_result()

    def __repr__(self, bar_length=40):
        w, l, n = self.get_scores_and_visits()
        if self.n_root_visits is None:
            self.n_root_visits = n
        for i in self:
            self.get_child()[i].n_root_visits = self.n_root_visits

        if self.base_depth is None:
            self.base_depth = self.depth
        for i in self:
            self.get_child()[i].base_depth = self.base_depth

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

        for i in self:
            s += '\n' + str(self.get_child()[i])

        self.n_root_visits = None
        self.base_depth = None

        return s


class Tree:

    def __init__(self, state, core_agent, fast_agent, player_id,
                 k_focus_branch=1.5, k_focus_decision=1., heuristic_par=1.5):
        self.root = Node(state, agent=core_agent, token=True)
        self.fast_agent = fast_agent
        self.active_nodes = [self.root]
        self.node_count = 1
        self.game_kwargs = {'board_size': state.board_size,
                            'stones': state.stones,
                            'komi': state.komi}
        self.player_id = player_id
        self.k_focus_branch = k_focus_branch
        self.k_focus_decision = k_focus_decision
        self.heuristic_par = heuristic_par

    def __repr__(self):
        return str(self.get_root())

    def get_root(self) -> Node:
        return self.root

    def set_root(self, node):
        self.root.token = False
        self.root = node
        self.root.token = True

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
            move_id = root.choose_move_id(self.k_focus_branch, self.heuristic_par)
            if move_id in root:           # move already linked to sub-node
                node = root.get_child()[move_id]
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
                rollout.play([self.fast_agent, self.fast_agent])  # init

                outcome = rollout.outcome

                # update tree
                node.backtracking(outcome)

    def get_final_policy_and_value(self):
        """use the resulting tree of the search to determine a policy and a value"""
        if len(self.get_root().get_child()) == 0:
            return self.get_root().get_policy(), self.get_root().get_value()

        p = [1 for _ in self.get_root().get_policy()]

        for i in self.get_root():
            p[i] = self.get_root().get_child()[i].get_visits()

        policy = ActionDistribution(p)
        policy *= self.get_root().state.legal_moves()
        policy = policy.focus(k=self.k_focus_decision)

        value = self.get_root().expectation()

        return policy, value

    def re_plant(self, move_id):
        """select ew root for the tree"""
        if move_id not in self.get_root().get_child():
            self.get_root().add_child(move_id)
        new_root = self.get_root().get_child()[move_id]
        self.set_root(new_root)
