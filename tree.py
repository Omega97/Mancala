
try:
    from game import Game
    from action_distribution import ActionDistribution
    from node import Node
except ImportError:
    from .game import Game
    from .action_distribution import ActionDistribution
    from .node import Node


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
        """set the new root to node"""
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

    def compute_tree_policy(self):
        """ return the visit distribution of the possible moves in the root state """
        p = [1 for _ in self.get_root().get_policy()]

        for i in self.get_root():
            p[i] = self.get_root().get_child()[i].get_visits()

        return ActionDistribution(p)

    def get_final_policy_and_value(self):
        """use the resulting tree of the search to determine a policy and a value"""
        if len(self.get_root().get_child()) == 0:
            return self.get_root().get_policy(), self.get_root().get_value()

        policy = self.compute_tree_policy()
        policy = policy.focus(k=self.k_focus_decision)
        value = self.get_root().expectation()

        return policy, value

    def re_plant(self, move_id):
        """select ew root for the tree"""
        if move_id not in self.get_root().get_child():
            self.get_root().add_child(move_id)
        new_root = self.get_root().get_child()[move_id]
        self.set_root(new_root)
