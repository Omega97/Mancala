
try:
    from action_distribution import ActionDistribution
except ImportError:
    from .action_distribution import ActionDistribution


class Agent:
    """
    Agent(state: State) -> (policy, value)
    illegal moves are automatically filtered
    """
    def open(self, initial_state, player_id):
        """initialize agent (like tree-search, load data etc.)"""
        pass

    def set_move(self, move: ActionDistribution):
        """modify agent's internal state given that move has been played"""
        pass

    def get_move(self, state) -> ActionDistribution:
        """
        agent suggests move WITHOUT modifying it's internal state (i.e. that move has not been played yet)
        :param state: current board state (State)
        :return: return move distribution from which a move will be chosen
        """
        raise NotImplementedError

    def get_raw_policy(self) -> ActionDistribution:
        """in case you are using a tree-search and you want to save the policy of the search,  """
        pass

    def close(self, final_state, outcome):
        """shut down agent (save data, do training etc.)"""
        pass

    def __call__(self, state):
        return self.get_move(state)

    def __repr__(self):
        """build bot name"""
        return 'unnamed agent'
