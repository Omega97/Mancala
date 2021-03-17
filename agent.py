
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

    def get_move(self, state):
        """agent suggests move WITHOUT modifying it's internal state (i.e. that move has not been played yet)"""
        raise NotImplementedError

    def close(self, final_state, outcome):
        """shut down agent (save data, do training etc.)"""
        pass

    def __call__(self, state):
        return self.get_move(state)

    def __repr__(self):
        """build bot name"""
        return 'unknown agent'
