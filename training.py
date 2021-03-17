
try:
    from game import Game
    from agents_collection import neural_net_agent
    from utils import i_range
except ImportError:
    from .game import Game
    from .agents_collection import neural_net_agent
    from .utils import i_range


class Training:

    def __init__(self, game: Game, n_rollouts=200):
        self.game = game
        self.n_rollouts = n_rollouts
        self.agent = None
        self.data = None

    def load_neural_net(self):
        """  """
        neural_net = ...
        self.agent = neural_net_agent(neural_net, n_rollouts=self.n_rollouts)

    def training(self):
        """  """
        ...

    def save(self):
        """save"""
        ...

    def training_loop(self, k_game_edit=3., epochs=-1):
        """
        neural_net:
        - callable: neural_net(subjective_state_representation) -> (policy: list, value: float)
        - .training(data, **kw): training method
        """

        for _ in i_range(epochs):
            self.game.play([self.agent, self.agent])     # <<<
            self.data = self.game.get_training_data(k=k_game_edit)
            self.training()
            self.save()


if __name__ == '__main__':

    training = Training(Game())
    training.load_neural_net()
    training.training_loop(epochs=10)
