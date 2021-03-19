"""
Here is where training actually happens
"""

try:
    from game import Game
    from agents_collection import neural_net_agent
    from utils import i_range
except ImportError:
    from .game import Game
    from .agents_collection import neural_net_agent
    from .utils import i_range


class Training:

    def __init__(self, game: Game, n_rollouts=100, k_edit=10.):
        self.game = game
        self.n_rollouts = n_rollouts
        self.k_edit = k_edit
        self.model = None
        self.agent1 = None
        self.agent2 = None
        self.raw_data = None
        self.data_set = None

    def load_neural_net(self, **__):
        """ try to load neural network from file, otw create it from scratch """
        ...

        def f(v):
            n = (len(v)-1) // 2
            return [1] * n

        self.model = f
        self.agent1 = neural_net_agent(self.model, n_rollouts=self.n_rollouts)
        self.agent2 = neural_net_agent(self.model, n_rollouts=self.n_rollouts)

    def prepare_dataset(self):
        """ use raw_data data to create data_set """
        ...

    def training(self):
        """ update model using data_set """
        ...

    def save(self):
        """save model on the device"""
        ...
    print('save')

    def training_loop(self, epochs=-1):
        """ here is where training actually happens"""
        assert self.agent1 is not None
        assert self.agent2 is not None

        for _ in i_range(epochs):
            self.game.play([self.agent1, self.agent2])
            self.raw_data = self.game.get_training_data(k=self.k_edit)
            self.prepare_dataset()
            self.training()
            self.save()


if __name__ == '__main__':

    training = Training(Game(board_size=3, stones=3, do_record=True, show=True),
                        n_rollouts=30, k_edit=5)
    training.load_neural_net()
    training.training_loop(epochs=2)
