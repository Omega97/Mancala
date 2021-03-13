from agents import neural_net_agent
from game import Game


def training_loop(neural_net, game: Game, n_rollouts):
    """

    neural_net:
    - callable: neural_net(subjective_state_representation) -> (policy: list, value: float)
    - .training(data, **kw): training method
    """

    assert hasattr(neural_net, '__call__')
    assert hasattr(neural_net, 'training')

    while True:

        agent = neural_net_agent(neural_net, n_rollouts=n_rollouts)
        game.play([agent, agent])
        data = ...
        neural_net.training(data)


if __name__ == '__main__':

    class NN:
        def __init__(self, board_size=6):
            self.board_size = board_size

        def __call__(self, *args, **kwargs):
            return [1 for _ in range(self.board_size)], .5

        def training(self, data):
            pass


    training_loop(NN(), Game(), n_rollouts=200)
