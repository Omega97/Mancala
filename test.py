from utils import *
from agents import *
from game import Game, State, compute_elo
from tree import Tree
from omar_utils.basic.iter_utils import yield_periodically


def test_mancala_gen():
    for i in mancala_gen(start=6, n_stones=10, board_size=3):
        print(i)


def test_action_distribution():
    a = ActionDistribution([0, 1, 2, 3, 4])
    print(a.choose_move())


def test_copy(board_size=3):
    s0 = State(board_size=board_size, stones=2, komi=.5)
    s1 = s0.make_move(ActionDistribution([1, 0, 0]))
    print(type(s0.board))
    print(s0)
    print(s1)


def test_adjust_values():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.linspace(0, 2 * np.pi, 50)
        x = np.cos(x / 2)
        # x = np.zeros(51)
        # x[-1] = 1

        y = adjust_values(x, k=1)
        plt.plot(x)
        plt.plot(y)
        plt.show()


def test_kifu(board_size=6, stones=4, komi=.5, k=1.):
    game = Game(board_size=board_size, stones=stones, komi=komi)
    game.play(RandomAgent(), RandomAgent())

    print(game)

    data = game.get_edited_player_data(0, k=k)
    i_print(data['kifu'])
    i_print(data['values'])

    data = game.get_edited_player_data(1, k=k)
    i_print(data['kifu'])
    i_print(data['values'])


def test_trees(player=0, n_rollouts=2000):
    fast_agent = RandomAgent()
    state0 = State(board_size=3, stones=1, komi=.5, player=player)
    state0.board = [3, 1, 1, 0,
                    0, 0, 1, 4]
    # state0.board = [3, 1, 0, 0, 3, 1, 0, 0]

    tree = Tree(state0, fast_agent, player_id=player)

    tree.search(n_rollouts=n_rollouts)
    print(tree)


def test_game(board_size=3, stones=1, komi=.5):
    # agent_1 = RandomAgent()
    agent_1 = TreeAgent(RandomAgent(), n_rollouts=50)
    # agent_1 = SimpleAgent()
    agent_2 = RandomAgent()

    game = Game(board_size=board_size, stones=stones, komi=komi)
    game.play(agent_1, agent_2)
    print(game)
    print(game.outcome)


def test_elo(board_size=3, stones=1, komi=.5, t=2.):
    agent_1 = TreeAgent(RandomAgent(), n_rollouts=100)
    # agent_1 = SimpleAgent()
    agent_2 = RandomAgent()

    gen = compute_elo(agent_1, agent_2,
                      board_size=board_size, stones=stones, komi=komi)
    gen = yield_periodically(gen, t)
    for i in gen:
        print(''.join([f'{j:+8.0f}' for j in i]))


if __name__ == '__main__':
    from random import seed
    seed(0)

    test_trees()
    # test_game(board_size=3, stones=1, komi=.5)
    # test_elo()
