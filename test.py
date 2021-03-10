from utils import *
from agents import *
from game import Game, State, compute_elo
from tree import Tree


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


def test_trees(board=None, board_size=3, stones=1, komi=.5, player=0, n_rollouts=200):

    state0 = State(board_size=board_size, stones=stones, komi=komi, player=player)

    if board is not None:
        state0.board = board

    tree = Tree(state0, core_agent=RandomAgent(), fast_agent=SimpleAgent(), player_id=player)

    tree.search(n_rollouts)
    policy, value = tree.get_policy_and_value()

    print('\n')
    print('n_visits =', tree.root.visits)
    print('p =', policy)
    print(f'v = {value:.3f}')
    print()
    print(tree)
    print()
    print('n_visits =', tree.root.visits)
    print('p =', policy)
    print(f'v = {value:.3f}')
    print('\n')


def test_trees_exp(expectation: int, board=None, board_size=3, stones=1, komi=.5, player=0, n_rollouts=30):

    state0 = State(board_size=board_size, stones=stones, komi=komi, player=player)

    if board is not None:
        state0.board = board

    tree = Tree(state0, core_agent=SimpleAgent(), fast_agent=SimpleAgent(), player_id=player)

    tree.search(n_rollouts)
    policy, value = tree.get_policy_and_value()

    if policy.quick_choose() != expectation:
        print('\n')
        print(tree.root.state)
        print()
        print('p =', policy)
        print('\n')


def test_game(board_size=5, stones=3, komi=.5, n_rollouts=30):
    agents = []
    agents += [TreeAgent(SimpleAgent(), SimpleAgent(), n_rollouts=n_rollouts)]
    agents += [SimpleAgent()]

    game = Game(board_size=board_size, stones=stones, komi=komi, show=True)
    game.play(*agents)


def test_elo(board_size=6, stones=4, komi=.5, t=1.):
    agents = []
    agents += [TreeAgent(SimpleAgent(), SimpleAgent(), n_rollouts=50)]
    # agents += [TreeAgent(SimpleAgent(), SimpleAgent(), n_rollouts=50)]
    agents += [RandomAgent()]

    gen = compute_elo(*agents, board_size=board_size, stones=stones, komi=komi, show=True)
    gen = yield_periodically(gen, t)
    for w0, w1, w, s, n in gen:
        print(f'{w0:+8.0f} {w1:+8.0f} {w:+8.0f} \t ({s} / {n})')
        print('\n')


def exercises():
    """shows the exercises that have not been solved correctly"""
    test_trees_exp(expectation=2, board=[3, 1, 1, 0, 0, 0, 1, 4], player=0)
    test_trees_exp(expectation=2, board=[3, 1, 1, 0, 0, 1, 1, 4], player=0)
    test_trees_exp(expectation=0, board=[3, 1, 0, 0, 1, 1, 1, 3], player=0)
    test_trees_exp(expectation=2, board=[3, 1, 2, 1, 3, 1, 0, 0], player=0)

    test_trees_exp(expectation=2, board=[0, 0, 1, 4, 3, 1, 1, 0], player=1, komi=-.5)
    test_trees_exp(expectation=2, board=[3, 1, 1, 0, 0, 1, 1, 4], player=1, komi=-.5)
    test_trees_exp(expectation=0, board=[1, 1, 1, 3, 3, 1, 0, 0], player=1, komi=-.5)
    test_trees_exp(expectation=2, board=[3, 1, 0, 0, 3, 1, 2, 1], player=1, komi=-.5)


if __name__ == '__main__':

    # test_trees(board_size=4, stones=3, n_rollouts=2000)

    # test_game()

    test_elo()
