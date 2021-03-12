from utils import *
from agents import *
from game import Game, State, compute_elo
from tree import Tree
import matplotlib.pyplot as plt


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


def test_kifu(board_size=6, stones=4, komi=.5, k_edit=3.):
    agents = []
    agents += [TreeAgent(SimpleAgent(), SimpleAgent(), n_rollouts=200)]
    agents += [TreeAgent(SimpleAgent(), SimpleAgent(), n_rollouts=800)]
    # agents += [SimpleAgent()]
    # agents += [RandomAgent()]

    game = Game(board_size=board_size, stones=stones, komi=komi, do_record=True, show=True)
    game.play(*agents)

    data00 = game.get_player_data(0)
    data01 = game.get_player_data(1)
    data10 = game.get_edited_player_data(0, k=k_edit)
    data11 = game.get_edited_player_data(1, k=k_edit)

    x0_ = [i.n_moves for i in data00['kifu']]
    x1_ = [i.n_moves for i in data01['kifu']]

    i_print(zip(data00['kifu'], data00['values']))
    i_print(zip(data01['kifu'], data01['values']))

    fig, ax = plt.subplots(2)
    ax[0].plot(x0_, data00['values'])
    ax[0].plot(x0_, data10['values'])
    ax[0].set_ylim([-.1, 1.1])
    ax[1].plot(x1_, data01['values'])
    ax[1].plot(x1_, data11['values'])
    ax[1].set_ylim([-.1, 1.1])
    plt.show()


def test_trees(board=None, board_size=6, stones=4, komi=.5, player=0, n_rollouts=300,
               k_focus=0, k_priority=1.5, show_tree=False, show=True):

    state0 = State(board_size=board_size, stones=stones, komi=komi, player=player)

    if board is not None:
        state0.board = board

    tree = Tree(state0, core_agent=RandomAgent(), fast_agent=SimpleAgent(),
                player_id=player, k_focus=k_focus, k_priority=k_priority)

    tree.search(n_rollouts)
    policy, value = tree.get_policy_and_value()

    if show:
        print('\n')
        print('n_visits =', tree.get_root().get_visits())
        print('p =', policy)
        print(f'v = {value:.3f}')

        if show_tree:
            print()
            print(tree)
            print()
            print('n_visits =', tree.get_root().get_visits())
            print('p =', policy)
            print(f'v = {value:.3f}')
            print('\n')

    return policy


def test_trees_exp(sol: tuple, board=None, board_size=3, stones=1, komi=.5, player=0, n_rollouts=100, show=True):

    state0 = State(board_size=board_size, stones=stones, komi=komi, player=player)

    if board is not None:
        state0.board = board

    tree = Tree(state0, core_agent=RandomAgent(), fast_agent=SimpleAgent(), player_id=player, k_focus=0)

    tree.search(n_rollouts)
    policy, value = tree.get_policy_and_value()

    if show:
        if policy.argmax() not in sol:
            print('\n')
            print(tree.get_root().state)
            print()
            print('p =', policy)
            print('\n')

    return policy.argmax() in sol


def test_game(board_size=6, stones=4, komi=.5, n_rollouts=50, show=True):
    agents = []
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=n_rollouts)]
    agents += [RandomAgent()]
    # agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=n_rollouts)]
    # agents += [HumanAgent()]

    game = Game(state=None, board_size=board_size, stones=stones, komi=komi, show=show)
    game.play(*agents)


def test_game_from_position(board, board_size, komi=.5, n_rollouts=100, player=0):
    agents = []
    # agents += [HumanAgent()]
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=n_rollouts)]
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=n_rollouts)]

    state = State(board=board, board_size=board_size, stones=1, komi=komi, player=player)
    game = Game(state=state, board_size=board_size, stones=1, komi=komi, show=True)
    game.play(*agents)
    print(game)


def test_elo(board_size=6, stones=4, komi=.5, t=1., show_game=True):
    agents = []
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=500, k_priority=1.5)]
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=100, k_priority=1.5)]
    # agents += [SimpleAgent()]

    print('Computing...\n')
    gen = compute_elo(agents, board_size=board_size, stones=stones, komi=komi, show=show_game)
    gen = yield_periodically(gen, t)
    for w0, w1, w, s, n in gen:
        print()
        print(f'{w0:+8.0f} {w1:+8.0f} {w:+8.0f} \t ({s} / {n})')
        print()


def exercises(n_rollouts=100):
    """shows the exercises that have not been solved correctly"""
    test_trees_exp(n_rollouts=n_rollouts, sol=(1, 3), board=[0, 3, 1, 2, 0, 0, 3, 1, 0, 0], player=0, board_size=4)
    test_trees_exp(n_rollouts=n_rollouts, sol=(2,), board=[3, 1, 1, 0, 0, 0, 1, 4], player=0, board_size=3)
    test_trees_exp(n_rollouts=n_rollouts, sol=(2,), board=[3, 1, 1, 0, 0, 1, 1, 4], player=0, board_size=3)
    test_trees_exp(n_rollouts=n_rollouts, sol=(0,), board=[3, 1, 0, 0, 1, 1, 1, 3], player=0, board_size=3)

    test_trees_exp(n_rollouts=n_rollouts, sol=(2,), board=[0, 2, 1, 0, 0, 2, 1, 0], player=0, board_size=3)

    test_trees_exp(n_rollouts=n_rollouts, sol=(2,), board=[0, 0, 1, 4, 3, 1, 1, 0], player=1, komi=-.5, board_size=3)
    test_trees_exp(n_rollouts=n_rollouts, sol=(2,), board=[3, 1, 1, 0, 0, 1, 1, 4], player=1, komi=-.5, board_size=3)
    test_trees_exp(n_rollouts=n_rollouts, sol=(0,), board=[1, 1, 1, 3, 3, 1, 0, 0], player=1, komi=-.5, board_size=3)
    test_trees_exp(n_rollouts=n_rollouts, sol=(2,), board=[3, 1, 0, 0, 3, 1, 2, 1], player=1, komi=-.5, board_size=3)


def test_time(n_rollouts=30):
    from time import time
    from random import seed
    import numpy as np

    v = []

    while True:
        seed(0)
        t = time()
        test_game(show=False, n_rollouts=n_rollouts)
        v += [time()-t]
        print(f'{np.average(v):10.3f}')
        print(f'{np.std(v):10.3f}')
        print()


def test_heuristic():

    # board = [1, 0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0]
    # board = [1, 0, 0, 0, 2, 1, 0,  0, 0, 0, 0, 0, 0, 2]
    # board = [1, 0, 0, 3, 1, 1, 0,  0, 0, 0, 0, 0, 0, 4]
    # board = [1, 0, 0, 0, 2, 1, 0,  0, 0, 0, 0, 2, 1, 0]
    # board = [1, 1, 1, 3, 1, 1, 0,  0, 0, 0, 0, 0, 0, 4]

    board = [0, 0, 0, 3, 1, 1, 0,  0, 0, 0, 0, 2, 1, 1]

    for n_ in range(5, 11):
        p = test_trees(board=board, board_size=6, player=0, show=False, n_rollouts=2 ** n_)
        plt.plot(list(range(1, len(board) // 2)), p, color='green', linewidth=(n_ - 4)/2)

    plt.show()


if __name__ == '__main__':
    # from random import seed
    # seed(0)

    # test_trees([0, 0, 0, 0, 3, 0, 0,
    #             0, 0, 0, 0, 2, 1, 0], player=0, n_rollouts=500)

    # board_ = [0, 0, 0, 3, 1, 1, 0,
    #           0, 0, 0, 0, 2, 1, 1]
    # test_trees(board=board_, board_size=6, player=0,
    #            show=True, show_tree=True, k_priority=1.5, n_rollouts=200)

    # test_heuristic()

    # exercises()

    # test_game()
    # test_time()
    # test_kifu()

    # test_game_from_position(board=[0, 0, 0, 0, 2, 1, 2,
    #                                0, 0, 0, 3, 1, 1, 0],
    #                         board_size=6, player=0,
    #                         n_rollouts=300)

    test_elo()
