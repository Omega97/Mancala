from random import seed
import numpy as np
import matplotlib.pyplot as plt
from game import Game, State
from elo import compute_elo
from agents_collection import *
from utils import *


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


def test_kifu(board_size=5, stones=3, komi=.5, k_edit=3.):
    agents = []
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=200, k_focus_branch=1., heuristic_par=1.2)]
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=200, k_focus_branch=1., heuristic_par=1.5)]

    game = Game(board_size=board_size, stones=stones, komi=komi, do_record=True, show=True)
    game.play(agents)

    data00 = game.get_player_data(0)
    data01 = game.get_player_data(1)
    data10 = game.get_edited_player_data(0, k=k_edit)
    data11 = game.get_edited_player_data(1, k=k_edit)

    x0_ = [i.n_moves for i in data00['kifu']]
    x1_ = [i.n_moves for i in data01['kifu']]

    i_print(game.get_training_data(k=3))

    length = len(game.kifu)

    fig, ax = plt.subplots(2)
    ax[0].plot(x0_, data00['values'])
    ax[0].plot(x0_, [i[1] for i in data10])
    ax[0].scatter(length-1, data00['outcome'], c='orange')
    ax[0].set_xlim([0, length])
    ax[0].set_ylim([-.1, 1.1])

    ax[1].plot(x1_, data01['values'])
    ax[1].plot(x1_, [i[1] for i in data11])
    ax[1].set_xlim([0, length])
    ax[1].set_ylim([-.1, 1.1])
    ax[1].scatter(length-1, data01['outcome'], c='orange')

    plt.show()


def test_trees(board=None, board_size=6, stones=4, komi=.5, player=0, n_rollouts=100,
               k_focus=0, k_priority=1.5, show_tree=False, show=True):

    state0 = State(board_size=board_size, stones=stones, komi=komi, player=player)

    if board is not None:
        state0.board = board

    tree = Tree(state0, core_agent=RandomAgent(), fast_agent=SimpleAgent(),
                player_id=player, k_focus_decision=k_focus, k_focus_branch=k_priority)

    tree.search(n_rollouts)
    policy, value = tree.get_final_policy_and_value()

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

    tree = Tree(state0, core_agent=RandomAgent(), fast_agent=SimpleAgent(), player_id=player, k_focus_decision=0)

    tree.search(n_rollouts)
    policy, value = tree.get_final_policy_and_value()

    if show:
        if policy.argmax() not in sol:
            print('\n')
            print(tree.get_root().state)
            print()
            print('p =', policy)
            print('\n')

    return policy.argmax() in sol


def test_game(board_size=6, stones=4, komi=.5, n_rollouts=100, show=True):
    agents = []
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=n_rollouts)]
    agents += [SimpleAgent()]

    game = Game(state=None, board_size=board_size, stones=stones, komi=komi, show=show)
    game.play(agents)


def test_game_from_position(board, board_size, komi=.5, n_rollouts=100, player=0):
    agents = []
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=n_rollouts)]
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=n_rollouts)]

    state = State(board=board, board_size=board_size, stones=1, komi=komi, player=player)
    game = Game(state=state, board_size=board_size, stones=1, komi=komi, show=True)
    game.play(agents)
    print(game)


def test_elo(board_size=5, stones=3, komi=.5, t=1., show_game=True):
    agents = []
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=100, k_focus_branch=1.5)]
    agents += [TreeAgent(RandomAgent(), SimpleAgent(), n_rollouts=100, k_focus_branch=1.5)]

    print('Computing...\n')
    gen = compute_elo(agents, board_size=board_size, stones=stones, komi=komi, show=show_game)
    gen = yield_periodically(gen, t)
    for d in gen:
        print(f'{d["n_won_sente"]:6} | {d["n_lost_sente"]:<6} {d["elo_sente"]:<+12.0f}'
              f'{d["n_won_gote"]:6} | {d["n_lost_gote"]:<6}  {d["elo_gote"]:<+12.0f}'
              f'{d["n_won"]:6} | {d["n_lost"]:<6} {d["elo"]:<+12.0f}')


def exercises(n_rollouts=100):
    """shows the exercises that have not been solved correctly"""
    problems = [{'board': [3, 1, 1, 0, 0, 0, 1, 4], 'player': 0, 'board_size': 3, 'sol': (2,)},
                {'board': [3, 1, 0, 0, 1, 1, 1, 3], 'player': 0, 'board_size': 3, 'sol': (0,)},
                {'board': [0, 2, 1, 0, 0, 2, 1, 0], 'player': 0, 'board_size': 3, 'sol': (2,)},
                {'board': [1, 0, 1, 4, 3, 1, 1, 0], 'player': 1, 'board_size': 3, 'sol': (2,)},
                {'board': [4, 1, 1, 0, 0, 1, 1, 4], 'player': 1, 'board_size': 3, 'sol': (2,)},
                {'board': [2, 1, 1, 3, 3, 1, 0, 0], 'player': 1, 'board_size': 3, 'sol': (0,)},
                {'board': [4, 1, 0, 0, 3, 1, 2, 1], 'player': 1, 'board_size': 3, 'sol': (2,)},
                {'board': [0, 3, 1, 2, 0, 0, 3, 1, 0, 0], 'player': 0, 'board_size': 4, 'sol': (1, 3)},
                ]

    for kw in problems:
        test_trees_exp(n_rollouts=n_rollouts, **kw)


def test_time(n_rollouts=30):

    v = []

    while True:
        seed(0)
        t = time()
        test_game(show=False, n_rollouts=n_rollouts)
        v += [time()-t]
        print(f'{np.average(v):10.3f} +- {np.std(v):.3f}')


def test_heuristic():

    board = [0, 0, 0, 3, 1, 1, 0,  0, 0, 0, 0, 2, 1, 1]

    for n_ in range(5, 11):
        p = test_trees(board=board, board_size=6, player=0, show=False, n_rollouts=2 ** n_)
        plt.plot(list(range(1, len(board) // 2)), p, color='green', linewidth=(n_ - 4)/2)

    plt.show()


if __name__ == '__main__':
    # seed(0)
    # test_elo()
    test_kifu()
    # exercises()
