from math import log

try:
    from game import Game
except ImportError:
    from .game import Game


def compute_elo(agents, board_size, stones, komi, show=True,
                n_won_sente=0, n_lost_sente=0, n_won_gote=0, n_lost_gote=0):

    assert len(agents) == 2

    def elo(n_won, n_lost, c_elo=log(10) / 400):
        score = (n_won + 1) / (n_won + n_lost + 2)
        return -log(1 / score - 1) / c_elo

    def gen(w_0, l_0, w_1, l_1):
        while True:
            game = Game(board_size=board_size, stones=stones, komi=komi, show=show)
            game.play(agents)

            if game.outcome == 1:
                w_0 += 1
            elif game.outcome == 0:
                l_0 += 1

            yield w_0, l_0, w_1, l_1

            game = Game(board_size=board_size, stones=stones, komi=komi, show=show)
            game.play(list(reversed(agents)))

            if game.outcome == 1:
                l_1 += 1
            elif game.outcome == 0:
                w_1 += 1

            yield w_0, l_0, w_1, l_1

    for w0, l0, w1, l1 in gen(n_won_sente, n_lost_sente, n_won_gote, n_lost_gote):

        yield {'n_won_sente': w0,
               'n_lost_sente': l0,
               'elo_sente': elo(w0, l0),
               'n_won_gote': w1,
               'n_lost_gote': l1,
               'elo_gote': elo(w1, l1),
               'n_won': w0 + w1,
               'n_lost': l0 + l1,
               'elo': elo(w0 + w1, l0 + l1),
               }
