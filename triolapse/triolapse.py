import numpy as np

from game import NoThreeInLine
from mcts import MonteCarloTreeSearch
from nn import ResNet
from visuals import draw_probabilities


def main():
    board_size = int(input("Grid size: "))
    n_searches = 1600
    game = NoThreeInLine(board_size)
    model = ResNet(board_size)
    tree = None

    while not game.is_terminal():
        mcts = MonteCarloTreeSearch(game, model)
        action_probs, tree = mcts.search(n_searches=n_searches)
        print('action probs: ', end="")
        draw_probabilities(action_probs)
        action = np.random.choice(board_size ** 2, p=action_probs)
        for child in tree.children:
            if child.action == action:
                tree = child
                break
        game.make_move(action)
        
        print(tree)


if __name__ == '__main__':
    main()