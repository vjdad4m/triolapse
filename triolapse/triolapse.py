from typing import List, Tuple

import numpy as np
import torch

from game import NoThreeInLine
from mcts import MonteCarloTreeSearch
from nn import ResNet
from visuals import draw_probabilities, draw_grid


def generate_self_play_rollout(grid_size: int, model: torch.nn.Module, n_searches: int = 1600, verbose: bool = True) -> Tuple[NoThreeInLine, List[np.ndarray]]:
    """
    Generate a self-play rollout of a game using Monte Carlo Tree Search (MCTS).

    Args:
        grid_size (int): The size of the game grid.
        model (torch.nn.Module): The neural network model for policy and value predictions.
        n_searches (int, optional): The number of MCTS iterations to perform. Default is 1600.
        verbose (bool, optional): Whether to display visualization (e.g., action probabilities). Default is True.

    Returns:
        Tuple[NoThreeInLine, List[np.ndarray]]: A tuple containing the final game state and a list of action probabilities
        at each step of the rollout.
    """
    game = NoThreeInLine(grid_size)
    tree = None
    action_probabilities = []

    while not game.is_terminal():
        mcts = MonteCarloTreeSearch(game, model)
        action_probs, tree = mcts.search(n_searches=n_searches, verbose=verbose)
        action_probabilities.append(action_probs)

        if verbose:
            draw_probabilities(action_probs)

        action = np.random.choice(grid_size ** 2, p=action_probs)
        for child in tree.children:
            if child.action == action:
                tree = child
                break

        game.make_move(action)
    
    return game, action_probabilities


def main():
    board_size = int(input("Grid size: "))
    model = ResNet(board_size)
    
    game, action_probabilities = generate_self_play_rollout(board_size, model, 3200)
    for idx, state in enumerate(game.states[1:]):
        print('Move:', idx + 1)
        draw_probabilities(action_probabilities[idx])
        draw_grid(state)
        print()
    
    n_placed, reward = game.calculate_reward()
    print("N Points Placed:", n_placed)
    print("Reward:", reward)

if __name__ == '__main__':
    main()