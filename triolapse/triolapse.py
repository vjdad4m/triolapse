import time
import multiprocessing
from typing import List, Tuple

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from game import NoThreeInLine
from mcts import MonteCarloTreeSearch
from nn import ResNet
from visuals import draw_probabilities, draw_grid


def generate_self_play_rollout(grid_size: int, model: torch.nn.Module, n_searches: int = 1600, verbose: bool = True, tqdm_level: int = 0) -> Tuple[NoThreeInLine, List[np.ndarray]]:
    """
    Generate a self-play rollout of a game using Monte Carlo Tree Search (MCTS).

    Args:
        grid_size (int): The size of the game grid.
        model (torch.nn.Module): The neural network model for policy and value predictions.
        n_searches (int, optional): The number of MCTS iterations to perform. Default is 1600.
        verbose (bool, optional): Whether to display visualization (e.g., action probabilities). Default is True.
        tqdm_level (int, optional): Which level should the tqdm progressbar use (used for multiprocessing). Default is 0.

    Returns:
        Tuple[NoThreeInLine, List[np.ndarray]]: A tuple containing the final game state and a list of action probabilities
        at each step of the rollout.
    """
    game = NoThreeInLine(grid_size)
    tree = None
    action_probabilities = []

    while not game.is_terminal():
        mcts = MonteCarloTreeSearch(game, model)
        action_probs, tree = mcts.search(n_searches=n_searches, verbose=verbose, tqdm_level=tqdm_level)
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

def get_training_data(game: NoThreeInLine, action_probabilities: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get training data for the neural network model from a game and corresponding action probabilities.

    Args:
        game (NoThreeInLine): The game instance containing states.
        action_probabilities (List[np.ndarray]): A list of action probabilities at each step of the game.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing tensors of states, action probabilities and the final reward.
    """
    states, probabilities = [], []
    for idx, state in enumerate(game.states[:-1]):
        states.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        probabilities.append(torch.tensor(action_probabilities[idx], dtype=torch.float32))

    states = torch.stack(states)
    probabilities = torch.stack(probabilities)

    reward_value = game.calculate_reward()[1]
    reward = torch.ones((states.shape[0], 1))
    reward *= reward_value

    return states, probabilities, reward

def train_model(model: nn.Module, states: torch.Tensor, action_probabilities: torch.Tensor, rewards: torch.Tensor, batch_size: int = 64, learning_rate: float = 0.001, num_epochs: int = 1) -> nn.Module:
    """
    Train a neural network model on the given training data for a specified number of epochs.

    Args:
        model (nn.Module): The neural network model to be trained.
        states (torch.Tensor): Tensor containing game states.
        action_probabilities (torch.Tensor): Tensor containing action probabilities.
        rewards (torch.Tensor): Tensor containing rewards.
        batch_size (int, optional): Batch size for training. Default is 64.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
        num_epochs (int, optional): Number of training epochs. Default is 1.

    Returns:
        nn.Module: The updated neural network model.
    """
    # Define loss function and optimizer
    criterion_value = nn.MSELoss()
    criterion_policy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for batching
    dataset = TensorDataset(states, action_probabilities, rewards)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in (tq := tqdm.trange(num_epochs)):
        total_loss = 0.0
        for batch_states, batch_probabilities, batch_rewards in dataloader:
            # Forward pass
            output_policy, output_value = model(batch_states)
            
            # Compute the loss
            value_loss = criterion_value(output_value, batch_rewards)
            policy_loss = criterion_policy(output_policy, batch_probabilities)
            loss = value_loss + policy_loss
            total_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Change tqdm description to loss for this epoch
        avg_loss = total_loss / len(dataloader)
        tq.set_description(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    return model

def self_play_worker(iter_idx: int, board_size: int, model: torch.nn.Module, n_searches: int):
    print("Generating self-play game", iter_idx + 1)
    time.sleep(2)
    game, action_probabilities = generate_self_play_rollout(board_size, model, n_searches=n_searches, tqdm_level=iter_idx)
    states, probabilities, reward = get_training_data(game, action_probabilities)
    return states, probabilities, reward

def main():
    board_size = int(input("Grid size: "))
    model = ResNet(board_size)

    N_WORKERS = 5

    N_SEARCHES = 200 # 1600
    N_SELF_PLAY = 10
    N_EPOCHS = 50

    iteration = 1

    if N_WORKERS == -1:
        N_WORKERS = multiprocessing.cpu_count()
    multiprocessing.set_start_method('spawn')

    while True:
        states_list, probabilities_list, reward_list = [], [], []

        with multiprocessing.Pool(processes=N_WORKERS) as pool:
            results = pool.starmap(self_play_worker, [(iter_idx, board_size, model, N_SEARCHES) for iter_idx in range(N_SELF_PLAY)])

        for result in results:
            states, probabilities, reward = result
            states_list.append(states)
            probabilities_list.append(probabilities)
            reward_list.append(reward)

        states = torch.cat(states_list)
        probabilities = torch.cat(probabilities_list)
        reward = torch.cat(reward_list)

        torch.save(states, f"self-play/states_{iteration}.pt")
        torch.save(probabilities, f"self-play/probabilities_{iteration}.pt")
        torch.save(reward, f"self-play/reward_{iteration}.pt")

        print("Saved iteration", iteration)

        train_model(model, states, probabilities, reward, num_epochs=N_EPOCHS)

        game, _ = generate_self_play_rollout(board_size, model)
        print('N placed and reward: ' + ', '.join(game.calculate_reward()))

        iteration += 1


if __name__ == '__main__':
    main()
