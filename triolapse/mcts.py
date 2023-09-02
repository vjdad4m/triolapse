from typing import List, Union

import numpy as np
import torch
import tqdm

from game import NoThreeInLine
from visuals import draw_grid


class TreeNode:
    """
    A class representing a node in the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes:
        game (NoThreeInLine): The game associated with this node.
        parent (TreeNode or None): The parent node of this node. None if it's the root.
        action (int or None): The action taken to reach this node. None if it's the root.
        prior (float): The prior probability of selecting this node.
        children (list of TreeNode): The child nodes of this node.
        visit_count (float): The number of times this node has been visited during MCTS.
        cumulative_value (float): The cumulative value associated with this node during MCTS.
    """
    def __init__(self, game: NoThreeInLine, parent: 'TreeNode' = None, action: int = None, prior: float = 1.0):
        """
        Initialize a TreeNode.

        Args:
            game (NoThreeInLine): The game state associated with this node.
            parent (TreeNode or None): The parent node of this node. None if it's the root.
            action (int or None): The action taken to reach this node. None if it's the root.
            prior (float): The prior probability of selecting this node.
        """
        self.game = game
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = []

        self.visit_count = 0.0
        self.cumulative_value = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_ucb(self) -> float:
        q = 0 if self.visit_count == 0 else self.cumulative_value / self.visit_count
        pv = 1 if self.parent is None else np.sqrt(self.parent.visit_count)
        
        return q + (pv / (self.visit_count + 1)) * self.prior

    def search(self) -> 'TreeNode':
        best_ucb = float('-inf')
        best_child = self
        for child in self.children:
            ucb = child.get_ucb()
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        return best_child    

    def expand(self, policy: np.ndarray):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state_list = self.game.states.copy()
                new_game = NoThreeInLine(self.game.grid_size)
                new_game.load_states(child_state_list)
                new_game.make_move(action)

                child = TreeNode(new_game, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value: float):
        self.cumulative_value += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)

    def __str__(self) -> str:
        """
        Generate a string representation of the TreeNode's attributes.

        Returns:
            str: A string containing information about the TreeNode's game state,
            action, visit count, cumulative value, prior, and whether it's a leaf node.
        """
        return (
            f"{draw_grid(self.game.states[-1])}\n"
            f"Action: {self.action}\n"
            f"Number of points placed: {int(np.sum(self.game.states[-1]))}\n"
            f"Visit Count: {self.visit_count}\n"
            f"Cumulative Value: {self.cumulative_value}\n"
            f"Prior: {self.prior}\n"
            f"Is Leaf Node: {self.is_leaf()}"
        )

class MonteCarloTreeSearch:
    """
    A class representing the Monte Carlo Tree Search (MCTS) algorithm for game AI.

    Attributes:
        game (NoThreeInLine): The game instance to which MCTS is applied.
        model (torch.nn.Module): The neural network model used for policy and value predictions.
    """
    def __init__(self, game: NoThreeInLine, model: torch.nn.Module):
        """
        Initialize the MonteCarloTreeSearch.

        Args:
            game (NoThreeInLine): The game instance to which MCTS is applied.
            model (torch.nn.Module): The neural network model used for policy and value predictions.
        """
        self.game = game
        self.model = model

    @torch.no_grad()
    def search(self, n_searches: int = 1600, state_list: List[np.ndarray] = [], root: TreeNode = None) -> (np.ndarray, TreeNode):
        """
        Perform Monte Carlo Tree Search.

        Args:
            n_searches (int): The number of MCTS iterations to perform.
            state_list (list[np.ndarray]): List of game states to initialize the game with.
            root (TreeNode): The root node of the search tree. If None, a new tree will be created.

        Returns:
            action_probs (numpy.ndarray): Array containing action probabilities after MCTS.
            root (TreeNode): The root node of the final search tree.
        """
        # Load state list if provided
        if state_list != []:
            self.game.load_states(state_list)

        # Create the root node if not provided
        if root is None:
            root = TreeNode(self.game)

        # Perform MCTS for a specified number of iterations
        for search in tqdm.trange(n_searches):
            node = root

            # Selection phase: Traverse the tree to select a leaf node using UCB (Upper Confidence Bound)
            while not node.is_leaf():
                node = node.search()
            
            current_game = node.game
            is_game_over = current_game.is_terminal()

            if not is_game_over:
                # Expansion phase: Expand the selected node by evaluating the policy and value predictions
                policy, value = self.model(current_game.get_state_tensor())
                policy = policy.cpu()[0]
                valid_moves = current_game.get_legal_moves()
                policy = torch.softmax(policy, 0).numpy()
                policy += 1e-8 # Numerical stability to ensure all probabilities are greater than 0.
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()

                # Create child nodes for valid moves and add them to the tree
                node.expand(policy)
            else:
                # Terminal state: Calculate the reward and backpropagate it
                value = current_game.calculate_reward()[1]
                if value == 1:
                    print('!! found solution !!')
                    draw_grid(current_game.states[-1])
        
            # Backpropagation phase: Update the cumulative value and visit count of nodes in the selected path
            node.backpropagate(value)
        
        # Calculate action probabilities based on visit counts of child nodes
        action_probs = np.zeros((self.game.grid_size ** 2))
        for child in root.children:
            action_probs[child.action] = child.visit_count
        action_probs /= np.sum(action_probs)

        return action_probs, root
