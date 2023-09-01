import numpy as np
import torch
import tqdm

from game import NoThreeInLine, draw_grid
from nn import ResNet


class TreeNode:
    def __init__(self, game, parent=None, action=None, prior=1.0):
        self.game = game
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = []

        self.visit_count = 0.0
        self.cumulative_value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def get_ucb(self):
        if self.visit_count == 0:
            q = 0
        else:
            q = 1 - ((self.cumulative_value / self.visit_count) + 1) / 2
        if self.parent is None:
            pv = 1
        else:
            pv = np.sqrt(self.parent.visit_count)
        return q +  (pv / (self.visit_count + 1)) * self.prior

    def search(self):
        best_ucb = float('-inf')
        best_child = self
        for child in self.children:
            ucb = child.get_ucb()
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        return best_child    

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state_list = self.game.states.copy()
                new_game = NoThreeInLine(self.game.grid_size)
                new_game.load_states(child_state_list)
                new_game.make_move(action)

                child = TreeNode(new_game, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.cumulative_value += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)

    def __str__(self):
        """
        Generate a string representation of the TreeNode's attributes.

        Returns:
            str: A string containing information about the TreeNode's game state,
            action, visit count, cumulative value, prior, and whether it's a leaf node.
        """
        return (
            f"{draw_grid(self.game.states[-1])}\n"
            f"Action: {self.action}\n"
            f"Visit Count: {self.visit_count}\n"
            f"Cumulative Value: {self.cumulative_value}\n"
            f"Prior: {self.prior}\n"
            f"Is Leaf Node: {self.is_leaf()}"
        )

class MonteCarloTreeSearch:
    def __init__(self, game, model):
        self.game = game
        self.model = model

    @torch.no_grad()
    def search(self, n_searches: int = 1600, state_list: list = [], root = None):
        # Load state list
        if state_list != []:
            self.game.load_states(state_list)

        if root is None:
            root = TreeNode(self.game)

        for search in tqdm.trange(n_searches):
            node = root
            while not node.is_leaf():
                node = node.search()
            current_game = node.game
            is_game_over = current_game.is_terminal()
            if not is_game_over:
                policy, value = self.model(current_game.get_state_tensor())
                policy = policy.cpu()[0]
                valid_moves = current_game.get_legal_moves()
                policy = torch.softmax(policy, 0).numpy()
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()

                node.expand(policy)
            else:
                value = current_game.calculate_reward()[1]
                if value == 1:
                    print('!! found solution !!')
                    draw_grid(current_game.states[-1])
        
            node.backpropagate(value)
        
        action_probs = np.zeros((self.game.grid_size ** 2))
        for child in root.children:
            action_probs[child.action] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs, root


def main():
    board_size = 7
    n_searches = 1600
    game = NoThreeInLine(board_size)
    model = ResNet(board_size)
    tree = None

    while not game.is_terminal():
        mcts = MonteCarloTreeSearch(game, model)
        action_probs, tree = mcts.search(n_searches=n_searches)
        print('action probs:', action_probs)
        
        action = np.random.choice(board_size ** 2, p=action_probs)
        for child in tree.children:
            if child.action == action:
                tree = child
                break
        
        game.make_move(action)
        
        print(tree)


if __name__ == '__main__':
    main()