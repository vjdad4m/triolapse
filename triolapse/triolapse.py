import numpy as np
import torch

from game import NoThreeInLine
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
        return max(self.children, lambda child: child.get_ucb())    

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


class MonteCarloTreeSearch:
    def __init__(self, game, model):
        self.game = game
        self.model = model

    @torch.no_grad()
    def search(self, state_list: list = []):
        # Load state list
        if state_list != []:
            self.game.states = state_list

        # TODO: Implement following pseudocode

        # for search in n_searches
        #   node <- root
        #   while not node.is_leaf()
        #       node <- node.search()
        #   game <- node.game
        #   is_game_over <- game.is_terminal()
        #   if not is_game_over
        #       policy, value <- model(game.states[-1])
        #       valid_moves <- game.get_legal_moves()
        #       policy = softmax(policy)  
        #       policy *= valid_moves
        #       policy /= sum(policy)
        #       node.expand(policy)
        #   else
        #       value <- game.calculate_reward()[1]
        #   node.backprop(value)
        # action_probs <- [0] * (N * N)
        # for child in root.children
        #   action_probs[child.action] = child.visit_count
        # action_probs /= sum(action_probs)
        # -> action_probs


def main():
    pass

if __name__ == '__main__':
    main()