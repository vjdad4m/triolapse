import numpy as np

from collinear import find_collinear 

class NoThreeInLine:
    """
    A class representing a game with a grid where players take turns to place pieces.
    
    Args:
        grid_size (int): The size of the grid.

    Attributes:
        grid_size (int): The size of the grid.
        states (list): A list of numpy arrays representing the game state at each turn.
    """
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.states = [np.zeros((grid_size, grid_size), dtype=np.float32)]

    def get_legal_moves(self) -> np.ndarray:
        """
        Get the legal moves for the current game state.

        Returns:
            np.ndarray: An array of legal moves, where each element is a boolean indicating
            whether a move is legal at that position.
        """
        moves = self.states[-1] == 0
        return moves.reshape(-1)

    def make_move(self, index: int):
        """
        Make a move in the game.

        Args:
            index (int): The index of the move to be made.
        """
        c = index // self.grid_size
        r = index % self.grid_size
        last_state = np.copy(self.states[-1])
        last_state[c][r] = 1
        self.states.append(last_state)

    def is_terminal(self, idx: int = -1) -> bool:
        """
        Check if the game has reached a terminal state.

        Args:
            idx (int, optional): The index of the game state to check. Defaults to -1.

        Returns:
            bool: True if the game is in a terminal state, False otherwise.
        """
        n_collinear, _ = find_collinear(self.states[idx])
        return n_collinear > 0

    def calculate_reward(self) -> tuple:
        """
        Calculate the reward for the current game state.

        Returns:
            tuple: A tuple containing the number of pieces placed and the reward value.
        """
        correct_idx = -1
        while self.is_terminal(correct_idx):
            correct_idx -= 1
        n_placed = int(np.sum(self.states[correct_idx]))
        if n_placed == 2 * self.grid_size:
            return 2 * self.grid_size, 1
        return n_placed, ((n_placed - self.grid_size) / self.grid_size)

    def load_states(self, state_list: list):
        """
        Load a list of game states into the `states` attribute.

        This function replaces the current game states with the provided list of states.

        Args:
            state_list (list): A list of numpy arrays representing the game states.
        """
        self.states = state_list

def draw_grid(grid):
    for row in grid:
        row_str = "".join(map(str, list(int(r) for r in row)))
        row_str = row_str.replace("0", "⬜")
        row_str = row_str.replace("1", "⬛")
        print(row_str)
