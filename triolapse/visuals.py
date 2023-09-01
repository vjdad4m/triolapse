from typing import List, Union
import numpy as np

def draw_grid(grid: np.ndarray):
    """
    Draw a grid using ASCII characters to represent values in the grid.

    Args:
        grid (np.ndarray): A 2D NumPy array representing the grid.

    Returns:
        None: This function doesn't return a value. It prints the grid.
    """
    for row in grid:
        row_str = "".join(map(str, list(int(r) for r in row)))
        row_str = row_str.replace("0", "⬜")
        row_str = row_str.replace("1", "⬛")
        print(row_str)

def draw_probabilities(probabilities: Union[List[float], np.ndarray]):
    """
    Draw a visual representation of probabilities using ASCII characters.

    Args:
        probabilities (Union[List[float], np.ndarray]): A list or NumPy array of probabilities.

    Returns:
        None: This function doesn't return a value. It prints the representation.
    """
    values = "▁▂▃▃▄▅▆▆▇█"
    for p in probabilities:
        idx = max(min(int(p / 0.1), len(values) - 1), 0) 
        print(values[idx], end="")
    print()
