from collections import namedtuple
import random

import numpy as np

Point = namedtuple('Point', ['x', 'y'])

def turn(a: Point, b: Point, c: Point) -> int:
    value = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
    if value > 0:
        return 1
    elif value < 0:
        return -1
    return 0

def find_collinear(m: np.ndarray) -> (int, np.ndarray):
    r = 0
    p = []
    points = []

    m_size = m.shape[0]

    for i in range(m_size):
        for j in range(m_size):
            if m[i][j]:
                p.append(Point(i, j))

    p_size = len(p)

    for i in range(p_size):
        for j in range(i + 1, p_size):
            for k in range(j + 1, p_size):
                if turn(p[i], p[j], p[k]) == 0:
                    r += 1
                    points.append([p[i], p[j], p[k]])

    return r, points

class NoThreeInLine:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.states = [np.zeros((grid_size, grid_size), dtype=np.float32)]

    def get_legal_moves(self):
        moves = self.states[-1] == 0
        return moves.reshape(-1)

    def make_move(self, index: int):
        c = index // self.grid_size
        r = index % self.grid_size
        last_state = np.copy(self.states[-1])
        last_state[c][r] = 1
        self.states.append(last_state)

    def is_terminal(self, idx: int = -1) -> bool:
        n_collinear, _ = find_collinear(self.states[idx])
        return n_collinear > 0

    def calculate_reward(self) -> float:
        correct_idx = -1
        while self.is_terminal(correct_idx):
            correct_idx -= 1
        n_placed = int(np.sum(self.states[correct_idx]))
        if n_placed == 2 * self.grid_size:
            return 2 * self.grid_size, 1
        return n_placed, ((n_placed - self.grid_size) / self.grid_size)

def draw_grid(grid):
    for row in grid:
        row_str = "".join(map(str, list(int(r) for r in row)))
        row_str = row_str.replace("0", "⬜")
        row_str = row_str.replace("1", "⬛")
        print(row_str)

def main():
    N = int(input('Grid size: '))
    best_reward = -1
    while True:
        game = NoThreeInLine(N)
        while not game.is_terminal():
            legal_moves = game.get_legal_moves()
            legal_move_idxs = [i for i in range(len(legal_moves)) if legal_moves[i]]
            move = random.choice(legal_move_idxs)
            game.make_move(move)
        
        n_placed, reward = game.calculate_reward()
        if reward > best_reward:
            best_reward = reward
            print('--', n_placed, '--')
            draw_grid(game.states[n_placed])
            if best_reward == 1:
                break

if __name__ == '__main__':
    main()
