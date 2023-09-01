from collections import namedtuple
import threading
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

def generate_greedy_solution(N: int) -> (int, np.ndarray):
    game = NoThreeInLine(N)
    column_counters = [0] * N
    moves = list(range(N))
    
    for row in range(N):
        for _ in range(2):
            move = random.choice(moves)
            column_counters[move] += 1
            if column_counters[move] == 2:
                moves.remove(move)
            game.make_move(move + row * N)

    n_placed, _ = game.calculate_reward()
    return n_placed, game.states[-1]

def greedy_solution_wrapper(N, callback, _id):
    while True:
        n_placed, state = generate_greedy_solution(N)
        callback((n_placed, state, _id))

def main():
    N = int(input('Grid size: '))
    highest_n_placed = -1
    lock = threading.Lock()
    should_stop = False

    def result_callback(result):
        nonlocal highest_n_placed, should_stop
        n_placed, state, _id = result
        with lock:
            if n_placed > highest_n_placed:
                highest_n_placed = n_placed
                text_length = (N + len(str(n_placed))) // 2
                print('-' * text_length, n_placed, '-' * text_length, 'from:', _id)
                draw_grid(state)
                if highest_n_placed == 2 * N:
                    should_stop = True
                    return True
            return False

    threads = []

    for i in range(32):
        thread = threading.Thread(target=greedy_solution_wrapper, args=(N, result_callback, i + 1), daemon=True)
        threads.append(thread)
        thread.start()
        print('starting thread', i+1)

    while not should_stop:
        pass

if __name__ == '__main__':
    main()
