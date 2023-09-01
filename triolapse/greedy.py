import random
import threading
import time

import numpy as np

from game import NoThreeInLine
from visuals import draw_grid


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
    time.sleep(4)
    while True:
        n_placed, state = generate_greedy_solution(N)
        callback((n_placed, state, _id))


def main():
    N = int(input("Grid size: "))
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
                print("-" * text_length, n_placed, "-" * text_length, "\tfrom:", _id)
                draw_grid(state)
                if highest_n_placed == 2 * N:
                    should_stop = True
                    return True
            return False

    threads = []

    for i in range(32):
        thread = threading.Thread(
            target=greedy_solution_wrapper,
            args=(N, result_callback, i + 1),
            daemon=True,
        )
        threads.append(thread)
        thread.start()
        print("starting thread", i + 1)

    while not should_stop:
        pass


if __name__ == "__main__":
    main()
