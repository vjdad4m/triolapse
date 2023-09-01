from collections import namedtuple

import numpy as np

Point = namedtuple("Point", ["x", "y"])


def _turn(a: Point, b: Point, c: Point) -> int:
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
                if _turn(p[i], p[j], p[k]) == 0:
                    r += 1
                    points.append([p[i], p[j], p[k]])

    return r, points
