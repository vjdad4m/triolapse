from collections import namedtuple

import numpy as np

# Define a named tuple "Point" with attributes "x" and "y" for representing 2D points.
Point = namedtuple("Point", ["x", "y"])


def _turn(a: Point, b: Point, c: Point) -> int:
    """
    Determine the orientation of three points (a, b, c) using the cross product.

    Args:
        a (Point): The first point.
        b (Point): The second point.
        c (Point): The third point.

    Returns:
        int: 1 if the points are in counterclockwise order, -1 if in clockwise order, 0 if collinear.
    """
    value = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
    if value > 0:
        return 1
    elif value < 0:
        return -1
    return 0


def find_collinear(m: np.ndarray) -> (int, np.ndarray):
    """
    Find collinear points in a given binary matrix.

    Args:
        m (np.ndarray): Binary matrix where 1 indicates the presence of a point.

    Returns:
        tuple: A tuple containing the number of collinear point sets and a list of collinear point sets.
    """
    n_collinear = 0  # Initialize the count of collinear point sets to 0
    p = []  # Initialize a list to store points with value 1 in the matrix
    points = [] # Initialize a list to store collinear point sets

    m_size = m.shape[0]

    # Iterate through the matrix to find points with value 1 and store them in the "p" list
    for i in range(m_size):
        for j in range(m_size):
            if m[i][j]:
                p.append(Point(i, j))

    p_size = len(p)

    # Iterate through combinations of three points to check for collinearity
    for i in range(p_size):
        for j in range(i + 1, p_size):
            for k in range(j + 1, p_size):
                if _turn(p[i], p[j], p[k]) == 0:
                    # If the points are collinear, increment the count and store the collinear points
                    n_collinear += 1
                    points.append([p[i], p[j], p[k]])

    return n_collinear, points
