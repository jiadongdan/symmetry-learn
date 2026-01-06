import numpy as np


def get_line(p1, p2):
    """
    Get integer positions of line points connecting p1 and p2.
    Uses Bresenham's line algorithm for efficient integer line drawing.

    Parameters
    ----------
    p1 : array-like, shape (2,)
        Starting point (x, y) in float
    p2 : array-like, shape (2,)
        Ending point (x, y) in float

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        Array of integer (x, y) coordinates along the line
    """
    # Convert to integers
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))

    points = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Determine direction of line
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    err = dx - dy

    x, y = x1, y1

    while True:
        points.append([x, y])

        # Reached endpoint
        if x == x2 and y == y2:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return np.array(points)

