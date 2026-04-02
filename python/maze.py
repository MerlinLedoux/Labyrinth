"""
Maze — miroir exact de src/core/maze.ts
Bitmask : NORTH=1  SOUTH=2  EAST=4  WEST=8
"""
import random

NORTH = 1
SOUTH = 2
EAST  = 4
WEST  = 8
ALL_WALLS = NORTH | SOUTH | EAST | WEST

OPPOSITE = { NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST }
DELTA    = { NORTH: (-1, 0), SOUTH: (1, 0), EAST: (0, 1), WEST: (0, -1) }
DIRS     = [NORTH, SOUTH, EAST, WEST]


class Maze:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        # walls[row][col] = bitmask des murs restants
        self.walls = [[ALL_WALLS] * cols for _ in range(rows)]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def has_wall(self, row: int, col: int, direction: int) -> bool:
        return bool(self.walls[row][col] & direction)

    def remove_wall(self, r1: int, c1: int, r2: int, c2: int) -> None:
        dr, dc = r2 - r1, c2 - c1
        if   (dr, dc) == (-1, 0): d = NORTH
        elif (dr, dc) == ( 1, 0): d = SOUTH
        elif (dr, dc) == ( 0, 1): d = EAST
        else:                      d = WEST
        self.walls[r1][c1] &= ~d
        self.walls[r2][c2] &= ~OPPOSITE[d]

    def passable_neighbors(self, row: int, col: int):
        result = []
        for d in DIRS:
            if not self.has_wall(row, col, d):
                dr, dc = DELTA[d]
                nr, nc = row + dr, col + dc
                if self.in_bounds(nr, nc):
                    result.append((nr, nc))
        return result


def generate_maze(rows: int, cols: int, extra_openings: int = 0) -> Maze:
    """Recursive backtracker + extra openings aléatoires."""
    m = Maze(rows, cols)

    # Recursive backtracker
    visited = [[False] * cols for _ in range(rows)]
    stack = [(0, 0)]
    visited[0][0] = True
    while stack:
        r, c = stack[-1]
        neighbors = [
            (r + dr, c + dc)
            for d in DIRS
            for dr, dc in [DELTA[d]]
            if m.in_bounds(r + dr, c + dc) and not visited[r + dr][c + dc]
        ]
        if not neighbors:
            stack.pop()
        else:
            nr, nc = random.choice(neighbors)
            m.remove_wall(r, c, nr, nc)
            visited[nr][nc] = True
            stack.append((nr, nc))

    # Extra openings — Fisher-Yates puis on prend les N premiers
    walls = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and m.has_wall(r, c, EAST):
                walls.append((r, c, r, c + 1))
            if r + 1 < rows and m.has_wall(r, c, SOUTH):
                walls.append((r, c, r + 1, c))
    random.shuffle(walls)
    for r1, c1, r2, c2 in walls[:extra_openings]:
        m.remove_wall(r1, c1, r2, c2)

    return m
