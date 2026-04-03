"""
MazeEnv — environnement Gymnasium avec mémoire séparée cellules/murs.

Vecteur d'observation (284 valeurs pour 10×10) :
  [0 : MEM_SIZE]                  cell_memory — état traversal :
                                      -1 = jamais observée
                                       0 = observée (FOV) mais pas traversée
                                       1 = traversée physiquement
  [MEM_SIZE : MEM_SIZE+WALL_SIZE] wall_memory — 180 murs internes uniques :
                                      -1 = mur pas encore révélé
                                       0 = pas de mur (passage libre)
                                       1 = mur présent
                                   index 0..89  : murs horizontaux h[r,c] entre (r,c)↔(r+1,c)
                                   index 90..179: murs verticaux   v[r,c] entre (r,c)↔(r,c+1)
                                   Les murs de bordure ne sont pas stockés (toujours présents).
  [MEM_SIZE+WALL_SIZE]            row normalisée ∈ [0, 1]
  [MEM_SIZE+WALL_SIZE+1]          col normalisée ∈ [0, 1]
  [MEM_SIZE+WALL_SIZE+2 : +4]     delta vers objectif normalisé ∈ [-1, 1]

Actions : 0=NORTH  1=SOUTH  2=EAST  3=WEST
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from maze import Maze, generate_maze, NORTH, SOUTH, EAST, WEST, DELTA

MAZE_SIZE  = 10
EXTRA_OPEN = 5
VIS_RANGE  = 3
MAX_STEPS  = 4 * MAZE_SIZE * MAZE_SIZE

MEM_SIZE   = MAZE_SIZE * MAZE_SIZE                    # 100
WALL_SIZE  = 2 * MAZE_SIZE * (MAZE_SIZE - 1)          # 180
H_WALLS    = (MAZE_SIZE - 1) * MAZE_SIZE              # 90 (offset murs verticaux)
OBS_DIM    = MEM_SIZE + WALL_SIZE + 2 + 2              # 284

ACTION_DIRS = [NORTH, SOUTH, EAST, WEST]
RAYS = [(NORTH, -1, 0), (SOUTH, 1, 0), (EAST, 0, 1), (WEST, 0, -1)]

_CORNERS = [(0, 0), (0, MAZE_SIZE - 1), (MAZE_SIZE - 1, 0), (MAZE_SIZE - 1, MAZE_SIZE - 1)]


def _wall_idx(r: int, c: int, direction, rows: int, cols: int) -> int:
    """
    Retourne l'index unique du mur dans wall_memory, ou -1 si bordure.
    h[r,c] = mur entre (r,c) et (r+1,c)  → index r*cols + c         (r < rows-1)
    v[r,c] = mur entre (r,c) et (r,c+1)  → index H_WALLS + r*(cols-1) + c  (c < cols-1)
    """
    h = (rows - 1) * cols
    if direction == NORTH:
        return -1 if r == 0        else (r - 1) * cols + c
    if direction == SOUTH:
        return -1 if r == rows - 1 else r * cols + c
    if direction == EAST:
        return -1 if c == cols - 1 else h + r * (cols - 1) + c
    # WEST
    return -1 if c == 0            else h + r * (cols - 1) + (c - 1)


class MazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        low  = np.full(OBS_DIM, -1.0, dtype=np.float32)
        high = np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

        self.maze       : Maze | None = None
        self.player_row = 0
        self.player_col = 0
        self.end_row    = MAZE_SIZE - 1
        self.end_col    = MAZE_SIZE - 1
        self.step_count = 0

        self.cell_memory = np.full(MEM_SIZE,  -1.0, dtype=np.float32)
        self.wall_memory = np.full(WALL_SIZE, -1.0, dtype=np.float32)

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.maze       = generate_maze(MAZE_SIZE, MAZE_SIZE, EXTRA_OPEN)
        self.step_count = 0
        self.cell_memory[:] = -1.0
        self.wall_memory[:] = -1.0

        # Coins de départ et d'arrivée aléatoires (distincts)
        start, end = self.np_random.choice(len(_CORNERS), size=2, replace=False)
        self.player_row, self.player_col = _CORNERS[start]
        self.end_row,    self.end_col    = _CORNERS[end]

        # Marquer la case de départ comme traversée
        self.cell_memory[self.player_row * MAZE_SIZE + self.player_col] = 1.0

        self._update_memory()
        return self._obs(), {}

    def step(self, action: int):
        assert self.maze is not None
        d      = ACTION_DIRS[action]
        dr, dc = DELTA[d]

        if self.maze.has_wall(self.player_row, self.player_col, d):
            reward = -0.3
        else:
            self.player_row += dr
            self.player_col += dc
            pos = self.player_row * self.maze.cols + self.player_col
            if self.cell_memory[pos] >= 1.0:   # déjà traversée
                reward = -0.3
            else:
                reward = -0.1
            self.cell_memory[pos] = 1.0

        self.step_count += 1

        cells_before = int((self.cell_memory >= 0).sum())
        self._update_memory()
        cells_after  = int((self.cell_memory >= 0).sum())
        if cells_after > cells_before:
            reward += 0.2 * (cells_after - cells_before)

        terminated = (self.player_row == self.end_row and
                      self.player_col == self.end_col)
        if terminated:
            reward = 50.0

        truncated = self.step_count >= MAX_STEPS
        return self._obs(), reward, terminated, truncated, {}

    # ── Mémoire ────────────────────────────────────────────────────────────────

    def _update_memory(self) -> None:
        m    = self.maze
        r, c = self.player_row, self.player_col
        self._observe_cell(r, c)
        for d, dr, dc in RAYS:
            cr, cc = r, c
            for _ in range(VIS_RANGE):
                if m.has_wall(cr, cc, d):
                    break
                cr += dr; cc += dc
                self._observe_cell(cr, cc)

    def _observe_cell(self, r: int, c: int) -> None:
        m   = self.maze
        idx = r * m.cols + c
        if self.cell_memory[idx] < 0:
            self.cell_memory[idx] = 0.0
        # Révéler les murs internes de cette cellule
        for direction in [NORTH, SOUTH, EAST, WEST]:
            widx = _wall_idx(r, c, direction, m.rows, m.cols)
            if widx >= 0:
                self.wall_memory[widx] = 1.0 if m.has_wall(r, c, direction) else 0.0

    # ── Observation ────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        m    = self.maze
        r, c = self.player_row, self.player_col

        pos = np.array([r / (m.rows - 1), c / (m.cols - 1)], dtype=np.float32)

        goal_delta = np.array([
            (self.end_row - r) / (m.rows - 1),
            (self.end_col - c) / (m.cols - 1),
        ], dtype=np.float32)

        return np.concatenate([
            self.cell_memory,   # (100,)
            self.wall_memory,   # (180,)
            pos,                # (2,)
            goal_delta,         # (2,)
        ]).astype(np.float32)
