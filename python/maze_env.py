"""
MazeEnv — environnement Gymnasium avec mémoire des cellules observées.

Vecteur d'observation (232 valeurs) :
  [0:225]   mémoire — bitmask murs de chaque cellule (N=1,E=2,S=4,W=8),
             -1 si la cellule n'a pas encore été observée.
  [225]     position joueur — indice normalisé ∈ [0, 1]
  [226:230] vision immédiate — [N, E, S, W] : 1=mur présent, 0=libre
  [230:232] delta vers objectif — [(end_r−r)/(SIZE−1), (end_c−c)/(SIZE−1)] ∈ [−1, 1]

Actions : 0=NORTH  1=SOUTH  2=EAST  3=WEST
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from maze import Maze, generate_maze, NORTH, SOUTH, EAST, WEST, DELTA

MAZE_SIZE  = 15
EXTRA_OPEN = 20
VIS_RANGE  = 3
MAX_STEPS  = 600

# Taille de chaque bloc
MEM_SIZE    = MAZE_SIZE * MAZE_SIZE          # 225
OBS_DIM     = MEM_SIZE + 1 + 4 + 2          # 232

ACTION_DIRS = [NORTH, SOUTH, EAST, WEST]
RAYS = [(NORTH, -1, 0), (SOUTH, 1, 0), (EAST, 0, 1), (WEST, 0, -1)]

# Convention bitmask pour les observations (indépendante du stockage interne)
_OBS_N, _OBS_E, _OBS_S, _OBS_W = 1, 2, 4, 8

def _cell_obs_bits(m: Maze, r: int, c: int) -> int:
    """Bitmask des murs de (r,c) selon la convention N=1,E=2,S=4,W=8."""
    return (
        (_OBS_N if m.has_wall(r, c, NORTH) else 0) |
        (_OBS_E if m.has_wall(r, c, EAST)  else 0) |
        (_OBS_S if m.has_wall(r, c, SOUTH) else 0) |
        (_OBS_W if m.has_wall(r, c, WEST)  else 0)
    )


class MazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        low  = np.array([-1.0]*MEM_SIZE + [0.0] + [0.0]*4 + [-1.0]*2, dtype=np.float32)
        high = np.array([15.0]*MEM_SIZE + [1.0] + [1.0]*4 + [ 1.0]*2, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

        self.maze       : Maze | None = None
        self.player_row = 0
        self.player_col = 0
        self.end_row    = MAZE_SIZE - 1
        self.end_col    = MAZE_SIZE - 1
        self.step_count = 0
        self.memory     = np.full(MEM_SIZE, -1.0, dtype=np.float32)

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.maze       = generate_maze(MAZE_SIZE, MAZE_SIZE, EXTRA_OPEN)
        self.player_row = 0
        self.player_col = 0
        self.end_row    = MAZE_SIZE - 1
        self.end_col    = MAZE_SIZE - 1
        self.step_count = 0
        self.memory[:]  = -1.0

        # Observer la position de départ
        self._update_memory()
        return self._obs(), {}

    def step(self, action: int):
        assert self.maze is not None
        d      = ACTION_DIRS[action]
        dr, dc = DELTA[d]

        if self.maze.has_wall(self.player_row, self.player_col, d):
            reward = -0.5                       # tentative de traverser un mur
        else:
            self.player_row += dr
            self.player_col += dc
            reward = -1.0                       # coût par déplacement

        self.step_count += 1
        self._update_memory()

        terminated = (self.player_row == self.end_row and
                      self.player_col == self.end_col)
        if terminated:
            reward = 100.0

        truncated = self.step_count >= MAX_STEPS

        return self._obs(), reward, terminated, truncated, {}

    # ── Mémoire ──────────────────────────────────────────────────────────────

    def _update_memory(self) -> None:
        """Met à jour la mémoire avec toutes les cellules visibles depuis la position actuelle."""
        m    = self.maze
        r, c = self.player_row, self.player_col

        self._memorize(r, c)

        for d, dr, dc in RAYS:
            cr, cc = r, c
            for _ in range(VIS_RANGE):
                if m.has_wall(cr, cc, d):
                    break
                cr += dr
                cc += dc
                self._memorize(cr, cc)

    def _memorize(self, r: int, c: int) -> None:
        """Enregistre le bitmask de murs de (r, c) dans la mémoire."""
        self.memory[r * self.maze.cols + c] = float(_cell_obs_bits(self.maze, r, c))

    # ── Observation ──────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        m    = self.maze
        r, c = self.player_row, self.player_col
        size = m.rows
        total = size * size

        player_idx = (r * m.cols + c) / (total - 1)   # normalisé [0, 1]

        immediate = np.array([
            1.0 if m.has_wall(r, c, NORTH) else 0.0,
            1.0 if m.has_wall(r, c, EAST)  else 0.0,
            1.0 if m.has_wall(r, c, SOUTH) else 0.0,
            1.0 if m.has_wall(r, c, WEST)  else 0.0,
        ], dtype=np.float32)

        goal_delta = np.array([
            (self.end_row - r) / (size - 1),
            (self.end_col - c) / (size - 1),
        ], dtype=np.float32)

        return np.concatenate([
            self.memory,
            [player_idx],
            immediate,
            goal_delta,
        ]).astype(np.float32)
