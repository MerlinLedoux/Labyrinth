"""
VecMazeEnv — Environnement RL vectorisé, zéro transfert CPU↔GPU pendant l'entraînement.

Grille 10×10, générée par recursive backtracker + passages supplémentaires.
N labyrinthes tournent en parallèle comme tenseurs PyTorch sur un seul device.

Observation (284 valeurs) :
  cell_mem  [100]  : -1 jamais vue, 0 vue (FOV) non visitée, 1 visitée physiquement
  wall_mem  [180]  : -1 jamais vue, 0 pas de mur, 1 mur
                     index   0..89 : murs horizontaux h[r,c] — entre (r,c) et (r+1,c)
                     index 90..179 : murs verticaux   v[r,c] — entre (r,c) et (r,c+1)
  pos       [2]    : [row/(ROWS-1), col/(COLS-1)]
  goal      [2]    : [(end_r - r)/(ROWS-1), (end_c - c)/(COLS-1)]

Actions : 0=N  1=S  2=E  3=W

Rewards :
  +50.0  objectif atteint
  +0.2   déplacement sur une case jamais visitée (cell_mem < 1)
  -0.1   par pas (coût du temps)
  -0.5   collision avec un mur
"""

import random
import torch

# ── Constantes ────────────────────────────────────────────────────────────────
ROWS       = 10
COLS       = 10
EXTRA_OPEN = 5          # passages supplémentaires après génération
VIS_RANGE  = 3          # portée du FOV (ray-casting cardinal)
MAX_STEPS  = 4 * ROWS * COLS   # 400

CELL_SIZE = ROWS * COLS                   # 100
H_WALLS   = (ROWS - 1) * COLS            # 90 : murs horizontaux internes
V_WALLS   = ROWS * (COLS - 1)            # 90 : murs verticaux internes
WALL_SIZE = H_WALLS + V_WALLS            # 180
OBS_DIM   = CELL_SIZE + WALL_SIZE + 4    # 284

POOL_SIZE = 500

# Déplacements par direction [N, S, E, W]
_DR  = [-1,  1,  0,  0]
_DC  = [ 0,  0,  1, -1]
_OPP = [ 1,  0,  3,  2]   # direction opposée


# ── Génération de labyrinthes (CPU, une seule fois) ───────────────────────────

def _gen_maze(rows: int, cols: int, extra: int) -> list:
    """
    Recursive backtracker.
    Retourne walls[r][c][d] bool — True = mur présent.
    Directions : 0=N  1=S  2=E  3=W
    """
    walls   = [[[True] * 4 for _ in range(cols)] for _ in range(rows)]
    visited = [[False] * cols for _ in range(rows)]

    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        r, c = stack[-1]
        neighbors = [
            (d, r + _DR[d], c + _DC[d])
            for d in range(4)
            if 0 <= r + _DR[d] < rows
            and 0 <= c + _DC[d] < cols
            and not visited[r + _DR[d]][c + _DC[d]]
        ]
        if neighbors:
            d, nr, nc = random.choice(neighbors)
            walls[r][c][d]          = False
            walls[nr][nc][_OPP[d]]  = False
            visited[nr][nc]         = True
            stack.append((nr, nc))
        else:
            stack.pop()

    # Passages supplémentaires (ouvrir des murs Sud aléatoires)
    opened = 0
    attempts = 0
    while opened < extra and attempts < extra * 10:
        attempts += 1
        r = random.randint(0, rows - 2)
        c = random.randint(0, cols - 1)
        if walls[r][c][1]:          # mur Sud encore présent
            walls[r][c][1]       = False
            walls[r + 1][c][0]   = False
            opened += 1

    return walls


def _build_pool(n: int, device: torch.device):
    """
    Génère n labyrinthes sur CPU puis transfère sur device en un seul appel.

    Retourne :
      pool_full (n, ROWS, COLS, 4) bool  — murs complets (pour step)
      pool_flat (n, WALL_SIZE)     float — 180 murs uniques (pour wall_mem)
    """
    pool_full = torch.zeros(n, ROWS, COLS, 4, dtype=torch.bool)
    pool_flat = torch.zeros(n, WALL_SIZE)

    for i in range(n):
        w = _gen_maze(ROWS, COLS, EXTRA_OPEN)

        for r in range(ROWS):
            for c in range(COLS):
                for d in range(4):
                    pool_full[i, r, c, d] = w[r][c][d]

        # Murs horizontaux : mur Sud de (r,c) = h_wall entre (r,c) et (r+1,c)
        for r in range(ROWS - 1):
            for c in range(COLS):
                pool_flat[i, r * COLS + c] = float(w[r][c][1])

        # Murs verticaux : mur Est de (r,c) = v_wall entre (r,c) et (r,c+1)
        for r in range(ROWS):
            for c in range(COLS - 1):
                pool_flat[i, H_WALLS + r * (COLS - 1) + c] = float(w[r][c][2])

    return pool_full.to(device), pool_flat.to(device)


def _build_wall_idx(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pour chaque cellule et chaque direction, index du mur dans wall_mem.
    wall_idx  (CELL_SIZE, 4) long  — -1 si mur de bordure (non stocké)
    is_border (CELL_SIZE, 4) bool
    """
    idx = torch.full((CELL_SIZE, 4), -1, dtype=torch.long)

    for r in range(ROWS):
        for c in range(COLS):
            cell = r * COLS + c
            if r > 0:       idx[cell, 0] = (r - 1) * COLS + c              # N
            if r < ROWS-1:  idx[cell, 1] = r * COLS + c                    # S
            if c < COLS-1:  idx[cell, 2] = H_WALLS + r * (COLS-1) + c      # E
            if c > 0:       idx[cell, 3] = H_WALLS + r * (COLS-1) + (c-1)  # W

    idx = idx.to(device)
    return idx, idx < 0


# ── Environnement vectorisé ───────────────────────────────────────────────────

class VecMazeEnv:
    """
    N labyrinthes en parallèle sur un device (CPU ou CUDA).
    Aucun transfert de données pendant step() ou _obs().
    """

    def __init__(self, n_envs: int = 32, device: str = "cuda"):
        self.n   = n_envs
        self.dev = torch.device(device)

        # ── Pool de labyrinthes ───────────────────────────────────────────────
        print(f"Génération du pool ({POOL_SIZE} labyrinthes)...", flush=True)
        self._pool_full, self._pool_flat = _build_pool(POOL_SIZE, self.dev)
        print("Pool prêt.", flush=True)

        # ── Tables de lookup ──────────────────────────────────────────────────
        self._wall_idx, self._is_border = _build_wall_idx(self.dev)

        # ── Constantes sur device ─────────────────────────────────────────────
        self._dr = torch.tensor(_DR, dtype=torch.long, device=self.dev)
        self._dc = torch.tensor(_DC, dtype=torch.long, device=self.dev)
        self._ei = torch.arange(n_envs, device=self.dev)

        self._corners = torch.tensor(
            [[0, 0], [0, COLS-1], [ROWS-1, 0], [ROWS-1, COLS-1]],
            dtype=torch.long, device=self.dev
        )

        # ── État ──────────────────────────────────────────────────────────────
        self.walls      = torch.zeros(n_envs, ROWS, COLS, 4,
                                      dtype=torch.bool, device=self.dev)
        self.walls_flat = torch.zeros(n_envs, WALL_SIZE, device=self.dev)
        self.pos        = torch.zeros(n_envs, 2, dtype=torch.long, device=self.dev)
        self.end_pos    = torch.zeros(n_envs, 2, dtype=torch.long, device=self.dev)
        self.cell_mem   = torch.full((n_envs, CELL_SIZE), -1., device=self.dev)
        self.wall_mem   = torch.full((n_envs, WALL_SIZE), -1., device=self.dev)
        self.step_count = torch.zeros(n_envs, dtype=torch.long, device=self.dev)

        mask = torch.ones(n_envs, dtype=torch.bool, device=self.dev)
        self._reset_envs(mask)
        self._update_fov()

    # ── API publique ──────────────────────────────────────────────────────────

    def reset(self) -> torch.Tensor:
        mask = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        self._reset_envs(mask)
        self._update_fov()
        return self._obs()

    def step(self, actions: torch.Tensor):
        """
        actions : (N,) long, sur self.dev
        Retourne obs, reward, terminated, truncated — tous sur self.dev
        """
        r = self.pos[:, 0]
        c = self.pos[:, 1]

        hit_wall = self.walls[self._ei, r, c, actions]
        moved    = ~hit_wall

        new_r = (r + self._dr[actions]).clamp(0, ROWS - 1)
        new_c = (c + self._dc[actions]).clamp(0, COLS - 1)
        self.pos[:, 0] = torch.where(moved, new_r, r)
        self.pos[:, 1] = torch.where(moved, new_c, c)
        self.step_count += 1

        # Récompenses
        reward = torch.where(
            hit_wall,
            torch.full((self.n,), -0.5, device=self.dev),
            torch.full((self.n,), -0.1, device=self.dev),
        )

        # Bonus case jamais visitée (+0.2)
        pos_idx  = self.pos[:, 0] * COLS + self.pos[:, 1]
        new_cell = moved & (self.cell_mem[self._ei, pos_idx] < 1.)
        reward   = torch.where(new_cell, reward + 0.2, reward)

        # Marquer comme visitée (1)
        self.cell_mem[self._ei[moved], pos_idx[moved]] = 1.

        # FOV
        self._update_fov()

        # Victoire
        terminated = (
            (self.pos[:, 0] == self.end_pos[:, 0]) &
            (self.pos[:, 1] == self.end_pos[:, 1])
        )
        reward = torch.where(
            terminated,
            torch.full((self.n,), 50., device=self.dev),
            reward,
        )

        truncated = self.step_count >= MAX_STEPS
        done      = terminated | truncated

        # Observation de l'état courant (avant reset)
        obs = self._obs()

        # Auto-reset des envs terminés, puis remplacer leur obs par l'état initial
        if done.any():
            self._reset_envs(done)
            self._update_fov()
            obs[done] = self._obs()[done]

        return obs, reward, terminated, truncated

    # ── Privé ─────────────────────────────────────────────────────────────────

    def _reset_envs(self, mask: torch.Tensor) -> None:
        n_reset = int(mask.sum().item())
        if n_reset == 0:
            return

        idx = torch.randint(0, POOL_SIZE, (n_reset,), device=self.dev)
        self.walls[mask]      = self._pool_full[idx]
        self.walls_flat[mask] = self._pool_flat[idx]
        self.step_count[mask] = 0
        self.cell_mem[mask]   = -1.
        self.wall_mem[mask]   = -1.

        # Start et end : 2 coins distincts tirés aléatoirement
        pairs = torch.stack([
            torch.randperm(4, device=self.dev)[:2] for _ in range(n_reset)
        ])                                              # (n_reset, 2)
        start = self._corners[pairs[:, 0]]             # (n_reset, 2)
        end   = self._corners[pairs[:, 1]]             # (n_reset, 2)

        self.pos[mask]     = start
        self.end_pos[mask] = end

        # Case de départ : directement visitée (1)
        env_ids    = self._ei[mask]
        start_cell = start[:, 0] * COLS + start[:, 1]
        self.cell_mem[env_ids, start_cell] = 1.

    def _update_fov(self) -> None:
        """Ray-casting depuis chaque agent — révèle cell_mem et wall_mem."""
        # Cellule courante toujours visible
        cur_idx = self.pos[:, 0] * COLS + self.pos[:, 1]
        self._reveal_cells(self._ei, cur_idx)

        for d in range(4):
            dr = int(self._dr[d].item())
            dc = int(self._dc[d].item())

            cur_r  = self.pos[:, 0].clone()
            cur_c  = self.pos[:, 1].clone()
            active = torch.ones(self.n, dtype=torch.bool, device=self.dev)

            for _ in range(VIS_RANGE):
                blocked = self.walls[self._ei, cur_r, cur_c, d]
                active  = active & ~blocked

                cur_r = torch.where(active, (cur_r + dr).clamp(0, ROWS-1), cur_r)
                cur_c = torch.where(active, (cur_c + dc).clamp(0, COLS-1), cur_c)

                a_ei    = self._ei[active]
                a_cells = (cur_r * COLS + cur_c)[active]
                if a_ei.numel() > 0:
                    self._reveal_cells(a_ei, a_cells)

    def _reveal_cells(self, env_ids: torch.Tensor, cell_ids: torch.Tensor) -> None:
        """Passe cell_mem de -1 → 0 (sans écraser 1) et révèle wall_mem."""
        cur = self.cell_mem[env_ids, cell_ids]
        self.cell_mem[env_ids, cell_ids] = torch.where(
            cur < 0., torch.zeros_like(cur), cur
        )

        widx      = self._wall_idx[cell_ids]    # (batch, 4)
        is_border = self._is_border[cell_ids]   # (batch, 4)

        for d in range(4):
            valid = ~is_border[:, d]
            if not valid.any():
                continue
            e  = env_ids[valid]
            wi = widx[valid, d]
            self.wall_mem[e, wi] = self.walls_flat[e, wi]

    def _obs(self) -> torch.Tensor:
        row_n = self.pos[:, 0].float() / (ROWS - 1)
        col_n = self.pos[:, 1].float() / (COLS - 1)
        pos   = torch.stack([row_n, col_n], dim=1)

        dy   = (self.end_pos[:, 0].float() - self.pos[:, 0].float()) / (ROWS - 1)
        dx   = (self.end_pos[:, 1].float() - self.pos[:, 1].float()) / (COLS - 1)
        goal = torch.stack([dy, dx], dim=1)

        return torch.cat([self.cell_mem, self.wall_mem, pos, goal], dim=1)
