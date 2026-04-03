"""
VecMazeEnv — N labyrinthes en parallèle, entièrement en tenseurs PyTorch.

Observation (284 valeurs pour 10×10) :
  cell_memory  (MEM_SIZE,)     : -1 jamais vue, 0 vue (FOV), 1 traversée
  wall_memory  (WALL_SIZE,)    : -1 pas encore révélé, 0 libre, 1 mur
                                  index 0..H_WALLS-1  : murs horizontaux h[r,c]
                                  index H_WALLS..179  : murs verticaux   v[r,c]
  pos          (2,)            : [row/9, col/9] normalisés indépendamment ∈ [0, 1]
  goal         (2,)            : delta normalisé vers objectif
"""

import torch
from maze import generate_maze, NORTH, SOUTH, EAST, WEST
from maze_env import MAZE_SIZE, EXTRA_OPEN, VIS_RANGE, MAX_STEPS, OBS_DIM, MEM_SIZE, WALL_SIZE, H_WALLS

# ── Constantes ─────────────────────────────────────────────────────────────────

_DELTA_R = torch.tensor([-1,  1,  0,  0], dtype=torch.long)
_DELTA_C = torch.tensor([ 0,  0,  1, -1], dtype=torch.long)

POOL_SIZE = 500


# ── Génération du pool ─────────────────────────────────────────────────────────

def _build_pool(n: int, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Génère n labyrinthes.
    Retourne :
      walls_full (n, rows, cols, 4) bool  — [N, S, E, W] par cellule (pour step)
      walls_flat (n, WALL_SIZE)    float  — 180 murs uniques (pour wall_memory init)
    """
    dirs = [NORTH, SOUTH, EAST, WEST]
    h    = (rows - 1) * cols   # nb murs horizontaux

    walls_full = torch.zeros(n, rows, cols, 4, dtype=torch.bool)
    walls_flat = torch.zeros(n, WALL_SIZE)

    for i in range(n):
        m = generate_maze(rows, cols, EXTRA_OPEN)
        for r in range(rows):
            for c in range(cols):
                for d, direction in enumerate(dirs):
                    walls_full[i, r, c, d] = m.has_wall(r, c, direction)

        # Murs horizontaux : h[r,c] = mur entre (r,c) et (r+1,c)
        for r in range(rows - 1):
            for c in range(cols):
                walls_flat[i, r * cols + c] = float(m.has_wall(r, c, SOUTH))

        # Murs verticaux : v[r,c] = mur entre (r,c) et (r,c+1)
        for r in range(rows):
            for c in range(cols - 1):
                walls_flat[i, h + r * (cols - 1) + c] = float(m.has_wall(r, c, EAST))

    return walls_full, walls_flat


# ── Tables de lookup pour wall_memory ─────────────────────────────────────────

def _build_wall_index_tables(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pour chaque cellule et chaque direction, précalcule l'index dans wall_memory.
    wall_idx  (rows*cols, 4) long  — index dans [0, WALL_SIZE-1], ou -1 si bordure
    is_border (rows*cols, 4) bool  — True si le mur est une bordure (pas stocké)
    directions : [N=0, S=1, E=2, W=3]
    """
    h    = (rows - 1) * cols
    size = rows * cols
    idx  = torch.full((size, 4), -1, dtype=torch.long)

    for r in range(rows):
        for c in range(cols):
            cell = r * cols + c
            # N : mur horizontal entre (r-1,c) et (r,c)
            if r > 0:
                idx[cell, 0] = (r - 1) * cols + c
            # S : mur horizontal entre (r,c) et (r+1,c)
            if r < rows - 1:
                idx[cell, 1] = r * cols + c
            # E : mur vertical entre (r,c) et (r,c+1)
            if c < cols - 1:
                idx[cell, 2] = h + r * (cols - 1) + c
            # W : mur vertical entre (r,c-1) et (r,c)
            if c > 0:
                idx[cell, 3] = h + r * (cols - 1) + (c - 1)

    is_border = idx < 0
    return idx, is_border


# ── Environnement vectorisé ───────────────────────────────────────────────────

class VecMazeEnv:
    def __init__(self, n_envs: int = 32, device: str = "cpu"):
        self.n    = n_envs
        self.rows = MAZE_SIZE
        self.cols = MAZE_SIZE
        self.size = MEM_SIZE
        self.dev  = torch.device(device)

        print(f"Generation du pool ({POOL_SIZE} labyrinthes)...", flush=True)
        self._pool_full, self._pool_flat = _build_pool(POOL_SIZE, self.rows, self.cols)
        print("Pool pret.", flush=True)

        # Tables de lookup (sur device)
        wall_idx, is_border = _build_wall_index_tables(self.rows, self.cols)
        self._wall_idx    = wall_idx.to(self.dev)     # (size, 4)
        self._is_border   = is_border.to(self.dev)    # (size, 4)

        self._delta_r = _DELTA_R.to(self.dev)
        self._delta_c = _DELTA_C.to(self.dev)
        self._env_idx = torch.arange(self.n, device=self.dev)

        # Murs complets (n, rows, cols, 4) bool — pour step()
        self.walls = torch.zeros(self.n, self.rows, self.cols, 4,
                                 dtype=torch.bool, device=self.dev)

        # Murs plats révélés (n, WALL_SIZE) float — pour observation
        # -1 = pas encore révélé, 0/1 = valeur du mur
        self.wall_memory = torch.full((self.n, WALL_SIZE), -1.0, device=self.dev)

        # Murs plats complets du labyrinthe (n, WALL_SIZE) float — source de vérité
        self.walls_flat = torch.zeros(self.n, WALL_SIZE, device=self.dev)

        self.pos     = torch.zeros(self.n, 2, dtype=torch.long, device=self.dev)
        self.end_pos = torch.zeros(self.n, 2, dtype=torch.long, device=self.dev)
        self.cell_memory = torch.full((self.n, self.size), -1.0, device=self.dev)
        self.step_count  = torch.zeros(self.n, dtype=torch.long, device=self.dev)

        # Coins disponibles (4 coins, 2 coordonnées)
        self._corners = torch.tensor(
            [[0, 0], [0, self.cols - 1], [self.rows - 1, 0], [self.rows - 1, self.cols - 1]],
            dtype=torch.long, device=self.dev
        )

        self._reset_envs(torch.ones(self.n, dtype=torch.bool, device=self.dev))
        self._update_memory()

    # ── API publique ──────────────────────────────────────────────────────────

    def reset(self) -> torch.Tensor:
        mask = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        self._reset_envs(mask)
        self._update_memory()
        return self._obs()

    def step(self, actions: torch.Tensor):
        r = self.pos[:, 0]
        c = self.pos[:, 1]

        hit_wall = self.walls[self._env_idx, r, c, actions]
        moved    = ~hit_wall

        new_r = (r + self._delta_r[actions]).clamp(0, self.rows - 1)
        new_c = (c + self._delta_c[actions]).clamp(0, self.cols - 1)
        self.pos[:, 0] = torch.where(moved, new_r, r)
        self.pos[:, 1] = torch.where(moved, new_c, c)

        self.step_count += 1

        reward = torch.where(hit_wall,
                             torch.full((self.n,), -0.3, device=self.dev),
                             torch.full((self.n,), -0.1, device=self.dev))

        # Pénalité revisit
        pos_idx      = self.pos[:, 0] * self.cols + self.pos[:, 1]
        already_trav = self.cell_memory[self._env_idx, pos_idx] >= 1.0
        reward       = torch.where(moved & already_trav, reward - 0.2, reward)

        env_moved    = self._env_idx[moved]
        cells_moved  = pos_idx[moved]
        self.cell_memory[env_moved, cells_moved] = 1.0

        # Bonus curiosité
        cells_before = (self.cell_memory >= 0).sum(dim=1).float()
        self._update_memory()
        cells_after  = (self.cell_memory >= 0).sum(dim=1).float()
        reward      += 0.2 * (cells_after - cells_before)

        terminated = (self.pos[:, 0] == self.end_pos[:, 0]) & \
                     (self.pos[:, 1] == self.end_pos[:, 1])
        reward     = torch.where(terminated,
                                 torch.full((self.n,), 50.0, device=self.dev),
                                 reward)
        truncated  = self.step_count >= MAX_STEPS
        done       = terminated | truncated

        obs = self._obs()

        if done.any():
            self._reset_envs(done)
            self._update_memory()

        return obs, reward, terminated, truncated

    # ── Privé ─────────────────────────────────────────────────────────────────

    def _reset_envs(self, mask: torch.Tensor) -> None:
        n_reset = int(mask.sum().item())
        if n_reset == 0:
            return
        idx = torch.randint(0, POOL_SIZE, (n_reset,))
        self.walls[mask]       = self._pool_full[idx].to(self.dev)
        self.walls_flat[mask]  = self._pool_flat[idx].to(self.dev)
        self.cell_memory[mask] = -1.0
        self.wall_memory[mask] = -1.0
        self.step_count[mask]  = 0

        # Tirer start et end parmi les 4 coins (distincts) pour chaque env resetté
        corner_pairs = torch.stack([
            torch.randperm(4)[:2] for _ in range(n_reset)
        ])                                                     # (n_reset, 2)
        start_idx = corner_pairs[:, 0]
        end_idx   = corner_pairs[:, 1]
        start_pos = self._corners[start_idx]                   # (n_reset, 2)
        end_pos   = self._corners[end_idx]                     # (n_reset, 2)

        self.pos[mask]     = start_pos
        self.end_pos[mask] = end_pos

        # Marquer la case de départ comme traversée
        start_cell = start_pos[:, 0] * self.cols + start_pos[:, 1]
        env_ids    = self._env_idx[mask]
        self.cell_memory[env_ids, start_cell] = 1.0

    def _update_memory(self) -> None:
        """Ray-casting cardinal, révèle cell_memory et wall_memory."""
        n = self.n

        pos_idx = self.pos[:, 0] * self.cols + self.pos[:, 1]
        self._observe_cells(self._env_idx, pos_idx)

        for dir_idx in range(4):
            dr = int(self._delta_r[dir_idx].item())
            dc = int(self._delta_c[dir_idx].item())

            cur_r  = self.pos[:, 0].clone()
            cur_c  = self.pos[:, 1].clone()
            active = torch.ones(n, dtype=torch.bool, device=self.dev)

            for _ in range(VIS_RANGE):
                wall   = self.walls[self._env_idx, cur_r, cur_c, dir_idx]
                active = active & ~wall

                cur_r = torch.where(active, (cur_r + dr).clamp(0, self.rows-1), cur_r)
                cur_c = torch.where(active, (cur_c + dc).clamp(0, self.cols-1), cur_c)

                active_envs  = self._env_idx[active]
                active_cells = (cur_r * self.cols + cur_c)[active]
                if active_envs.numel() > 0:
                    self._observe_cells(active_envs, active_cells)

    def _observe_cells(self, env_ids: torch.Tensor, cell_ids: torch.Tensor) -> None:
        """Marque les cellules comme vues et révèle leurs murs dans wall_memory."""
        # cell_memory : -1 → 0 (ne pas écraser 1)
        current = self.cell_memory[env_ids, cell_ids]
        self.cell_memory[env_ids, cell_ids] = torch.where(
            current < 0, torch.zeros_like(current), current
        )

        # Révéler les 4 murs de chaque cellule observée
        # _wall_idx[cell_id] → (4,) indices dans wall_memory, -1 si bordure
        widx      = self._wall_idx[cell_ids]        # (..., 4)
        is_border = self._is_border[cell_ids]       # (..., 4)

        # Valeur réelle du mur dans ce labyrinthe
        # walls_flat[env, wall_idx] — on doit gather par env_id et wall_idx
        # env_ids peut avoir des répétitions, on itère sur les 4 directions
        for d in range(4):
            border_mask = is_border[:, d]           # True si bordure → ignorer
            valid       = ~border_mask
            if not valid.any():
                continue
            e_v  = env_ids[valid]
            wi_v = widx[valid, d]
            val  = self.walls_flat[e_v, wi_v]       # valeur réelle (0 ou 1)
            self.wall_memory[e_v, wi_v] = val

    def _obs(self) -> torch.Tensor:
        # Position : row et col séparés, normalisés indépendamment
        row_norm = self.pos[:, 0].float() / (self.rows - 1)   # (n,)
        col_norm = self.pos[:, 1].float() / (self.cols - 1)   # (n,)
        pos      = torch.stack([row_norm, col_norm], dim=1)   # (n, 2)

        dy   = (self.end_pos[:, 0].float() - self.pos[:, 0].float()) / (self.rows - 1)
        dx   = (self.end_pos[:, 1].float() - self.pos[:, 1].float()) / (self.cols - 1)
        goal = torch.stack([dy, dx], dim=1)                   # (n, 2)

        return torch.cat([self.cell_memory, self.wall_memory, pos, goal], dim=1)
