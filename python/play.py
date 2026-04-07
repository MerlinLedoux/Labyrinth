"""
play.py — Interface graphique tkinter pour jouer manuellement dans le labyrinthe.

Contrôles :
  Flèches directionnelles : déplacer le joueur
  R                        : nouveau labyrinthe
  Q / Echap                : quitter

Légende :
  Gris foncé  — cellule jamais observée
  Gris clair  — observée (FOV), jamais visitée
  Blanc       — visitée physiquement
  Bleu        — joueur
  Vert        — objectif
  Rouge       — murs visibles
"""

import tkinter as tk
import torch
from maze_env import VecMazeEnv, ROWS, COLS, MAX_STEPS

# ── Paramètres visuels ────────────────────────────────────────────────────────
CELL      = 52        # pixels par cellule
WALL_W    = 3         # épaisseur des murs
MARGIN    = 30        # marge autour de la grille
INFO_W    = 220       # largeur du panneau d'infos

GRID_W = COLS * CELL
GRID_H = ROWS * CELL
WIN_W  = GRID_W + 2 * MARGIN + INFO_W
WIN_H  = GRID_H + 2 * MARGIN

C_BG      = "#1e1e1e"
C_UNSEEN  = "#373737"
C_SEEN    = "#a0a0a0"
C_VISITED = "#e6e6e6"
C_PLAYER  = "#4287f5"
C_GOAL    = "#32c850"
C_WALL    = "#c83232"
C_OUTLINE = "#505050"
C_TEXT    = "#dcdcdc"
C_DIM     = "#828282"

KEY_ACTION = {
    "Up": 0, "Down": 1, "Right": 2, "Left": 3,
    "w":  0, "s":    1, "d":     2, "a":   3,
}


class MazeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Labyrinthe — Mode Joueur")
        root.configure(bg=C_BG)
        root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=WIN_W, height=WIN_H,
                                bg=C_BG, highlightthickness=0)
        self.canvas.pack()

        self.env          = VecMazeEnv(n_envs=1, device="cpu")
        self.step_count   = 0
        self.total_reward = 0.0
        self.wins         = 0
        self.last_msg     = ""

        root.bind("<KeyPress>", self._on_key)
        self._draw()

    # ── Gestion des touches ───────────────────────────────────────────────────

    def _on_key(self, event):
        key = event.keysym

        if key in ("q", "Escape"):
            self.root.destroy()
            return

        if key in ("r", "R"):
            self.env.reset()
            self.step_count   = 0
            self.total_reward = 0.0
            self.last_msg     = "Nouveau labyrinthe"
            self._draw()
            return

        if key not in KEY_ACTION:
            return

        action  = KEY_ACTION[key]
        actions = torch.tensor([action], dtype=torch.long)
        _, reward, terminated, truncated = self.env.step(actions)

        r_val          = float(reward[0].item())
        self.total_reward += r_val
        self.step_count   += 1

        dirs = {0: "Nord", 1: "Sud", 2: "Est", 3: "Ouest"}
        self.last_msg = f"{dirs[action]}  {r_val:+.1f}"

        if bool(terminated[0].item()):
            self.wins += 1
            self.last_msg = f"Objectif atteint !  ({self.wins} victoire(s))"
            self.env.reset()
            self.step_count   = 0
            self.total_reward = 0.0
        elif bool(truncated[0].item()):
            self.last_msg = "Timeout — nouveau labyrinthe"
            self.env.reset()
            self.step_count   = 0
            self.total_reward = 0.0

        self._draw()

    # ── Rendu ─────────────────────────────────────────────────────────────────

    def _draw(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_info()

    def _draw_grid(self):
        env      = self.env
        pr       = int(env.pos[0, 0].item())
        pc       = int(env.pos[0, 1].item())
        gr       = int(env.end_pos[0, 0].item())
        gc       = int(env.end_pos[0, 1].item())
        cell_mem = env.cell_mem[0].cpu()
        walls    = env.walls[0].cpu()

        for r in range(ROWS):
            for c in range(COLS):
                x  = MARGIN + c * CELL
                y  = MARGIN + r * CELL
                x2 = x + CELL
                y2 = y + CELL

                idx   = r * COLS + c
                state = float(cell_mem[idx].item())

                # Fond de cellule
                if r == pr and c == pc:
                    fill = C_PLAYER
                elif r == gr and c == gc:
                    fill = C_GOAL
                elif state < 0:
                    fill = C_UNSEEN
                elif state < 1:
                    fill = C_SEEN
                else:
                    fill = C_VISITED

                self.canvas.create_rectangle(x, y, x2, y2,
                                             fill=fill, outline=C_OUTLINE, width=1)

                # Murs (uniquement si cellule observée ou position joueur)
                if state >= 0 or (r == pr and c == pc):
                    w = walls[r, c]
                    hw = WALL_W
                    if bool(w[0].item()):   # Nord
                        self.canvas.create_line(x, y, x2, y, fill=C_WALL, width=hw)
                    if bool(w[1].item()):   # Sud
                        self.canvas.create_line(x, y2, x2, y2, fill=C_WALL, width=hw)
                    if bool(w[2].item()):   # Est
                        self.canvas.create_line(x2, y, x2, y2, fill=C_WALL, width=hw)
                    if bool(w[3].item()):   # Ouest
                        self.canvas.create_line(x, y, x, y2, fill=C_WALL, width=hw)

        # Bordure extérieure
        bx, by = MARGIN, MARGIN
        bx2 = bx + GRID_W
        by2 = by + GRID_H
        self.canvas.create_rectangle(bx, by, bx2, by2,
                                     outline=C_WALL, width=WALL_W + 1)

    def _draw_info(self):
        x  = GRID_W + 2 * MARGIN + 10
        y  = MARGIN
        lh = 22   # hauteur de ligne

        def txt(label, value="", color=C_TEXT, dim=False):
            nonlocal y
            c = C_DIM if dim else color
            self.canvas.create_text(x, y, anchor="nw", fill=c,
                                    font=("Consolas", 11),
                                    text=f"{label}{value}")
            y += lh

        pr = int(self.env.pos[0, 0].item())
        pc = int(self.env.pos[0, 1].item())
        gr = int(self.env.end_pos[0, 0].item())
        gc = int(self.env.end_pos[0, 1].item())

        txt("LABYRINTHE", color="#ffffff")
        y += 6
        txt(f"Position   ({pr}, {pc})")
        txt(f"Objectif   ({gr}, {gc})")
        y += 6
        txt(f"Pas        {self.step_count} / {MAX_STEPS}")
        txt(f"Reward     {self.total_reward:+.1f}")
        txt(f"Victoires  {self.wins}")
        y += 6
        txt(self.last_msg, color="#f0c040")
        y += 12

        txt("Légende", dim=True)
        legends = [
            (C_UNSEEN,  "Jamais vue"),
            (C_SEEN,    "Vue (FOV)"),
            (C_VISITED, "Visitée"),
            (C_PLAYER,  "Joueur"),
            (C_GOAL,    "Objectif"),
        ]
        for color, label in legends:
            self.canvas.create_rectangle(x, y + 2, x + 14, y + 14,
                                         fill=color, outline="")
            self.canvas.create_text(x + 20, y, anchor="nw", fill=C_DIM,
                                    font=("Consolas", 11), text=label)
            y += lh

        y += 10
        for line in ("Touches :", "↑↓←→ / WASD : bouger",
                     "R : reset", "Q : quitter"):
            txt(line, dim=True)


if __name__ == "__main__":
    root = tk.Tk()
    MazeApp(root)
    root.mainloop()
