"""
watch.py — Visualisation des décisions de l'IA dans le labyrinthe.

Affiche :
  - Le labyrinthe avec la mémoire de l'agent (même légende que play.py)
  - Les probabilités des 4 actions sous forme de barres colorées
  - La valeur estimée par le critique
  - La vitesse de simulation (délai ajustable)

Contrôles :
  R         : nouveau labyrinthe
  Espace    : pause / reprendre
  + / -     : accélérer / ralentir
  Q / Echap : quitter

Utilisation :
  python watch.py models/maze_ppo.pt
"""

import sys
import tkinter as tk
import torch
import torch.nn as nn
from torch.distributions import Categorical
from maze_env import VecMazeEnv, ROWS, COLS, MAX_STEPS, CELL_SIZE, WALL_SIZE, OBS_DIM

# ── Paramètres visuels ────────────────────────────────────────────────────────
CELL   = 52
WALL_W = 3
MARGIN = 30
INFO_W = 260

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

# Couleurs des 4 actions
C_ACTIONS = ["#5b9bd5", "#e06c75", "#98c379", "#e5c07b"]  # N S E W
DIR_NAMES = ["N", "S", "E", "O"]

SPEEDS    = [600, 300, 150, 60, 20, 5]   # délais en ms (index 0 = lent)
SPEED_IDX = 2                             # vitesse par défaut


# ── Chargement du modèle (copie de l'architecture de train.py) ────────────────
SPATIAL_DIM = CELL_SIZE + WALL_SIZE

class MemoryNavigator(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Linear(SPATIAL_DIM, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.nav    = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.actor  = nn.Sequential(nn.Linear(288, 64), nn.ReLU(), nn.Linear(64, 4))
        self.critic = nn.Sequential(nn.Linear(288, 64), nn.ReLU(), nn.Linear(64, 1))

    def _fuse(self, obs):
        return torch.cat([
            self.spatial(obs[..., :SPATIAL_DIM]),
            self.nav(obs[..., SPATIAL_DIM:]),
        ], dim=-1)

    def forward(self, obs):
        f = self._fuse(obs)
        return self.actor(f), self.critic(f).squeeze(-1)


# ── Application ───────────────────────────────────────────────────────────────
class WatchApp:
    def __init__(self, root: tk.Tk, model_path: str):
        self.root = root
        root.title("Labyrinthe — Visualisation IA")
        root.configure(bg=C_BG)
        root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=WIN_W, height=WIN_H,
                                bg=C_BG, highlightthickness=0)
        self.canvas.pack()

        # Modèle
        self.model = MemoryNavigator()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu",
                                              weights_only=True))
        self.model.eval()

        # Env
        self.env          = VecMazeEnv(n_envs=1, device="cpu")
        self.obs          = self.env.reset()
        self.step_count   = 0
        self.total_reward = 0.0
        self.wins         = 0
        self.paused       = False
        self.speed_idx    = SPEED_IDX
        self.last_probs   = [0.25] * 4
        self.last_value   = 0.0
        self.last_action  = -1
        self.last_msg     = "Démarrage..."
        self._job         = None

        root.bind("<KeyPress>", self._on_key)
        self._schedule_step()

    # ── Contrôles ─────────────────────────────────────────────────────────────

    def _on_key(self, event):
        key = event.keysym
        if key in ("q", "Escape"):
            self.root.destroy()
        elif key in ("r", "R"):
            self._new_episode("Reset manuel")
        elif key == "space":
            self.paused = not self.paused
            if not self.paused:
                self._schedule_step()
        elif key in ("plus", "equal"):
            self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1)
        elif key in ("minus",):
            self.speed_idx = max(self.speed_idx - 1, 0)

    # ── Boucle de simulation ──────────────────────────────────────────────────

    def _schedule_step(self):
        if self._job is not None:
            self.root.after_cancel(self._job)
        if not self.paused:
            self._job = self.root.after(SPEEDS[self.speed_idx], self._step)

    def _step(self):
        with torch.no_grad():
            logits, value = self.model(self.obs)
            probs         = torch.softmax(logits, dim=-1)[0]
            action        = Categorical(logits=logits).sample()

        self.last_probs  = [float(probs[i].item()) for i in range(4)]
        self.last_value  = float(value[0].item())
        self.last_action = int(action[0].item())

        self.obs, reward, terminated, truncated = self.env.step(action)
        r_val              = float(reward[0].item())
        self.total_reward += r_val
        self.step_count   += 1

        dirs = {0: "Nord", 1: "Sud", 2: "Est", 3: "Ouest"}
        self.last_msg = f"{dirs[self.last_action]}  reward {r_val:+.2f}"

        if bool(terminated[0].item()):
            self.wins += 1
            self.last_msg = f"Objectif atteint ! ({self.wins} victoire(s))"
            self._draw()
            self.root.after(800, lambda: self._new_episode())
            return
        elif bool(truncated[0].item()):
            self.last_msg = "Timeout"
            self._draw()
            self.root.after(400, lambda: self._new_episode())
            return

        self._draw()
        self._schedule_step()

    def _new_episode(self, msg: str = ""):
        self.obs          = self.env.reset()
        self.step_count   = 0
        self.total_reward = 0.0
        self.last_probs   = [0.25] * 4
        self.last_value   = 0.0
        self.last_action  = -1
        self.last_msg     = msg or "Nouveau labyrinthe"
        self._draw()
        self._schedule_step()

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
        cell_mem = env.cell_mem[0]
        walls    = env.walls[0]

        for r in range(ROWS):
            for c in range(COLS):
                x  = MARGIN + c * CELL
                y  = MARGIN + r * CELL
                x2, y2 = x + CELL, y + CELL
                idx   = r * COLS + c
                state = float(cell_mem[idx].item())

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

                if state >= 0 or (r == pr and c == pc):
                    w = walls[r, c]
                    if bool(w[0].item()):
                        self.canvas.create_line(x, y, x2, y, fill=C_WALL, width=WALL_W)
                    if bool(w[1].item()):
                        self.canvas.create_line(x, y2, x2, y2, fill=C_WALL, width=WALL_W)
                    if bool(w[2].item()):
                        self.canvas.create_line(x2, y, x2, y2, fill=C_WALL, width=WALL_W)
                    if bool(w[3].item()):
                        self.canvas.create_line(x, y, x, y2, fill=C_WALL, width=WALL_W)

        self.canvas.create_rectangle(
            MARGIN, MARGIN, MARGIN + GRID_W, MARGIN + GRID_H,
            outline=C_WALL, width=WALL_W + 1
        )

    def _draw_info(self):
        x  = GRID_W + 2 * MARGIN + 10
        y  = MARGIN
        lh = 22

        def txt(label, color=C_TEXT, dim=False):
            nonlocal y
            self.canvas.create_text(x, y, anchor="nw",
                                    fill=C_DIM if dim else color,
                                    font=("Consolas", 11), text=label)
            y += lh

        pr = int(self.env.pos[0, 0].item())
        pc = int(self.env.pos[0, 1].item())
        gr = int(self.env.end_pos[0, 0].item())
        gc = int(self.env.end_pos[0, 1].item())

        txt("IA — Memory Navigator", color="#ffffff")
        y += 4
        txt(f"Position   ({pr}, {pc})")
        txt(f"Objectif   ({gr}, {gc})")
        y += 4
        txt(f"Pas        {self.step_count} / {MAX_STEPS}")
        txt(f"Reward     {self.total_reward:+.1f}")
        txt(f"Victoires  {self.wins}")
        y += 4
        txt(self.last_msg, color="#f0c040")
        y += 10

        # ── Probabilités des actions ──────────────────────────────────────────
        txt("Probabilités :", dim=True)
        bar_w  = INFO_W - 30
        bar_h  = 18
        gap    = 6
        for i, (name, prob, color) in enumerate(
                zip(DIR_NAMES, self.last_probs, C_ACTIONS)):
            # Fond de barre
            self.canvas.create_rectangle(x, y, x + bar_w, y + bar_h,
                                         fill="#2a2a2a", outline="")
            # Remplissage proportionnel
            filled = int(bar_w * prob)
            if filled > 0:
                c_fill = color if i != self.last_action else "#ffffff"
                self.canvas.create_rectangle(x, y, x + filled, y + bar_h,
                                             fill=c_fill, outline="")
            # Label
            label = f"{name}  {prob*100:4.1f}%"
            self.canvas.create_text(x + 6, y + bar_h // 2, anchor="w",
                                    fill="#1a1a1a" if filled > 50 else C_TEXT,
                                    font=("Consolas", 10, "bold"), text=label)
            y += bar_h + gap

        y += 6

        # ── Valeur estimée ────────────────────────────────────────────────────
        txt(f"Valeur     {self.last_value:+.2f}")
        y += 6

        # ── Vitesse ───────────────────────────────────────────────────────────
        speed_label = ["Très lent", "Lent", "Normal", "Rapide", "Très rapide", "Max"]
        txt(f"Vitesse    {speed_label[self.speed_idx]}", dim=True)
        y += 4

        # ── Légende ───────────────────────────────────────────────────────────
        txt("Légende :", dim=True)
        for color, label in [
            (C_UNSEEN, "Jamais vue"), (C_SEEN, "Vue (FOV)"),
            (C_VISITED, "Visitée"),   (C_PLAYER, "Joueur"),
            (C_GOAL, "Objectif"),
        ]:
            self.canvas.create_rectangle(x, y + 2, x + 14, y + 14,
                                         fill=color, outline="")
            self.canvas.create_text(x + 20, y, anchor="nw", fill=C_DIM,
                                    font=("Consolas", 11), text=label)
            y += lh

        y += 6
        for line in ("R : reset", "+ / - : vitesse",
                     "Espace : pause", "Q : quitter"):
            txt(line, dim=True)


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python watch.py <chemin_modele.pt>")
        print("Exemple : python watch.py models/maze_ppo.pt")
        sys.exit(1)

    model_path = sys.argv[1]
    root = tk.Tk()
    WatchApp(root, model_path)
    root.mainloop()
