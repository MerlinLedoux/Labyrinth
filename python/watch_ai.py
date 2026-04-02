"""
Visualisation du modèle PPO entraîné.
Lancer : python -X utf8 watch_ai.py [chemin_modele]
         python -X utf8 watch_ai.py          # modèle par défaut : models/maze_ppo.pt

Contrôles :
  ESPACE       — pause / reprise
  R            — nouveau labyrinthe
  +/-          — vitesse (ms entre chaque pas)
  Q / Echap    — quitter
"""
import sys
import tkinter as tk
import numpy as np
import torch
from train import ActorCritic
from maze_env import MazeEnv, OBS_DIM, MEM_SIZE, MAZE_SIZE
from maze import NORTH, SOUTH, EAST, WEST

MODEL_PATH  = sys.argv[1] if len(sys.argv) > 1 else "models/maze_ppo.pt"
ACTION_NAME = ["NORTH", "SOUTH", "EAST", "WEST"]
VIS_RANGE   = 3
SPEED_DEFAULT = 300   # ms entre chaque pas

# ── Couleurs ───────────────────────────────────────────────────────────────────
COL_BG       = "#1a1a1a"
COL_UNSEEN   = "#2a2a2a"
COL_SEEN     = "#4a4a4a"
COL_VISIBLE  = "#888888"
COL_PLAYER   = "#c63737"
COL_GOAL     = "#ef643a"
COL_WALL     = "#000000"
COL_WALL_MEM = "#222222"
COL_TRAIL    = "#5a3a3a"   # cases déjà piétinées par l'agent
CELL         = 60
MARGIN       = 20
WALL_W       = 3


def compute_visible(m, row, col):
    vis = set()
    vis.add((row, col))
    for d, dr, dc in [(NORTH,-1,0),(SOUTH,1,0),(EAST,0,1),(WEST,0,-1)]:
        cr, cc = row, col
        for _ in range(VIS_RANGE):
            if m.has_wall(cr, cc, d):
                break
            cr += dr; cc += dc
            vis.add((cr, cc))
    return vis


class WatchApp:
    def __init__(self, root: tk.Tk, model: ActorCritic):
        self.root  = root
        self.model = model
        self.env   = MazeEnv()
        self.speed = SPEED_DEFAULT
        self.paused = False
        self.after_id = None

        root.title("Watch AI")
        root.configure(bg=COL_BG)
        root.resizable(False, False)

        size = MAZE_SIZE * CELL + 2 * MARGIN

        self.canvas = tk.Canvas(root, width=size, height=size,
                                bg=COL_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)

        # ── Panneau droite ─────────────────────────────────────────────────────
        panel = tk.Frame(root, bg=COL_BG, padx=12, pady=12)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        def lbl(text, bold=False, color="#cccccc"):
            return tk.Label(panel, text=text, bg=COL_BG, fg=color,
                            font=("Consolas", 10, "bold" if bold else "normal"),
                            anchor="w")

        lbl("MODELE", bold=True).pack(anchor="w")
        self.lbl_model = lbl(MODEL_PATH, color="#888888")
        self.lbl_model.pack(anchor="w")

        tk.Label(panel, bg=COL_BG).pack()

        lbl("EPISODE", bold=True).pack(anchor="w")
        self.lbl_ep     = lbl("Episode  : 1")
        self.lbl_step   = lbl("Pas      : 0")
        self.lbl_reward = lbl("Reward   : 0.0")
        self.lbl_result = lbl("")
        for w in (self.lbl_ep, self.lbl_step, self.lbl_reward, self.lbl_result):
            w.pack(anchor="w")

        tk.Label(panel, bg=COL_BG).pack()

        lbl("DERNIER PAS", bold=True).pack(anchor="w")
        self.lbl_action = lbl("Action   : —")
        self.lbl_rew    = lbl("Reward   : —")
        self.lbl_wall   = lbl("Mur      : —")
        for w in (self.lbl_action, self.lbl_rew, self.lbl_wall):
            w.pack(anchor="w")

        tk.Label(panel, bg=COL_BG).pack()

        lbl("OBSERVATION", bold=True).pack(anchor="w")
        self.lbl_pos  = lbl("Position : —")
        self.lbl_imm  = lbl("Vision   : —")
        self.lbl_goal = lbl("Objectif : —")
        self.lbl_mem  = lbl("Memoire  : —")
        for w in (self.lbl_pos, self.lbl_imm, self.lbl_goal, self.lbl_mem):
            w.pack(anchor="w")

        tk.Label(panel, bg=COL_BG).pack()

        lbl("STATS", bold=True).pack(anchor="w")
        self.lbl_wins  = lbl("Victoires : 0")
        self.lbl_rate  = lbl("Win rate  : —")
        self.lbl_wins.pack(anchor="w")
        self.lbl_rate.pack(anchor="w")

        tk.Label(panel, bg=COL_BG).pack()

        self.lbl_speed = lbl(f"Vitesse  : {self.speed} ms")
        self.lbl_speed.pack(anchor="w")
        self.lbl_pause = lbl("PAUSE" if self.paused else "EN COURS", color="#9FCF65")
        self.lbl_pause.pack(anchor="w")

        tk.Label(panel, bg=COL_BG).pack()
        lbl("ESPACE=pause  R=reset\n+=rapide  -=lent  Q=quitter",
            color="#555555").pack(anchor="w")

        root.bind("<KeyPress>", self.on_key)

        # Stats globales
        self.episode_count = 0
        self.win_count     = 0
        self.revealed      = set()
        self.trail         = {}   # (row,col) -> nombre de passages
        self.obs           = None

        self.new_game()

    # ── Nouvelle partie ────────────────────────────────────────────────────────
    def new_game(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.obs, _       = self.env.reset()
        self.step_count   = 0
        self.total_reward = 0.0
        self.revealed     = set()
        self.trail        = {}
        self.episode_count += 1
        self.paused        = False

        self.lbl_ep.config(text=f"Episode  : {self.episode_count}")
        self.lbl_step.config(text="Pas      : 0")
        self.lbl_reward.config(text="Reward   : 0.0")
        self.lbl_result.config(text="")
        self.lbl_pause.config(text="EN COURS", fg="#9FCF65")

        self._update_revealed()
        self.draw()

        if not self.paused:
            self.after_id = self.root.after(self.speed, self.ai_step)

    # ── Pas IA ─────────────────────────────────────────────────────────────────
    def ai_step(self):
        if self.paused:
            return

        obs_t = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(obs_t)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            probs  = torch.softmax(logits, dim=1).squeeze().tolist()
        print(f"  Probas  N={probs[0]*100:4.1f}%  S={probs[1]*100:4.1f}%  E={probs[2]*100:4.1f}%  O={probs[3]*100:4.1f}%")

        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.obs           = obs
        self.step_count   += 1
        self.total_reward += reward
        hit_wall = (reward == -1.0)

        # Trail
        pos = (self.env.player_row, self.env.player_col)
        self.trail[pos] = self.trail.get(pos, 0) + 1
        self._update_revealed()

        # Labels
        self.lbl_step.config(  text=f"Pas      : {self.step_count}")
        self.lbl_reward.config(text=f"Reward   : {self.total_reward:+.1f}")
        self.lbl_action.config(text=f"Action   : {ACTION_NAME[action]}")
        self.lbl_rew.config(   text=f"Reward   : {reward:+.2f}")
        self.lbl_wall.config(  text=f"Mur      : {'OUI' if hit_wall else 'non'}",
                               fg="#c63737" if hit_wall else "#9FCF65")
        self._refresh_obs()

        self.draw()

        if terminated:
            self.win_count += 1
            rate = self.win_count / self.episode_count * 100
            self.lbl_result.config(text=f"VICTOIRE en {self.step_count} pas!", fg="#9FCF65")
            self.lbl_wins.config(  text=f"Victoires : {self.win_count}/{self.episode_count}")
            self.lbl_rate.config(  text=f"Win rate  : {rate:.0f}%")
            print(f"Episode {self.episode_count:3d} | VICTOIRE en {self.step_count:3d} pas | "
                  f"reward={self.total_reward:+.1f} | win rate={rate:.0f}%")
            self.after_id = self.root.after(1200, self.new_game)

        elif truncated:
            rate = self.win_count / self.episode_count * 100
            self.lbl_result.config(text=f"Timeout ({self.step_count} pas)", fg="#c63737")
            self.lbl_wins.config(  text=f"Victoires : {self.win_count}/{self.episode_count}")
            self.lbl_rate.config(  text=f"Win rate  : {rate:.0f}%")
            print(f"Episode {self.episode_count:3d} | timeout     {self.step_count:3d} pas | "
                  f"reward={self.total_reward:+.1f} | win rate={rate:.0f}%")
            self.after_id = self.root.after(800, self.new_game)

        else:
            self.after_id = self.root.after(self.speed, self.ai_step)

    # ── Clavier ────────────────────────────────────────────────────────────────
    def on_key(self, event):
        k = event.keysym
        if k in ("q", "Q", "Escape"):
            self.root.destroy()
        elif k in ("r", "R"):
            self.new_game()
        elif k == "space":
            self.paused = not self.paused
            self.lbl_pause.config(
                text="PAUSE" if self.paused else "EN COURS",
                fg="#c63737" if self.paused else "#9FCF65"
            )
            if not self.paused:
                self.after_id = self.root.after(self.speed, self.ai_step)
        elif k in ("plus", "equal"):
            self.speed = max(50, self.speed - 50)
            self.lbl_speed.config(text=f"Vitesse  : {self.speed} ms")
        elif k in ("minus", "underscore"):
            self.speed = min(2000, self.speed + 50)
            self.lbl_speed.config(text=f"Vitesse  : {self.speed} ms")

    # ── Mémoire visuelle ───────────────────────────────────────────────────────
    def _update_revealed(self):
        m   = self.env.maze
        r,c = self.env.player_row, self.env.player_col
        for cell in compute_visible(m, r, c):
            self.revealed.add(cell)

    # ── Observation labels ─────────────────────────────────────────────────────
    def _refresh_obs(self):
        obs = self.obs
        mem       = obs[:MEM_SIZE]
        player    = obs[MEM_SIZE]
        immediate = obs[MEM_SIZE+1:MEM_SIZE+5]
        goal      = obs[MEM_SIZE+5:MEM_SIZE+7]
        idx       = round(player * (MEM_SIZE - 1))
        row, col  = divmod(idx, MAZE_SIZE)
        revealed  = (mem >= 0).sum()

        self.lbl_pos.config( text=f"Position : ({row},{col})")
        self.lbl_imm.config( text=f"Vision   : N={'█' if immediate[0] else '·'} "
                                   f"E={'█' if immediate[1] else '·'} "
                                   f"S={'█' if immediate[2] else '·'} "
                                   f"O={'█' if immediate[3] else '·'}")
        self.lbl_goal.config(text=f"Objectif : dy={goal[0]:+.2f} dx={goal[1]:+.2f}")
        self.lbl_mem.config( text=f"Memoire  : {revealed}/{MEM_SIZE} cellules")

    # ── Dessin ─────────────────────────────────────────────────────────────────
    def draw(self):
        self.canvas.delete("all")
        if not self.env.maze:
            return
        m   = self.env.maze
        pr  = self.env.player_row
        pc  = self.env.player_col
        vis = compute_visible(m, pr, pc)

        for row in range(MAZE_SIZE):
            for col in range(MAZE_SIZE):
                x    = MARGIN + col * CELL
                y    = MARGIN + row * CELL
                cell = (row, col)

                is_vis = cell in vis
                is_mem = (not is_vis) and (cell in self.revealed)
                passes = self.trail.get(cell, 0)

                if is_vis:
                    color = COL_VISIBLE
                elif is_mem:
                    # Plus la cellule a été visitée, plus elle est marquée
                    intensity = min(passes, 5)
                    r_val = 0x4a + intensity * 0x10
                    color = f"#{r_val:02x}2a2a" if passes > 0 else COL_SEEN
                else:
                    color = COL_UNSEEN

                self.canvas.create_rectangle(x, y, x+CELL, y+CELL,
                                             fill=color, outline="")

                if is_vis or is_mem:
                    wc = COL_WALL if is_vis else COL_WALL_MEM
                    if m.has_wall(row, col, NORTH):
                        self.canvas.create_line(x, y, x+CELL, y, fill=wc, width=WALL_W)
                    if m.has_wall(row, col, SOUTH):
                        self.canvas.create_line(x, y+CELL, x+CELL, y+CELL, fill=wc, width=WALL_W)
                    if m.has_wall(row, col, WEST):
                        self.canvas.create_line(x, y, x, y+CELL, fill=wc, width=WALL_W)
                    if m.has_wall(row, col, EAST):
                        self.canvas.create_line(x+CELL, y, x+CELL, y+CELL, fill=wc, width=WALL_W)

        # Objectif
        gx = MARGIN + (MAZE_SIZE-1) * CELL
        gy = MARGIN + (MAZE_SIZE-1) * CELL
        self.canvas.create_rectangle(gx+4, gy+4, gx+CELL-4, gy+CELL-4,
                                     fill=COL_GOAL, outline="")
        self.canvas.create_text(gx+CELL//2, gy+CELL//2, text="★",
                                fill="white", font=("Arial", 16, "bold"))

        # Joueur
        px = MARGIN + pc * CELL + CELL//2
        py = MARGIN + pr * CELL + CELL//2
        r  = CELL * 0.32
        self.canvas.create_oval(px-r, py-r, px+r, py+r,
                                fill=COL_PLAYER, outline="")


# ── Chargement du modèle ───────────────────────────────────────────────────────
def load_model(path: str) -> ActorCritic:
    model = ActorCritic()
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"Modele charge : {path}")
    except FileNotFoundError:
        print(f"ERREUR : modele introuvable ({path})")
        print("Lance d'abord : python train.py")
        sys.exit(1)
    return model


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    root  = tk.Tk()
    app   = WatchApp(root, model)
    root.mainloop()
