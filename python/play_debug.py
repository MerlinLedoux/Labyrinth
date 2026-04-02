"""
Interface graphique de debug — jouer manuellement + voir obs/reward dans le terminal.
Lancer : python -X utf8 play_debug.py
Controles : fleches directionnelles, R = nouveau labyrinthe, Q = quitter
"""
import sys
import tkinter as tk
import numpy as np
from maze_env import MazeEnv, OBS_DIM, MEM_SIZE, MAZE_SIZE, ACTION_DIRS
from maze import NORTH, SOUTH, EAST, WEST

# Config affichage
CELL   = 40          # pixels par cellule
MARGIN = 20          # marge autour du labyrinthe
WALL_W = 3           # epaisseur des murs

COL_BG       = "#1a1a1a"
COL_UNSEEN   = "#2a2a2a"
COL_SEEN     = "#4a4a4a"
COL_VISIBLE  = "#888888"
COL_PLAYER   = "#c63737"
COL_GOAL     = "#ef643a"
COL_WALL     = "#000000"
COL_WALL_MEM = "#222222"

ACTION_KEY = {
    "Up":    0,   # NORTH
    "Down":  1,   # SOUTH
    "Right": 2,   # EAST
    "Left":  3,   # WEST
}
ACTION_NAME = ["NORTH", "SOUTH", "EAST", "WEST"]
VIS_RANGE = 3


def compute_visible(m, row, col):
    """Cases visibles par ray-casting cardinal."""
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


class MazeApp:
    def __init__(self, root: tk.Tk):
        self.root  = root
        self.env   = MazeEnv()
        self.obs   = None
        self.step  = 0
        self.total_reward = 0.0
        self.revealed = set()

        root.title("Maze Debug")
        root.configure(bg=COL_BG)
        root.resizable(False, False)

        size = MAZE_SIZE * CELL + 2 * MARGIN

        # Canvas du labyrinthe
        self.canvas = tk.Canvas(root, width=size, height=size,
                                bg=COL_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)

        # Panneau info a droite
        info_frame = tk.Frame(root, bg=COL_BG, padx=12, pady=12)
        info_frame.pack(side=tk.LEFT, fill=tk.Y)

        lbl = lambda text, bold=False: tk.Label(
            info_frame, text=text, bg=COL_BG, fg="#cccccc",
            font=("Consolas", 10, "bold" if bold else "normal"), anchor="w"
        )

        lbl("INFOS PARTIE", bold=True).pack(anchor="w")
        self.lbl_step   = lbl("Pas : 0")
        self.lbl_step.pack(anchor="w")
        self.lbl_reward = lbl("Reward total : 0.0")
        self.lbl_reward.pack(anchor="w")
        self.lbl_status = lbl("")
        self.lbl_status.pack(anchor="w")

        tk.Label(info_frame, bg=COL_BG).pack()

        lbl("OBSERVATION", bold=True).pack(anchor="w")
        self.lbl_pos    = lbl("Position : ")
        self.lbl_pos.pack(anchor="w")
        self.lbl_imm    = lbl("Vision   : ")
        self.lbl_imm.pack(anchor="w")
        self.lbl_goal   = lbl("Objectif : ")
        self.lbl_goal.pack(anchor="w")
        self.lbl_mem    = lbl("Memoire  : ")
        self.lbl_mem.pack(anchor="w")

        tk.Label(info_frame, bg=COL_BG).pack()

        lbl("DERNIER PAS", bold=True).pack(anchor="w")
        self.lbl_action = lbl("Action  : -")
        self.lbl_action.pack(anchor="w")
        self.lbl_rew    = lbl("Reward  : -")
        self.lbl_rew.pack(anchor="w")
        self.lbl_wall   = lbl("Mur     : -")
        self.lbl_wall.pack(anchor="w")

        tk.Label(info_frame, bg=COL_BG).pack()

        tk.Label(info_frame, text="Fleches = deplacer\nR = nouveau labyrinthe\nQ = quitter",
                 bg=COL_BG, fg="#666666", font=("Consolas", 9), justify=tk.LEFT).pack(anchor="w")

        root.bind("<KeyPress>", self.on_key)

        self.new_game()

    # Nouvelle partie
    def new_game(self):
        self.obs, _ = self.env.reset()
        self.step         = 0
        self.total_reward = 0.0
        self.revealed     = set()
        self._update_revealed()
        self.lbl_status.config(text="", fg="#cccccc")
        self._refresh_info(action=None, reward=None, hit_wall=None)
        self.draw()
        print("\n" + "="*60)
        print("NOUVEAU LABYRINTHE")
        print("="*60)
        self._print_obs(self.obs)

    # Clavier
    def on_key(self, event):
        if event.keysym == "r" or event.keysym == "R":
            self.new_game()
            return
        if event.keysym in ("q", "Q", "Escape"):
            self.root.destroy()
            return
        if event.keysym not in ACTION_KEY:
            return

        action   = ACTION_KEY[event.keysym]
        obs, reward, terminated, truncated, _ = self.env.step(action)
        hit_wall = (reward == -1.0)

        self.obs           = obs
        self.step         += 1
        self.total_reward += reward
        self._update_revealed()

        # Terminal
        print(f"\n--- Pas {self.step} | Action : {ACTION_NAME[action]} ---")
        print(f"    Reward     : {reward:+.2f}  ({'mur' if hit_wall else 'deplacement'})")
        print(f"    Reward tot : {self.total_reward:+.2f}")
        self._print_obs(obs)

        # Labels GUI
        self._refresh_info(action, reward, hit_wall)
        self.lbl_step.config(text=f"Pas : {self.step}")
        self.lbl_reward.config(text=f"Reward total : {self.total_reward:+.2f}")

        if terminated:
            self.lbl_status.config(text=f"VICTOIRE en {self.step} pas !", fg="#9FCF65")
            print(f"\n{'='*60}\nVICTOIRE en {self.step} pas !\n{'='*60}")
        elif truncated:
            self.lbl_status.config(text="Timeout", fg="#c63737")
            print(f"\n{'='*60}\nTIMEOUT\n{'='*60}")

        self.draw()

    # Memoire visuelle
    def _update_revealed(self):
        m   = self.env.maze
        r,c = self.env.player_row, self.env.player_col
        for cell in compute_visible(m, r, c):
            self.revealed.add(cell)

    # Dessin
    def draw(self):
        self.canvas.delete("all")
        m   = self.env.maze
        pr  = self.env.player_row
        pc  = self.env.player_col
        vis = compute_visible(m, pr, pc)

        for row in range(MAZE_SIZE):
            for col in range(MAZE_SIZE):
                x = MARGIN + col * CELL
                y = MARGIN + row * CELL
                cell = (row, col)

                is_vis = cell in vis
                is_mem = (not is_vis) and (cell in self.revealed)

                if is_vis:
                    color = COL_VISIBLE
                elif is_mem:
                    color = COL_SEEN
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
        self.canvas.create_text(gx+CELL//2, gy+CELL//2, text="*",
                                fill="white", font=("Arial", 14, "bold"))

        # Joueur
        px = MARGIN + pc * CELL + CELL//2
        py = MARGIN + pr * CELL + CELL//2
        r  = CELL * 0.32
        self.canvas.create_oval(px-r, py-r, px+r, py+r,
                                fill=COL_PLAYER, outline="")

    # Affichage terminal
    def _print_obs(self, obs: np.ndarray):
        mem       = obs[:MEM_SIZE]
        player    = obs[MEM_SIZE]
        immediate = obs[MEM_SIZE+1 : MEM_SIZE+5]
        goal      = obs[MEM_SIZE+5 : MEM_SIZE+7]

        revealed  = (mem >= 0).sum()
        idx       = round(player * (MEM_SIZE - 1))
        row, col  = divmod(idx, MAZE_SIZE)

        imm_str = f"N={'mur' if immediate[0] else 'ok '}  E={'mur' if immediate[1] else 'ok '}  " \
                  f"S={'mur' if immediate[2] else 'ok '}  O={'mur' if immediate[3] else 'ok '}"

        print(f"    Position   : ({row},{col})  idx={idx}  norm={player:.4f}")
        print(f"    Vision imm : {imm_str}")
        print(f"    Delta obj  : dy={goal[0]:+.3f}  dx={goal[1]:+.3f}")
        print(f"    Memoire    : {revealed}/{MEM_SIZE} cellules connues")

    def _refresh_info(self, action, reward, hit_wall):
        obs = self.obs
        if obs is None:
            return

        mem       = obs[:MEM_SIZE]
        player    = obs[MEM_SIZE]
        immediate = obs[MEM_SIZE+1 : MEM_SIZE+5]
        goal      = obs[MEM_SIZE+5 : MEM_SIZE+7]
        idx       = round(player * (MEM_SIZE - 1))
        row, col  = divmod(idx, MAZE_SIZE)
        revealed  = (mem >= 0).sum()

        self.lbl_pos.config( text=f"Position : ({row},{col})  idx={idx}")
        self.lbl_imm.config( text=f"Vision   : N={'#' if immediate[0] else '.'}  "
                                   f"E={'#' if immediate[1] else '.'}  "
                                   f"S={'#' if immediate[2] else '.'}  "
                                   f"O={'#' if immediate[3] else '.'}")
        self.lbl_goal.config(text=f"Objectif : dy={goal[0]:+.2f}  dx={goal[1]:+.2f}")
        self.lbl_mem.config( text=f"Memoire  : {revealed}/{MEM_SIZE} cellules")

        if action is not None:
            self.lbl_action.config(text=f"Action  : {ACTION_NAME[action]}")
            self.lbl_rew.config(   text=f"Reward  : {reward:+.2f}")
            self.lbl_wall.config(  text=f"Mur     : {'OUI' if hit_wall else 'non'}",
                                   fg="#c63737" if hit_wall else "#9FCF65")


if __name__ == "__main__":
    root = tk.Tk()
    app  = MazeApp(root)
    root.mainloop()
