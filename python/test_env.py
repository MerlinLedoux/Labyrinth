"""
Diagnostic de l'environnement MazeEnv.
Lance : python test_env.py
"""
import numpy as np
from maze_env import MazeEnv, OBS_DIM, MEM_SIZE, ACTION_DIRS, MAZE_SIZE
from maze import NORTH, SOUTH, EAST, WEST

ACTION_NAMES = {NORTH: "NORTH", SOUTH: "SOUTH", EAST: "EAST", WEST: "WEST"}
ACTION_IDX   = ["NORTH", "SOUTH", "EAST", "WEST"]

def print_obs(obs: np.ndarray) -> None:
    mem       = obs[:MEM_SIZE]
    player    = obs[MEM_SIZE]
    immediate = obs[MEM_SIZE+1 : MEM_SIZE+5]
    goal      = obs[MEM_SIZE+5 : MEM_SIZE+7]

    revealed = (mem >= 0).sum()
    print(f"  Mémoire       : {revealed}/{MEM_SIZE} cellules observées "
          f"| min={mem[mem>=0].min():.2f} max={mem[mem>=0].max():.2f}" if revealed > 0
          else f"  Mémoire       : {revealed}/{MEM_SIZE} cellules observées")
    print(f"  Position      : {player:.4f}  (idx={round(player*(MEM_SIZE-1))})")
    print(f"  Vision [N,E,S,W]: {immediate.astype(int).tolist()}")
    print(f"  Delta objectif: dy={goal[0]:+.3f}  dx={goal[1]:+.3f}")


def check_obs_range(obs: np.ndarray, step: int) -> None:
    """Vérifie que toutes les valeurs sont dans [-1, 1]."""
    out = np.where((obs < -1.0) | (obs > 1.0))[0]
    if len(out):
        print(f"  ⚠️  VALEURS HORS BORNES au step {step} — indices : {out.tolist()}")
        print(f"      valeurs : {obs[out].tolist()}")


# ── Test 1 : reset ─────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1 — reset()")
env = MazeEnv()
obs, info = env.reset()
print(f"  Shape obs : {obs.shape}  (attendu : ({OBS_DIM},))")
assert obs.shape == (OBS_DIM,), "ERREUR : mauvaise shape"
check_obs_range(obs, step=0)
print_obs(obs)
print("  ✓ reset OK\n")


# ── Test 2 : actions valides ────────────────────────────────────────────────────
print("=" * 60)
print("TEST 2 — 4 actions depuis la position de départ")
obs, _ = env.reset()
for action_idx, name in enumerate(ACTION_IDX):
    env2 = MazeEnv()
    obs2, _ = env2.reset()
    # même seed → même labyrinthe impossible sans seed, on teste juste la mécanique
    next_obs, reward, terminated, truncated, _ = env2.step(action_idx)
    check_obs_range(next_obs, step=1)
    moved = not np.array_equal(obs2[OBS_DIM-7], next_obs[OBS_DIM-7])  # position changée ?
    wall  = obs2[MEM_SIZE+1 + action_idx] == 1.0                      # mur dans cette direction ?
    print(f"  Action {action_idx} ({name:5s}) : reward={reward:+.2f}  "
          f"mur={'oui' if wall else 'non'}  déplacement={'non' if wall else 'oui'}  "
          f"terminé={terminated}")
print("  ✓ actions OK\n")


# ── Test 3 : épisode complet avec politique aléatoire ─────────────────────────
print("=" * 60)
print("TEST 3 — épisode aléatoire complet")
obs, _ = env.reset()
total_reward = 0.0
steps = 0
wall_hits = 0

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    check_obs_range(obs, steps)
    total_reward += reward
    steps += 1
    if reward == -0.5:
        wall_hits += 1
    if terminated or truncated:
        break

revealed = (obs[:MEM_SIZE] >= 0).sum()
print(f"  Résultat      : {'VICTOIRE ✓' if terminated else 'timeout'}")
print(f"  Pas           : {steps}")
print(f"  Hits mur      : {wall_hits}")
print(f"  Reward total  : {total_reward:.1f}")
print(f"  Cellules vues : {revealed}/{MEM_SIZE}")
print("  ✓ épisode OK\n")


# ── Test 4 : cohérence mémoire ──────────────────────────────────────────────────
print("=" * 60)
print("TEST 4 — cohérence mémoire")
obs, _ = env.reset()
mem = obs[:MEM_SIZE]
# Au départ seules les cellules visibles depuis (0,0) sont révélées
revealed_start = (mem >= 0).sum()
print(f"  Cellules visibles au départ : {revealed_start}")
assert revealed_start >= 1, "ERREUR : au moins la case de départ doit être révélée"

# Après quelques pas, on doit en avoir plus
for _ in range(10):
    env.step(env.action_space.sample())
obs_after = env._obs()
revealed_after = (obs_after[:MEM_SIZE] >= 0).sum()
print(f"  Cellules visibles après 10 pas : {revealed_after}")
assert revealed_after >= revealed_start, "ERREUR : la mémoire ne doit pas diminuer"

# Vérifier que les valeurs révélées sont dans [0, 1]
revealed_vals = mem[mem >= 0]
print(f"  Valeurs mémoire révélées : min={revealed_vals.min():.3f}  max={revealed_vals.max():.3f}")
assert (revealed_vals >= 0).all() and (revealed_vals <= 1).all(), \
    "ERREUR : valeurs mémoire hors [0,1]"
print("  ✓ mémoire OK\n")


# ── Test 5 : aperçu visuel du labyrinthe ───────────────────────────────────────
print("=" * 60)
print("TEST 5 — aperçu visuel du labyrinthe 15×15")
env2 = MazeEnv()
env2.reset()
m = env2.maze
print()
for r in range(MAZE_SIZE):
    # Ligne du haut
    top = ""
    for c in range(MAZE_SIZE):
        top += "+" + ("---" if m.has_wall(r, c, NORTH) else "   ")
    print(top + "+")
    # Ligne du milieu
    mid = ""
    for c in range(MAZE_SIZE):
        left  = "|" if m.has_wall(r, c, WEST) else " "
        here  = " P " if (r == env2.player_row and c == env2.player_col) else \
                " X " if (r == MAZE_SIZE-1 and c == MAZE_SIZE-1) else "   "
        mid  += left + here
    print(mid + "|")
# Dernière ligne du bas
bot = ""
for c in range(MAZE_SIZE):
    bot += "+" + ("---" if m.has_wall(MAZE_SIZE-1, c, SOUTH) else "   ")
print(bot + "+")
print(f"\n  P = joueur (départ)   X = objectif")

print("\n" + "=" * 60)
print("Tous les tests ont passé ✓")
