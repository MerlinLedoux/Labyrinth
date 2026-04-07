"""
train.py — PPO "Memory-Augmented Navigator"

Architecture Actor-Critic :
  Branche spatiale : 280 → Dense(512) + ReLU + LayerNorm → Dense(256) + ReLU
  Branche nav      :   4 → Dense(32)  + ReLU
  Fusion           : cat(256, 32) = 288
  Acteur           : Dense(64) + ReLU → Dense(4)
  Critique         : Dense(64) + ReLU → Dense(1)

Tout tourne sur GPU si disponible — aucun transfert CPU↔GPU pendant l'entraînement.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from maze_env import VecMazeEnv, OBS_DIM, CELL_SIZE, WALL_SIZE

# ── Device ────────────────────────────────────────────────────────────────────
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


# ── Hyperparamètres ────────────────────────────────────────────────────────────
N_ENVS         = 32
N_STEPS        = 128            # pas par env par rollout  (128×32 = 4096)
N_EPOCHS       = 4
BATCH_SIZE     = 128
TOTAL_STEPS    = 3_000_000
LEARNING_RATE  = 2.5e-4
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
ENT_COEF_START = 0.05
ENT_COEF_END   = 0.01
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5
EVAL_EVERY     = 20_000
EVAL_EPISODES  = 100
SAVE_PATH      = "models/maze_ppo.pt"

SPATIAL_DIM = CELL_SIZE + WALL_SIZE   # 280
NAV_DIM     = 4                       # pos(2) + goal(2)


# ── Architecture ───────────────────────────────────────────────────────────────
class MemoryNavigator(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Linear(SPATIAL_DIM, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.nav = nn.Sequential(
            nn.Linear(NAV_DIM, 32),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(288, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.critic = nn.Sequential(
            nn.Linear(288, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _fuse(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.spatial(obs[..., :SPATIAL_DIM]),
            self.nav(obs[..., SPATIAL_DIM:]),
        ], dim=-1)

    def forward(self, obs: torch.Tensor):
        f = self._fuse(obs)
        return self.actor(f), self.critic(f).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


# ── GAE ────────────────────────────────────────────────────────────────────────
def compute_gae(
    rewards: torch.Tensor,    # (T, N)
    values:  torch.Tensor,    # (T, N)
    dones:   torch.Tensor,    # (T, N)
    last_v:  torch.Tensor,    # (N,)
) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    dev  = rewards.device
    adv  = torch.zeros(T, N, device=dev)
    gae  = torch.zeros(N, device=dev)

    for t in reversed(range(T)):
        nxt       = last_v if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta     = rewards[t] + GAMMA * nxt * nonterminal - values[t]
        gae       = delta + GAMMA * GAE_LAMBDA * nonterminal * gae
        adv[t]    = gae

    return adv, adv + values


# ── Évaluation (tous les épisodes en parallèle, zéro sync GPU dans la boucle) ──
_eval_env: "VecMazeEnv | None" = None   # créé une seule fois

def evaluate(model: MemoryNavigator) -> tuple[float, float]:
    from maze_env import MAX_STEPS as _MAX
    global _eval_env
    if _eval_env is None:
        _eval_env = VecMazeEnv(n_envs=EVAL_EPISODES, device=str(DEVICE))
    env         = _eval_env
    obs         = env.reset()
    wins        = torch.zeros(EVAL_EPISODES, device=DEVICE)
    ep_rewards  = torch.zeros(EVAL_EPISODES, device=DEVICE)
    done        = torch.zeros(EVAL_EPISODES, dtype=torch.bool, device=DEVICE)

    model.eval()
    with torch.no_grad():
        for _ in range(_MAX):
            logits, _           = model(obs)
            actions             = Categorical(logits=logits).sample()
            obs, reward, term, trunc = env.step(actions)
            ep_rewards         += (~done).float() * reward   # cumul tant que non terminé
            wins               += (~done & term).float()
            done               |= term | trunc

    model.train()
    win_rate   = float(wins.sum().item()) / EVAL_EPISODES
    mean_rew   = float(ep_rewards.mean().item())
    return win_rate, mean_rew


# ── Entraînement ───────────────────────────────────────────────────────────────
def train():
    os.makedirs("models", exist_ok=True)
    print(f"Device : {DEVICE}")

    env   = VecMazeEnv(n_envs=N_ENVS, device=str(DEVICE))
    model = MemoryNavigator().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

    n_updates = (TOTAL_STEPS // (N_STEPS * N_ENVS)) * N_EPOCHS
    scheduler = optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.0, total_iters=n_updates
    )

    # Buffers sur DEVICE — zéro transfert pendant la collecte
    obs_buf  = torch.zeros(N_STEPS, N_ENVS, OBS_DIM,        device=DEVICE)
    act_buf  = torch.zeros(N_STEPS, N_ENVS, dtype=torch.long, device=DEVICE)
    lp_buf   = torch.zeros(N_STEPS, N_ENVS,                  device=DEVICE)
    rew_buf  = torch.zeros(N_STEPS, N_ENVS,                  device=DEVICE)
    done_buf = torch.zeros(N_STEPS, N_ENVS,                  device=DEVICE)
    val_buf  = torch.zeros(N_STEPS, N_ENVS,                  device=DEVICE)

    obs_t         = env.reset()
    global_step   = 0
    ep_count      = 0
    best_wr       = 0.0
    last_eval     = 0

    print(f"Démarrage — {TOTAL_STEPS:,} transitions | "
          f"{N_ENVS} envs | rollout {N_STEPS}×{N_ENVS}={N_STEPS*N_ENVS:,}")

    while global_step < TOTAL_STEPS:

        # ── Collecte du rollout ───────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            for t in range(N_STEPS):
                logits, value = model(obs_t)
                dist   = Categorical(logits=logits)
                action = dist.sample()

                obs_buf[t]  = obs_t
                act_buf[t]  = action
                lp_buf[t]   = dist.log_prob(action)
                val_buf[t]  = value

                obs_t, reward, terminated, truncated = env.step(action)
                done = (terminated | truncated).float()

                rew_buf[t]  = reward
                done_buf[t] = done

                ep_count    += int(done.sum().item())
                global_step += N_ENVS

        model.train()

        # ── GAE ───────────────────────────────────────────────────────────────
        with torch.no_grad():
            _, last_v = model(obs_t)

        adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_v)

        # Aplatir (T, N) → (T*N,)
        n_samples  = N_STEPS * N_ENVS
        flat_obs   = obs_buf.view(n_samples, OBS_DIM)
        flat_act   = act_buf.view(n_samples)
        flat_lp    = lp_buf.view(n_samples)
        flat_adv   = adv.view(n_samples)
        flat_ret   = ret.view(n_samples)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        # ── Mise à jour PPO ───────────────────────────────────────────────────
        idx      = torch.randperm(n_samples, device=DEVICE)
        ent_coef = (ENT_COEF_START
                    + (ENT_COEF_END - ENT_COEF_START)
                    * (global_step / TOTAL_STEPS))

        for _ in range(N_EPOCHS):
            for start in range(0, n_samples, BATCH_SIZE):
                mb = idx[start:start + BATCH_SIZE]

                new_lp, entropy, new_val = model.evaluate_actions(
                    flat_obs[mb], flat_act[mb]
                )

                ratio   = (new_lp - flat_lp[mb]).exp()
                mb_adv  = flat_adv[mb]
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS),
                ).mean()

                vf_loss = nn.functional.mse_loss(new_val, flat_ret[mb])
                loss    = pg_loss + VF_COEF * vf_loss - ent_coef * entropy.mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()

            scheduler.step()

        # ── Évaluation périodique ─────────────────────────────────────────────
        if global_step - last_eval >= EVAL_EVERY:
            last_eval    = global_step
            wr, mean_rew = evaluate(model)
            lr           = opt.param_groups[0]["lr"]
            print(f"  step {global_step:>9,} | ep={ep_count:>6,} | "
                  f"win={wr:.0%} | rew={mean_rew:+.1f} | ent={ent_coef:.4f} | lr={lr:.2e}")
            if wr > best_wr:
                best_wr = wr
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"             -> meilleur modèle ({wr:.0%})")

    if best_wr == 0.0:
        torch.save(model.state_dict(), SAVE_PATH)

    print(f"\nTerminé. Meilleur win rate : {best_wr:.0%}")
    print(f"Modèle sauvegardé : {SAVE_PATH}")


if __name__ == "__main__":
    train()
