"""
PPO (Proximal Policy Optimization) — implémentation pure PyTorch.
Architecture : Actor-Critic  OBS_DIM → 128 → 128, têtes acteur et critique séparées.
Algorithme   : PPO avec GAE, clipping et normalisation des avantages.
Environnement: VecMazeEnv — 32 labyrinthes en parallèle (tenseurs CPU).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from maze_env import MazeEnv, OBS_DIM
from vec_maze_env import VecMazeEnv

# ── Hyperparamètres ────────────────────────────────────────────────────────────
N_ENVS         = 32           # environnements parallèles
TOTAL_STEPS    = 2_000_000    # pas totaux (= transitions × N_ENVS)
N_STEPS        = 64           # pas par env par rollout  (64×32 = 2048)
N_EPOCHS       = 4
BATCH_SIZE     = 64
LEARNING_RATE  = 3e-4
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
ENT_COEF_START = 0.05
ENT_COEF_END   = 0.015
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5
HIDDEN         = 128
EVAL_EVERY     = 20_000
EVAL_EPISODES  = 50
SAVE_PATH      = "models/maze_ppo.pt"

# ── Réseau Actor-Critic ────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = 4, hidden: int = HIDDEN):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, obs: torch.Tensor):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


# ── GAE vectorisé (T, N) ──────────────────────────────────────────────────────
def compute_gae(
    rewards:    torch.Tensor,   # (T, N)
    values:     torch.Tensor,   # (T, N)
    dones:      torch.Tensor,   # (T, N)
    last_value: torch.Tensor,   # (N,)
) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    advantages = torch.zeros(T, N)
    last_gae   = torch.zeros(N)

    for t in reversed(range(T)):
        next_val          = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta    = rewards[t] + GAMMA * next_val * next_non_terminal - values[t]
        last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values


# ── Évaluation (single env, stochastique) ────────────────────────────────────
def evaluate(model: ActorCritic, n_episodes: int = EVAL_EPISODES) -> float:
    env  = MazeEnv()
    wins = 0
    model.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            while not done:
                obs_t     = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _ = model(obs_t)
                action    = Categorical(logits=logits).sample().item()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            if terminated:
                wins += 1
    model.train()
    return wins / n_episodes


# ── Entraînement vectorisé ────────────────────────────────────────────────────
def train():
    import os
    os.makedirs("models", exist_ok=True)

    env   = VecMazeEnv(n_envs=N_ENVS, device="cpu")
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Buffers : (N_STEPS, N_ENVS, ...)
    obs_buf      = torch.zeros(N_STEPS, N_ENVS, OBS_DIM)
    actions_buf  = torch.zeros(N_STEPS, N_ENVS, dtype=torch.long)
    logprobs_buf = torch.zeros(N_STEPS, N_ENVS)
    rewards_buf  = torch.zeros(N_STEPS, N_ENVS)
    dones_buf    = torch.zeros(N_STEPS, N_ENVS)
    values_buf   = torch.zeros(N_STEPS, N_ENVS)

    obs_t         = env.reset()                 # (N_ENVS, OBS_DIM)
    global_step   = 0
    episode_count = 0
    best_win_rate = 0.0
    last_eval     = 0

    print(f"Démarrage PPO vectorisé — {TOTAL_STEPS:,} transitions, "
          f"{N_ENVS} envs, rollout={N_STEPS}×{N_ENVS}={N_STEPS*N_ENVS}")

    while global_step < TOTAL_STEPS:

        # ── Collecte du rollout ───────────────────────────────────────────────
        model.eval()
        for step in range(N_STEPS):
            with torch.no_grad():
                logits, value = model(obs_t)           # (N, 4), (N,)
                dist    = Categorical(logits=logits)
                action  = dist.sample()                # (N,)
                logprob = dist.log_prob(action)        # (N,)

            obs_buf[step]      = obs_t
            actions_buf[step]  = action
            logprobs_buf[step] = logprob
            values_buf[step]   = value

            next_obs, reward, terminated, truncated = env.step(action)
            done = (terminated | truncated).float()

            rewards_buf[step] = reward
            dones_buf[step]   = done

            episode_count += int(done.sum().item())
            global_step   += N_ENVS
            obs_t          = next_obs

        model.train()

        # ── GAE ───────────────────────────────────────────────────────────────
        with torch.no_grad():
            _, last_value = model(obs_t)               # (N,)

        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf, last_value
        )

        # Aplatir (T, N) → (T*N,)
        flat_obs      = obs_buf.view(-1, OBS_DIM)
        flat_actions  = actions_buf.view(-1)
        flat_logprobs = logprobs_buf.view(-1)
        flat_advs     = advantages.view(-1)
        flat_returns  = returns.view(-1)

        flat_advs = (flat_advs - flat_advs.mean()) / (flat_advs.std() + 1e-8)

        # ── Mise à jour PPO ───────────────────────────────────────────────────
        n_samples = N_STEPS * N_ENVS
        indices   = torch.randperm(n_samples)
        ent_coef  = ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * (global_step / TOTAL_STEPS)

        for _ in range(N_EPOCHS):
            for start in range(0, n_samples, BATCH_SIZE):
                mb = indices[start:start + BATCH_SIZE]

                new_logprobs, entropy, new_values = model.evaluate_actions(
                    flat_obs[mb], flat_actions[mb]
                )

                ratio   = (new_logprobs - flat_logprobs[mb]).exp()
                mb_advs = flat_advs[mb]
                pg_loss = torch.max(
                    -mb_advs * ratio,
                    -mb_advs * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS),
                ).mean()

                loss = (pg_loss
                        + VF_COEF * nn.functional.mse_loss(new_values, flat_returns[mb])
                        - ent_coef * entropy.mean())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # ── Évaluation périodique ─────────────────────────────────────────────
        if global_step - last_eval >= EVAL_EVERY:
            last_eval = global_step
            win_rate  = evaluate(model)
            print(f"  step {global_step:>9,} | episodes={episode_count:>7,} | "
                  f"win rate={win_rate:.0%} | ent={ent_coef:.4f}")
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"             -> nouveau meilleur modele ({win_rate:.0%})")

    if best_win_rate == 0.0:
        torch.save(model.state_dict(), SAVE_PATH)

    print(f"\nTermine. Meilleur win rate : {best_win_rate:.0%}")
    print(f"Modele : {SAVE_PATH}")
    print("Lance maintenant : python export_model.py")


if __name__ == "__main__":
    train()
