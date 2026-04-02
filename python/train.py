"""
PPO (Proximal Policy Optimization) — implémentation pure PyTorch.
Architecture : Actor-Critic  232 → 128 → 128, têtes acteur et critique séparées.
Algorithme   : PPO avec GAE, clipping et normalisation des avantages.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from maze_env import MazeEnv, OBS_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparamètres ────────────────────────────────────────────────────────────
TOTAL_STEPS   = 500_000
N_STEPS       = 512         # pas collectés avant chaque mise à jour
N_EPOCHS      = 8           # passes PPO sur chaque rollout
BATCH_SIZE    = 64
LEARNING_RATE = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95        # λ pour le calcul des avantages (GAE)
CLIP_EPS      = 0.2         # ε de clipping PPO
ENT_COEF      = 0.02        # coefficient d'entropie (plus d'exploration)
VF_COEF       = 0.5         # coefficient de la perte critique
MAX_GRAD_NORM = 0.5
HIDDEN        = 128         # adapté à OBS_DIM=107 (10x10)
EVAL_EVERY    = 20_000
EVAL_EPISODES = 50
SAVE_PATH     = "models/maze_ppo.pt"


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
        self.actor  = nn.Linear(hidden, action_dim)   # logits de politique
        self.critic = nn.Linear(hidden, 1)             # estimation de valeur V(s)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, obs: torch.Tensor):
        logits, value = self(obs)
        dist    = Categorical(logits=logits)
        action  = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


# ── GAE (Generalized Advantage Estimation) ────────────────────────────────────
def compute_gae(
    rewards: torch.Tensor,
    values:  torch.Tensor,
    dones:   torch.Tensor,
    last_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae   = 0.0
    for t in reversed(range(len(rewards))):
        next_val         = last_value if t == len(rewards) - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta    = rewards[t] + GAMMA * next_val * next_non_terminal - values[t]
        last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ── Évaluation ────────────────────────────────────────────────────────────────
def evaluate(model: ActorCritic, n_episodes: int = EVAL_EPISODES) -> float:
    """Taux de réussite avec politique greedy (argmax)."""
    env  = MazeEnv()
    wins = 0
    model.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            while not done:
                obs_t        = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                logits, _    = model(obs_t)
                action       = torch.distributions.Categorical(logits=logits).sample().item()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            if terminated:
                wins += 1
    model.train()
    return wins / n_episodes


# ── Entraînement ──────────────────────────────────────────────────────────────
def train():
    import os
    os.makedirs("models", exist_ok=True)

    env   = MazeEnv()
    model = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Buffers du rollout (taille fixe N_STEPS) — sur le device
    obs_buf      = torch.zeros(N_STEPS, OBS_DIM,    device=DEVICE)
    actions_buf  = torch.zeros(N_STEPS, dtype=torch.long, device=DEVICE)
    logprobs_buf = torch.zeros(N_STEPS,              device=DEVICE)
    rewards_buf  = torch.zeros(N_STEPS,              device=DEVICE)
    dones_buf    = torch.zeros(N_STEPS,              device=DEVICE)
    values_buf   = torch.zeros(N_STEPS,              device=DEVICE)

    obs, _  = env.reset()
    obs_t   = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    global_step   = 0
    episode_count = 0
    best_win_rate = 0.0
    last_eval     = 0

    print(f"Démarrage PPO — {TOTAL_STEPS:,} pas, rollout={N_STEPS}, epochs={N_EPOCHS}")
    print(f"Device : {DEVICE}")

    while global_step < TOTAL_STEPS:

        # ── Collecte d'un rollout ──────────────────────────────────────────────
        for step in range(N_STEPS):
            global_step += 1

            with torch.no_grad():
                action, logprob, _, value = model.get_action(obs_t.unsqueeze(0))

            obs_buf[step]      = obs_t
            actions_buf[step]  = action
            logprobs_buf[step] = logprob
            values_buf[step]   = value.squeeze()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rewards_buf[step] = reward
            dones_buf[step]   = float(done)

            if done:
                next_obs, _ = env.reset()
                episode_count += 1

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

        # ── Calcul des avantages (GAE) ─────────────────────────────────────────
        with torch.no_grad():
            _, last_value = model(obs_t.unsqueeze(0))

        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf, last_value.squeeze()
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Mise à jour PPO ───────────────────────────────────────────────────
        indices = torch.randperm(N_STEPS, device=DEVICE)
        for _ in range(N_EPOCHS):
            for start in range(0, N_STEPS, BATCH_SIZE):
                mb = indices[start:start + BATCH_SIZE]

                new_logprobs, entropy, new_values = model.evaluate_actions(
                    obs_buf[mb], actions_buf[mb]
                )

                ratio    = (new_logprobs - logprobs_buf[mb]).exp()
                mb_advs  = advantages[mb]
                pg_loss  = torch.max(
                    -mb_advs * ratio,
                    -mb_advs * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS),
                ).mean()

                vf_loss  = nn.functional.mse_loss(new_values, returns[mb])
                ent_loss = entropy.mean()

                loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # ── Évaluation périodique ─────────────────────────────────────────────
        if global_step - last_eval >= EVAL_EVERY:
            last_eval = global_step
            win_rate  = evaluate(model)
            print(f"  step {global_step:>9,} | épisodes={episode_count:>6,} | win rate={win_rate:.0%}")
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"             → nouveau meilleur modèle ({win_rate:.0%})")

    # Sauvegarde finale
    if best_win_rate == 0.0:
        torch.save(model.state_dict(), SAVE_PATH)

    print(f"\nTerminé. Meilleur win rate : {best_win_rate:.0%}")
    print(f"Modèle : {SAVE_PATH}")
    print("Lance maintenant : python export_model.py")


if __name__ == "__main__":
    train()
