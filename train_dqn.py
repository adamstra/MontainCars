"""
DQN amélioré pour MountainCarContinuous-v0
Corrections :
  - Reward shaping basé sur l'énergie mécanique
  - Plus d'épisodes + epsilon decay plus agressif
  - Batch size plus grand, LR ajusté
  - Sauvegarde du meilleur modèle + courbe d'apprentissage
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Hyperparamètres ────────────────────────────────────────────────
ENV_NAME       = "MountainCarContinuous-v0"
N_ACTIONS      = 21            # plus de granularité : [-1, -0.9, ..., 0.9, 1]
HIDDEN_SIZE    = 256
LR             = 5e-4
GAMMA          = 0.99
BUFFER_SIZE    = 100_000
BATCH_SIZE     = 128
EPSILON_START  = 1.0
EPSILON_END    = 0.01
EPSILON_DECAY  = 0.998         # decay plus lent mais sur plus d'épisodes
TARGET_UPDATE  = 5
MAX_EPISODES   = 800
SAVE_PATH      = "dqn_mountain_car.pth"
PLOT_PATH      = "training_curve.png"

# Reward shaping
SHAPING        = True          # activer le reward shaping
SHAPING_COEF   = 3.0          # coefficient de l'énergie potentielle
# ────────────────────────────────────────────────────────────────────

ACTION_SPACE = np.linspace(-1.0, 1.0, N_ACTIONS)


# ─── Réseau Q (plus profond) ─────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ─── Replay Buffer ───────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s_)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Reward Shaping ──────────────────────────────────────────────────
def shaped_reward(state, next_state, reward):
    """
    Bonus basé sur l'énergie potentielle : sin(3*pos) représente la colline.
    On récompense le gain d'altitude ET de vitesse (énergie cinétique).
    """
    pos, vel   = state
    npos, nvel = next_state

    # Énergie potentielle : plus haute = meilleure
    potential_now  = SHAPING_COEF * np.sin(3 * npos)
    potential_prev = SHAPING_COEF * np.sin(3 * pos)

    # Bonus vitesse absolue (encourage à prendre de l'élan)
    speed_bonus = 10.0 * (abs(nvel) - abs(vel))

    return reward + (potential_now - potential_prev) + speed_bonus


# ─── Agent DQN ───────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim, n_actions):
        self.n_actions = n_actions
        self.epsilon   = EPSILON_START
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device : {self.device}")

        self.q_net      = QNetwork(state_dim, n_actions, HIDDEN_SIZE).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, HIDDEN_SIZE).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss : plus stable que MSE

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_t).argmax().item()

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return None

        s, a, r, s_, d = self.buffer.sample(BATCH_SIZE)
        s, a, r, s_, d = (x.to(self.device) for x in (s, a, r, s_, d))

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN : action choisie par q_net, évaluée par target_net
            best_actions = self.q_net(s_).argmax(1, keepdim=True)
            next_q       = self.target_net(s_).gather(1, best_actions).squeeze(1)
            target       = r + GAMMA * next_q * (1 - d)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)  # gradient clipping
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path):
        torch.save({
            "q_net_state":  self.q_net.state_dict(),
            "epsilon":      self.epsilon,
            "n_actions":    self.n_actions,
            "hidden_size":  HIDDEN_SIZE,
        }, path)


# ─── Plot courbe d'apprentissage ─────────────────────────────────────
def save_plot(rewards, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.4, color="#4a9eff", label="reward brut")
    window = min(20, len(rewards))
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), ma, color="#ff6b35",
                linewidth=2, label=f"moyenne mobile ({window} ep)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(90, color="green", linestyle="--", linewidth=0.8, label="succès (~90+)")
    ax.set_xlabel("Épisode"); ax.set_ylabel("Reward total")
    ax.set_title("Courbe d'apprentissage DQN — MountainCarContinuous")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"📊  Courbe sauvegardée → {path}")


# ─── Boucle d'entraînement ───────────────────────────────────────────
def train():
    env   = gym.make(ENV_NAME)
    agent = DQNAgent(state_dim=2, n_actions=N_ACTIONS)

    rewards_history = []
    best_reward     = -np.inf
    success_count   = 0

    print(f"\n{'Épisode':>8}  {'Avg-20':>10}  {'Best':>8}  {'ε':>7}  {'Succès':>7}")
    print("─" * 50)

    for ep in range(1, MAX_EPISODES + 1):
        state, _     = env.reset()
        total_reward = 0.0

        while True:
            action_idx  = agent.select_action(state)
            action_cont = np.array([ACTION_SPACE[action_idx]], dtype=np.float32)
            next_state, reward, terminated, truncated, _ = env.step(action_cont)
            done = terminated or truncated

            # Reward shaping (seulement pendant l'entraînement)
            r_shaped = shaped_reward(state, next_state, reward) if SHAPING else reward

            agent.buffer.push(state, action_idx, r_shaped, next_state, float(done))
            agent.train_step()

            state        = next_state
            total_reward += reward   # on logue le reward ORIGINAL (non shapé)

            if done:
                break

        agent.decay_epsilon()
        agent.scheduler.step()
        rewards_history.append(total_reward)

        if terminated:   # a atteint le sommet
            success_count += 1

        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(SAVE_PATH)

        if ep % 20 == 0:
            avg20 = np.mean(rewards_history[-20:])
            print(f"{ep:>8}  {avg20:>10.2f}  {best_reward:>8.2f}  "
                  f"{agent.epsilon:>7.4f}  {success_count:>7}")
            success_count = 0
            save_plot(rewards_history, PLOT_PATH)

    env.close()

    save_plot(rewards_history, PLOT_PATH)
    print(f"\n🏆  Meilleur reward (original) : {best_reward:.2f}")
    print(f"📦  Modèle sauvegardé          : {SAVE_PATH}")
    print(f"📊  Courbe d'apprentissage      : {PLOT_PATH}")


if __name__ == "__main__":
    train()