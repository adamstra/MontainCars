"""
Test du modèle DQN v2 entraîné sur MountainCarContinuous-v0
Rendu visuel avec pygame (render_mode="human")
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# ─── Paramètres ─────────────────────────────────────────────────────
SAVE_PATH    = "dqn_mountain_car.pth"
N_EPISODES   = 5
ACTION_SPACE_DEFAULT_SIZE = 21   # mis à jour pour v2
# ────────────────────────────────────────────────────────────────────


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


def load_model(path):
    if not os.path.exists(path):
        print(f"❌  Fichier introuvable : {path}")
        print("   Lance d'abord : python train_dqn_v2.py")
        sys.exit(1)

    checkpoint  = torch.load(path, map_location="cpu", weights_only=False)
    n_actions   = checkpoint.get("n_actions",   ACTION_SPACE_DEFAULT_SIZE)
    hidden_size = checkpoint.get("hidden_size", 256)

    model = QNetwork(state_dim=2, n_actions=n_actions, hidden_size=hidden_size)
    model.load_state_dict(checkpoint["q_net_state"])
    model.eval()

    print(f"✅  Modèle chargé depuis {path}")
    print(f"   Actions discrètes : {n_actions}  |  Hidden : {hidden_size}")
    print(f"   Epsilon sauvegardé : {checkpoint.get('epsilon', '?'):.4f}")

    action_space = np.linspace(-1.0, 1.0, n_actions)
    return model, action_space


def test(render=True):
    render_mode = "human" if render else "rgb_array"
    env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)

    model, action_space = load_model(SAVE_PATH)

    print(f"\n{'Épisode':>8}  {'Reward':>10}  {'Steps':>6}  {'Succès':>8}")
    print("─" * 42)

    results = []
    for ep in range(1, N_EPISODES + 1):
        state, _     = env.reset()
        total_reward = 0.0
        steps        = 0
        success      = False

        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_idx = model(state_t).argmax().item()

            action_cont = np.array([action_space[action_idx]], dtype=np.float32)
            state, reward, terminated, truncated, _ = env.step(action_cont)

            total_reward += reward
            steps        += 1

            if terminated:
                success = True
            if terminated or truncated:
                break

        symbol = "🏆" if success else "❌"
        print(f"{ep:>8}  {total_reward:>10.2f}  {steps:>6}  {symbol}")
        results.append(success)

    env.close()

    wins = sum(results)
    print(f"\n   Résultat final : {wins}/{N_EPISODES} succès "
          f"({'✅ bon modèle !' if wins >= 3 else '⚠️  modèle à ré-entraîner'})")


if __name__ == "__main__":
    # Passe False si tu veux sans fenêtre graphique
    test(render=True)