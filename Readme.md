# Documentation DQN — Deep Q-Network
### Appliqué à MountainCarContinuous-v0

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Le problème : Q-Learning classique](#2-le-problème--q-learning-classique)
3. [Architecture du réseau de neurones](#3-architecture-du-réseau-de-neurones)
4. [Les composants clés du DQN](#4-les-composants-clés-du-dqn)
   - 4.1 [Replay Buffer](#41-replay-buffer)
   - 4.2 [Target Network](#42-target-network)
   - 4.3 [Politique ε-greedy](#43-politique-ε-greedy)
5. [La boucle d'entraînement](#5-la-boucle-dentraînement)
6. [Le calcul de la perte (loss)](#6-le-calcul-de-la-perte-loss)
7. [Double DQN](#7-double-dqn)
8. [Reward Shaping](#8-reward-shaping)
9. [Discrétisation des actions continues](#9-discrétisation-des-actions-continues)
10. [Hyperparamètres expliqués](#10-hyperparamètres-expliqués)
11. [Flux complet de l'algorithme](#11-flux-complet-de-lalgorithme)
12. [Problèmes courants et solutions](#12-problèmes-courants-et-solutions)

---

## 1. Vue d'ensemble

Le **DQN (Deep Q-Network)** est un algorithme de **Reinforcement Learning** (RL) qui combine :

- le **Q-Learning** (algorithme de RL classique basé sur les valeurs)
- les **réseaux de neurones profonds** (pour approximer la fonction Q)

Il a été introduit par DeepMind en 2013 et a permis à une IA d'apprendre à jouer aux jeux Atari directement depuis les pixels.

### Idée centrale

L'agent apprend une fonction **Q(s, a)** qui estime la **récompense future cumulée** attendue si on est dans l'état `s` et qu'on prend l'action `a`.

```
Q(s, a) = récompense immédiate + récompenses futures espérées
```

Une fois cette fonction apprise, la stratégie optimale est simplement :

```
action* = argmax_a Q(s, a)
```

→ On choisit toujours l'action qui maximise Q.

---

## 2. Le problème : Q-Learning classique

En Q-Learning classique, on stocke Q dans une **table** (Q-table) :

```
Q-table[état][action] = valeur
```

**Problème** : MountainCar a un espace d'états continu (position ∈ [-1.2, 0.6], vitesse ∈ [-0.07, 0.07]). Une table avec toutes les combinaisons possibles serait **infinie**.

**Solution DQN** : on remplace la table par un **réseau de neurones** qui prend l'état en entrée et retourne les valeurs Q pour toutes les actions.

```
         Réseau de neurones
état s ──────────────────────▶  [Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)]
(2 valeurs)                      (21 valeurs, une par action)
```

---

## 3. Architecture du réseau de neurones

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),   # couche 1 : 2 → 256
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # couche 2 : 256 → 256
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), # couche 3 : 256 → 128
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions),   # sortie : 128 → 21
        )
```

### Entrée
- **2 neurones** : position et vitesse de la voiture

### Sortie
- **21 neurones** : une valeur Q par action discrète possible

### Couches cachées
- **ReLU** (Rectified Linear Unit) : fonction d'activation `f(x) = max(0, x)`
  - Introduit la non-linéarité nécessaire pour apprendre des patterns complexes
  - Évite le problème de disparition du gradient

### Visualisation

```
Input       Hidden 1    Hidden 2    Hidden 3    Output
[pos ]  ──▶ [256   ] ──▶ [256   ] ──▶ [128   ] ──▶ [Q(a₁) ]
[vel ]      [  ...  ]    [  ...  ]    [  ...  ]    [Q(a₂) ]
            [  ...  ]    [  ...  ]    [  ...  ]    [  ...  ]
                                                   [Q(a₂₁)]
```

---

## 4. Les composants clés du DQN

### 4.1 Replay Buffer

**Problème sans Replay Buffer** : si on entraîne le réseau sur chaque transition une par une, les données consécutives sont **fortement corrélées** (t, t+1, t+2...). Le réseau ne généralise pas bien.

**Solution** : on stocke les expériences passées et on tire des **mini-batchs aléatoires**.

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # file circulaire

    def push(self, s, a, r, s_, done):
        # Stocke une transition (état, action, récompense, état suivant, fin?)
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        # Tire batch_size transitions au hasard
        batch = random.sample(self.buffer, batch_size)
        ...
```

Une **transition** est un tuple `(s, a, r, s', done)` :

| Symbole | Signification | Exemple |
|---------|---------------|---------|
| `s`     | état courant  | `[-0.5, 0.01]` (position, vitesse) |
| `a`     | action choisie (index) | `15` (force = +0.4) |
| `r`     | récompense reçue | `-0.1` |
| `s'`    | état suivant  | `[-0.49, 0.012]` |
| `done`  | épisode terminé ? | `False` |

**Avantages** :
- Casse les corrélations temporelles
- Chaque expérience peut être utilisée plusieurs fois
- Stabilise l'entraînement

### 4.2 Target Network

**Problème sans Target Network** : on calcule la cible d'apprentissage avec le même réseau qu'on est en train de modifier → la cible "bouge" à chaque update → instabilité, oscillations, divergence.

**Solution** : deux réseaux identiques :

```
q_net      → réseau principal, mis à jour à chaque step
target_net → copie gelée, mise à jour toutes les N=5 épisodes
```

```python
# Calcul de la cible (target) avec le réseau gelé
with torch.no_grad():
    next_q  = target_net(next_states).max(1)[0]
    target  = reward + gamma * next_q * (1 - done)

# Mise à jour du réseau principal
loss = loss_fn(q_net(states), target)
optimizer.step()

# Sync périodique
if episode % TARGET_UPDATE == 0:
    target_net.load_state_dict(q_net.state_dict())
```

**Analogie** : c'est comme viser une cible fixe plutôt qu'une cible qui se déplace en même temps que vous tirez.

### 4.3 Politique ε-greedy

L'agent doit équilibrer **exploration** (découvrir de nouvelles actions) et **exploitation** (utiliser ce qu'il a appris).

```python
if random.random() < epsilon:
    action = random.randrange(n_actions)  # exploration : action aléatoire
else:
    action = q_net(state).argmax()        # exploitation : meilleure action connue
```

**Évolution de ε au cours du temps** :

```
ε = 1.0  →  0.998  →  0.996  →  ...  →  0.01
        (decay × 0.998 après chaque épisode)

Ep 1-50  : ~100% aléatoire  (exploration pure)
Ep 200   : ~67% aléatoire
Ep 400   : ~45% aléatoire
Ep 800   : ~1%  aléatoire   (exploitation quasi-totale)
```

---

## 5. La boucle d'entraînement

```
┌─────────────────────────────────────────────────────────────────┐
│  Pour chaque épisode :                                          │
│                                                                 │
│    état = env.reset()                                           │
│                                                                 │
│    ┌─── Tant que non terminé : ───────────────────────────┐    │
│    │                                                       │    │
│    │  1. Choisir action (ε-greedy)                        │    │
│    │     ├── avec prob ε  : action aléatoire              │    │
│    │     └── avec prob 1-ε: argmax Q(état)                │    │
│    │                                                       │    │
│    │  2. Exécuter action dans l'environnement             │    │
│    │     → obtenir (état', récompense, done)              │    │
│    │                                                       │    │
│    │  3. Stocker (s, a, r, s', done) dans ReplayBuffer    │    │
│    │                                                       │    │
│    │  4. Si buffer assez plein :                          │    │
│    │     → Tirer un mini-batch                            │    │
│    │     → Calculer la perte (loss)                       │    │
│    │     → Backpropagation + update des poids             │    │
│    │                                                       │    │
│    │  5. état = état'                                      │    │
│    └───────────────────────────────────────────────────────┘    │
│                                                                 │
│    Décroître ε                                                  │
│    Si épisode % 5 == 0 : synchroniser target_net               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Le calcul de la perte (loss)

C'est le **cœur mathématique** du DQN.

### Équation de Bellman

La valeur Q optimale satisfait l'équation de Bellman :

```
Q*(s, a) = r + γ · max_a' Q*(s', a')
            ↑         ↑
   récompense       meilleure valeur
   immédiate        dans l'état suivant
```

où `γ` (gamma) est le **facteur d'actualisation** (discount factor), ici `γ = 0.99`.

### Calcul dans le code

```python
# 1. Q prédit par le réseau principal pour les actions effectuées
q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#                         └── on sélectionne la Q-valeur de l'action choisie

# 2. Q cible calculée avec le réseau gelé (Double DQN)
with torch.no_grad():
    best_actions = q_net(next_states).argmax(1, keepdim=True)  # choix de l'action
    next_q       = target_net(next_states).gather(1, best_actions).squeeze(1)  # évaluation
    target       = reward + gamma * next_q * (1 - done)
    #                                          └── = 0 si épisode terminé (pas de futur)

# 3. Perte = différence entre prédit et cible
loss = SmoothL1Loss(q_values, target)  # Huber loss
```

### Huber Loss vs MSE

On utilise la **Huber Loss** (SmoothL1) plutôt que MSE :

```
        │ 0.5·(y - ŷ)²      si |y - ŷ| ≤ 1    ← quadratique (lisse)
L(y,ŷ) =│
        │ |y - ŷ| - 0.5     si |y - ŷ| > 1    ← linéaire (robuste aux outliers)
```

**Pourquoi ?** Les Q-values peuvent avoir des erreurs très grandes au début de l'entraînement. MSE les amplifierait (erreur²), déstabilisant les gradients. Huber est plus robuste.

---

## 7. Double DQN

**Problème du DQN standard** : il a tendance à **surestimer** les Q-values.

Avec le DQN classique :
```
target = r + γ · max_a' target_net(s', a')
```
Le `max` introduit un biais positif car on prend toujours le maximum d'estimations bruitées.

**Solution Double DQN** : on sépare le **choix** de l'action et son **évaluation** :

```python
# Choix de la meilleure action → réseau PRINCIPAL (à jour)
best_actions = q_net(next_states).argmax(1, keepdim=True)

# Évaluation de cette action → réseau CIBLE (stable)
next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
```

```
DQN standard  : target_net choisit ET évalue  → biais haut
Double DQN    : q_net choisit, target_net évalue → biais réduit
```

---

## 8. Reward Shaping

MountainCar a un **reward très sparse** : +100 seulement quand la voiture atteint le sommet, -0.1×action² à chaque step sinon.

Sans jamais atteindre le sommet par hasard (quasiment impossible en exploration aléatoire), l'agent ne reçoit jamais de signal positif et n'apprend rien.

### Solution : ajouter un signal dense

```python
def shaped_reward(state, next_state, reward):
    pos,  vel  = state
    npos, nvel = next_state

    # Bonus altitude : sin(3·x) reflète la forme de la vallée
    # Plus haute est la position, meilleur est le bonus
    potential_gain = 3.0 * (np.sin(3 * npos) - np.sin(3 * pos))

    # Bonus vitesse : encourage à prendre de l'élan
    speed_gain = 10.0 * (abs(nvel) - abs(vel))

    return reward + potential_gain + speed_gain
```

### Pourquoi sin(3·x) ?

La dynamique de l'environnement utilise `cos(3·pos)` dans la mise à jour de la vitesse :
```
velocity_new = velocity + force × 0.0015 - 0.0025 × cos(3 × position)
```
La colline a donc la forme `sin(3·x)`. Donner un bonus proportionnel à cette hauteur guide naturellement la voiture vers le sommet.

**Important** : le reward shaping est appliqué **seulement pendant l'entraînement**. Pour évaluer les performances, on utilise le reward original de l'environnement.

---

## 9. Discrétisation des actions continues

MountainCarContinuous a un espace d'action **continu** : force ∈ [-1.0, 1.0].

Le DQN standard ne fonctionne qu'avec des actions **discrètes** (il prédit une Q-value par action possible).

**Solution** : on discrétise l'espace continu en N bins uniformément espacés.

```python
N_ACTIONS    = 21
ACTION_SPACE = np.linspace(-1.0, 1.0, N_ACTIONS)
# → [-1.0, -0.9, -0.8, ..., -0.1, 0.0, 0.1, ..., 0.8, 0.9, 1.0]
```

Le réseau prédit 21 Q-values, et on mappe l'index choisi vers la force réelle :

```python
action_idx  = q_net(state).argmax()          # ex : 15
action_cont = ACTION_SPACE[action_idx]       # ex : 0.4
env.step([action_cont])                       # force = 0.4
```

**Compromis** :
- Plus de bins → plus de précision → mais plus lent à apprendre
- 21 bins offre un bon équilibre pour ce problème

---

## 10. Hyperparamètres expliqués

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `N_ACTIONS` | 21 | Nombre de bins pour discrétiser l'action |
| `HIDDEN_SIZE` | 256 | Largeur des couches cachées |
| `LR` | 5e-4 | Learning rate Adam — vitesse d'apprentissage |
| `GAMMA` | 0.99 | Discount factor — importance des récompenses futures |
| `BUFFER_SIZE` | 100 000 | Capacité du replay buffer |
| `BATCH_SIZE` | 128 | Nombre de transitions par update |
| `EPSILON_START` | 1.0 | Exploration initiale (100% aléatoire) |
| `EPSILON_END` | 0.01 | Exploration minimale (1% aléatoire) |
| `EPSILON_DECAY` | 0.998 | Multiplicateur de ε après chaque épisode |
| `TARGET_UPDATE` | 5 | Fréquence de sync du target network |
| `MAX_EPISODES` | 800 | Nombre total d'épisodes d'entraînement |
| `SHAPING_COEF` | 3.0 | Intensité du bonus d'altitude |

### Focus sur GAMMA

```
γ = 0.0 → agent myope, ne regarde que la récompense immédiate
γ = 0.5 → récompense dans 10 steps vaut 0.5^10 ≈ 0.001 (quasi ignorée)
γ = 0.99 → récompense dans 10 steps vaut 0.99^10 ≈ 0.90 (très importants)
γ = 1.0 → toutes les récompenses futures comptent également (peut diverger)
```

Pour MountainCar, on a besoin d'un γ élevé car la récompense finale (+100) est loin dans le temps.

---

## 11. Flux complet de l'algorithme

```
INITIALISATION
├── Créer q_net (poids aléatoires)
├── Copier q_net → target_net
├── Créer ReplayBuffer (vide)
└── ε = 1.0

POUR chaque épisode e = 1..800 :
│
├── s = env.reset()
│
├── POUR chaque step t :
│   │
│   ├── [CHOIX ACTION]
│   │   ├── si rand() < ε : a = aléatoire
│   │   └── sinon         : a = argmax_a Q_θ(s, a)
│   │
│   ├── [INTERACTION ENV]
│   │   └── s', r, done = env.step(ACTION_SPACE[a])
│   │
│   ├── [REWARD SHAPING]
│   │   └── r_shaped = r + Δpotential + Δspeed
│   │
│   ├── [STOCKER]
│   │   └── buffer.push(s, a, r_shaped, s', done)
│   │
│   ├── [ENTRAÎNEMENT] si |buffer| ≥ 128 :
│   │   │
│   │   ├── Tirer mini-batch de 128 transitions
│   │   │
│   │   ├── [Double DQN targets]
│   │   │   ├── a* = argmax_a Q_θ(s', a)         ← q_net choisit
│   │   │   ├── Q_next = Q_θ⁻(s', a*)            ← target_net évalue
│   │   │   └── target = r + 0.99 · Q_next · (1-done)
│   │   │
│   │   ├── Q_pred = Q_θ(s).gather(actions)
│   │   ├── loss = HuberLoss(Q_pred, target)
│   │   ├── loss.backward()
│   │   ├── clip_grad_norm_(1.0)
│   │   └── optimizer.step()
│   │
│   ├── s = s'
│   └── si done : break
│
├── ε = max(0.01, ε × 0.998)
├── scheduler.step()
└── si e % 5 == 0 : target_net ← q_net  [sync]

SAUVEGARDE si meilleur reward → dqn_mountain_car.pth
```

---

## 12. Problèmes courants et solutions

### ❌ L'agent ne bouge pas / reward ≈ 0

**Cause** : reward trop sparse, l'agent n'atteint jamais le sommet par hasard.
**Solution** : activer le **reward shaping** (`SHAPING = True`).

---

### ❌ L'entraînement diverge (loss explose)

**Causes possibles** :
- Learning rate trop élevé → réduire `LR` (ex : 1e-4)
- Q-values surestimées → utiliser **Double DQN**
- Gradients explosifs → activer `clip_grad_norm_`

---

### ❌ ε trop élevé en fin d'entraînement

Si ε > 0.1 après 800 épisodes, l'agent explore encore trop.
**Solutions** :
- Augmenter `EPSILON_DECAY` (ex : 0.999 → 0.995)
- Ou augmenter `MAX_EPISODES`

Avec `EPSILON_DECAY = 0.998` sur 800 épisodes :
```
ε_final = 1.0 × 0.998^800 ≈ 0.20
```
Avec `EPSILON_DECAY = 0.995` sur 800 épisodes :
```
ε_final = 1.0 × 0.995^800 ≈ 0.018  ✅
```

---

### ❌ Le modèle sauvegardé est mauvais

Le modèle est sauvegardé quand `total_reward > best_reward`. Si l'entraînement est trop court, même le "meilleur" modèle n'est pas bon.
**Solution** : lancer plus d'épisodes, ou vérifier la courbe `training_curve.png`.

---

### ✅ Signe que l'entraînement se passe bien

Dans la console, on doit voir la colonne `Avg-20` augmenter progressivement :

```
 Épisode       Avg-20      Best       ε    Succès
──────────────────────────────────────────────────
      20      -98.12    -45.30   0.9608        0
      40      -94.87    -40.12   0.9231        0
     100      -72.45    -10.23   0.8187        1
     200      -38.12     55.40   0.6703        4
     400       12.34     82.10   0.4493       12
     600       68.90     95.20   0.3012       18
     800       88.45     97.30   0.2018       19   ← bon modèle
```

Un reward final > **90** indique que la voiture atteint le sommet de manière fiable.

---

*Documentation rédigée pour MountainCarContinuous-v0 avec DQN + Double DQN + Reward Shaping*
