# Deep Reinforcement Learning: REINFORCE & DQN on LunarLander-v3 and Assault

This project explores deep reinforcement learning (DRL) on two environments—**LunarLander-v3** (Box2D) and **Assault-v5** (Atari)—using two representative algorithms: **REINFORCE** and **DQN**, along with various enhancement techniques inspired by Rainbow DQN.

## Environments
- **LunarLander-v3** (Gymnasium Box2D)
- **ALE/Assault-v5** (Atari)

## Algorithms

### REINFORCE
A Monte Carlo Policy Gradient method trained episode-by-episode.

Enhancement Techniques:
- Reward Normalization
- Baseline Function (Value Network)
- Generalized Advantage Estimation (GAE)
- Entropy Regularization

### DQN
Value-based off-policy method with experience replay and target network.

Rainbow-inspired Enhancements:
- Reward Clipping
- Double DQN
- Dueling Network
- Prioritized Experience Replay (PER)

## Ablation Study
Each algorithm is tested across various combinations of enhancements, including:
- Individual techniques applied incrementally
- Full combined model

Evaluation is conducted over multiple random seeds, and learning curves are recorded.

## Getting Started

### Installation
```bash
conda create -n DRL python=3.9
conda activate DRL
pip install -r requirements.txt
```

### Run Training
```
# REINFORCE on LunarLander
python REINFORCE.py --env LunarLander-v3 --seed 0

# DQN on Assault with Prioritized Experience Replay
python DQN.py --env ALE/Assault-v5 --seed 0 --PER True
```

### Record Demo Video
```
python record.py --env ALE/Assault-v5 --model_path outputs/Vanilla/ALE_Assault-v5_Q0.pth --n_episodes 1
```

## Directory Structure
```
.
├── REINFORCE.py             # Main training loop for REINFORCE
├── DQN.py                   # Main training loop for DQN
├── utils_reinforce.py       # Policy network & logic for REINFORCE
├── utils_DQN.py             # Q-networks, buffers, agents for DQN
├── record.py                # Recording demo videos
├── logs/                    # CSV logs for training curves
├── outputs/                 # Saved models
├── graphs/                  # Learning curve plots
└── videos/                  # Recorded gameplay demos
```

## Demo Videos
- [LunarLander-v3 Demo](https://youtu.be/iTTuoe_qEwA)
- [Assault-v5 Demo](https://youtu.be/J-nJomskWcY)