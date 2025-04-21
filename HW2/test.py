import torch
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)
print(torch.cuda.is_available())

env = gym.make("LunarLander-v3")
print(env)

env = gym.make("ALE/Assault-v5")
print(env)