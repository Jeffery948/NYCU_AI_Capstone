import argparse
import torch
import numpy as np
import gymnasium as gym
import ale_py
from utils_reinforce import Actor
from utils_DQN import Agent
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack

def test_policy(env_name, seed, model_path, n_episodes):
    """
        Test the learned model
    """
    gym.register_envs(ale_py)

    if env_name != 'LunarLander-v3':
        env = gym.make(env_name, frameskip=1, render_mode='human')
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
        env = FrameStack(env, num_stack=4)
    else:
        env = gym.make(env_name, render_mode='human')
        
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    max_steps = env.spec.max_episode_steps if env.spec.max_episode_steps is not None else 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if env_name != 'LunarLander-v3':
        agent = Agent(env, device=device)
        agent.evaluate_net.load_state_dict(torch.load(model_path, map_location=device))
        agent.evaluate_net.eval()
    else:
        agent = Actor(env, hidden_size=128, gamma=0.99, device=device).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()

    returns = []

    with torch.no_grad():
        for ep in range(n_episodes):
            state, _ = env.reset()
            ep_return = 0
            for _ in range(max_steps):
                if env_name != 'LunarLander-v3':
                    action = agent.select_action(state, epsilon=0.0)
                else:
                    action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_return += reward
                env.render()
                if done:
                    break
                state = next_state
            returns.append(ep_return)

    print(f"Test result over {n_episodes} episodes:")
    print(f"Mean return: {np.mean(returns):.2f}, Std: {np.std(returns):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='ALE/Assault-v5')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--model_path", default='outputs/Vanilla/ALE_Assault-v5_Q0.pth')
    parser.add_argument("--n_episodes", default=5, type=int)
    args = parser.parse_args()

    test_policy(args.env, args.seed, args.model_path, args.n_episodes)