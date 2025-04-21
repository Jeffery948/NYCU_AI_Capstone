import gymnasium as gym
import numpy as np
import ale_py

def evaluate_random_policy(env_name, seed=0, n_episodes=100):
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)

    total_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = env.action_space.sample()  # 隨機選一個動作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"Random Policy on {env_name} over {n_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f}, Std: {std_reward:.2f}")

    return total_rewards

if __name__ == "__main__":
    gym.register_envs(ale_py)

    all_returns = []
    seeds=[0, 4, 9]

    for seed in seeds:
        print(f"\nTesting with seed {seed}...")
        returns = evaluate_random_policy("LunarLander-v3", seed, n_episodes=100)
        all_returns.extend(returns)

    print(f"\nCombined test result over {len(all_returns)} episodes from {len(seeds)} seeds:")
    print(f"Mean return: {np.mean(all_returns):.2f}, Std: {np.std(all_returns):.2f}")
    all_returns = []

    for seed in seeds:
        print(f"\nTesting with seed {seed}...")
        returns = evaluate_random_policy("ALE/Breakout-v5", seed, n_episodes=100)
        all_returns.extend(returns)

    print(f"\nCombined test result over {len(all_returns)} episodes from {len(seeds)} seeds:")
    print(f"Mean return: {np.mean(all_returns):.2f}, Std: {np.std(all_returns):.2f}")