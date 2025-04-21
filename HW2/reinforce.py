import os
import csv
import argparse
import gymnasium as gym
import numpy as np

import torch
import torch.optim as optim
from utils_reinforce import Actor, Critic, evaluate

def train(args, device, env, features_str):
    """
        Train the model using Adam (via backpropagation)
        (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update the policy network at the end of episode

        (2): In each episode, 
        1. evaluate the policy by running it for 10 episodes
    """
    
    # Instantiate the policy model
    if features_str == "Vanilla":
        agent = Actor(env, args.hidden_size, args.discount, device).to(device)
        actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    elif features_str == "RewardNorm":
        agent = Actor(env, args.hidden_size, args.discount, device, reward_norm=True).to(device)
        actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    elif features_str == "Baseline":
        agent = Actor(env, args.hidden_size, args.discount, device, baseline=True).to(device)
        actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
        critic = Critic(env, args.hidden_size, device).to(device)
        critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate)
    elif features_str == "GAE":
        agent = Actor(env, args.hidden_size, args.discount, device, GAE=True, gae_lambda=args.lambda_).to(device)
        actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
        critic = Critic(env, args.hidden_size, device).to(device)
        critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate)
    elif features_str == "EntropyReg":
        agent = Actor(env, args.hidden_size, args.discount, device, regularization=True).to(device)
        actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    elif features_str == "RewardNorm_Baseline_EntropyReg":
        agent = Actor(env, args.hidden_size, args.discount, device, reward_norm=True, baseline=True, regularization=True).to(device)
        actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
        critic = Critic(env, args.hidden_size, device).to(device)
        critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate)
    elif features_str == "RewardNorm_GAE_EntropyReg":
        agent = Actor(env, args.hidden_size, args.discount, device, reward_norm=True, GAE=True, gae_lambda=args.lambda_, regularization=True).to(device)
        actor_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
        critic = Critic(env, args.hidden_size, device).to(device)
        critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate)

    save_env_name = args.env
    episodes = 4000
    eval_interval = 20
    best_eval_returns = -np.inf
    max_steps = env.spec.max_episode_steps if env.spec.max_episode_steps is not None else 1500
	
    log_path = './logs/{}/{}_seed{}_eval_log.csv'.format(features_str, save_env_name, args.seed)
    csv_file = open(log_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "eval_return"])

    for i_episode in range(episodes + 1):
        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0
        t = 0
        
        # For each episode, only run environment maximum steps to avoid entering infinite loop during the learning process
        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)   
            done = terminated or truncated
            ep_reward += reward

            if done:
                break
            state = next_state

        actor_optimizer.zero_grad()
        policy_loss, critic_loss = agent.calculate_loss(critic) if args.baseline or args.GAE else agent.calculate_loss()
        policy_loss.backward()
        actor_optimizer.step()

        if args.baseline or args.GAE:
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
        
        agent.clear_memory()
        
        # log the results
        if critic_loss.item() != 0:
            print(f"Episode {i_episode}\tlength: {t}\treward: {ep_reward:.2f}\t actor loss: {policy_loss.item():.4f}\t critic loss: {critic_loss.item():.4f}")
        else:
            print(f"Episode {i_episode}\tlength: {t}\treward: {ep_reward:.2f}\t actor loss: {policy_loss.item():.4f}")

        # evaluate the policy every eval_interval episodes
        if i_episode % eval_interval == 0:
            eval_return = evaluate(agent, args.env, args.seed, n_episodes=10)
            print('Evaluation return: {:.2f}'.format(eval_return))
            csv_writer.writerow([i_episode, eval_return])

            if eval_return > best_eval_returns:
                torch.save(agent.state_dict(), './outputs/{}/{}_actor{}.pth'.format(features_str, save_env_name, args.seed))
                best_eval_returns = eval_return
				
    csv_file.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default='LunarLander-v3')           # OpenAI gym environment name
	parser.add_argument("--seed", default=10, type=int)               # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--learning_rate", default=1e-3, type=float) # Learning rate
	parser.add_argument("--discount", default=0.99, type=float)      # Discount factor
	parser.add_argument("--hidden_size", default=128, type=int)      # Hidden layer size
	parser.add_argument("--lambda_", default=0.98, type=float)       # Lamdda for GAE
	parser.add_argument("--reward_nor", default=False, type=bool)    # Whether to normalize the reward
	parser.add_argument("--baseline", default=False, type=bool)      # Whether to use baseline
	parser.add_argument("--GAE", default=False, type=bool)           # Whether to use GAE
	parser.add_argument("--regularize", default=False, type=bool)    # Whether to use regularization
	args = parser.parse_args()

	enabled_features = []
	if args.reward_nor:
		enabled_features.append("RewardNorm")
	if args.baseline:
		enabled_features.append("Baseline")
	if args.GAE:
		enabled_features.append("GAE")
	if args.regularize:
		enabled_features.append("EntropyReg")

	features_str = "_".join(enabled_features) if enabled_features else "Vanilla"

	print("---------------------------------------")    
	print(f"Setting: Training REINFORCE, Env: {args.env}, Seed: {args.seed}")
	print(f"Enabled Techniques: {features_str}")
	print("---------------------------------------")

	if not os.path.exists("./outputs"):
		os.makedirs("./outputs")
            
	if not os.path.exists("./logs"):
		os.makedirs("./logs")
            
	if not os.path.exists("./outputs/{}".format(features_str)):
		os.makedirs("./outputs/{}".format(features_str))
    
	if not os.path.exists("./logs/{}".format(features_str)):
		os.makedirs("./logs/{}".format(features_str))
		
	env = gym.make(args.env)
	env.reset(seed=args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train(args, device, env, features_str)