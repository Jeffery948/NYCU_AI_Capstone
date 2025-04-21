import os
import csv
import argparse
import gymnasium as gym
import ale_py
import numpy as np

import torch
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack
from utils_DQN import Agent, evaluate, sanitize_env_name, reset_and_fire

def train(args, device, env, features_str):
    """
        Train the model using Adam (via backpropagation)
        (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update the policy network at the end of episode
        3. evaluate the policy by running it for 10 episodes
    """
    
    # Instantiate the policy model
    if features_str == "Vanilla" or features_str == "RewardClip":
        agent = Agent(env, device, learning_rate=args.learning_rate, gamma=args.discount)
    elif features_str == "Double":
        agent = Agent(env, device, learning_rate=args.learning_rate, gamma=args.discount, double_=True)
    elif features_str == "Dueling":
        agent = Agent(env, device, learning_rate=args.learning_rate, gamma=args.discount, dueling=True)
    elif features_str == "PER":
        agent = Agent(env, device, learning_rate=args.learning_rate, gamma=args.discount, per=True)
    elif features_str == "Double_Dueling_PER":
        agent = Agent(env, device, learning_rate=args.learning_rate, gamma=args.discount, double_=True, dueling=True, per=True)
    
    # Instantiate the scheduler
    #actor_scheduler = Scheduler.StepLR(actor_optimizer, step_size=1000, gamma=args.discount)
    #lr_lambda = lambda epoch: 1 - min(epoch / 5000, 1.0)
    #actor_scheduler = Scheduler.LambdaLR(actor_optimizer, lr_lambda=lr_lambda)
        
    save_env_name = sanitize_env_name(args.env)
    episodes = 10000
    eval_interval = 50
    best_eval_returns = -np.inf
    max_steps = 10000
    update_freq = 4  # 每 4 步更新一次
    step_count = 0

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 200000
    warmup_steps = 50000
    epsilon = epsilon_start
	
    log_path = './logs/{}/{}_seed{}_eval_log.csv'.format(features_str, save_env_name, args.seed)
    csv_file = open(log_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "eval_return"])

    for i_episode in range(episodes + 1):
        # reset environment and episode reward
        state = reset_and_fire(env)
        ep_reward = 0
        t = 0
        losses = 0.0
        
        # For each episode, only run environment maximum steps to avoid entering infinite loop during the learning process
        for t in range(max_steps):
            if step_count > warmup_steps:
                epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (step_count - warmup_steps) / epsilon_decay)

            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated
            ep_reward += reward

            clipped_reward = np.clip(reward, -1.0, 1.0) if args.reward_clip else reward
            if args.per:
                agent.store_transition(state, action, clipped_reward, next_state, done)
            else:
                agent.buffer.insert(state, action, clipped_reward, next_state, done)
			
            if len(agent.buffer) >= warmup_steps and step_count % update_freq == 0: # 每 4 步更新一次
                loss = agent.update()
                losses += loss
				
            if step_count % 5000 == 0:
                agent.update_target_net()

            step_count += 1
            if done:
                break
            state = next_state
        
        # log the results
        print(f"Episode {i_episode}\tlength: {t}\treward: {ep_reward:.2f}\t loss: {losses:.4f}")

        # evaluate the policy every eval_interval episodes
        if i_episode % eval_interval == 0:
            eval_return = evaluate(agent, args.env, args.seed, n_episodes=10)
            print('Evaluation return: {:.2f}'.format(eval_return))
            csv_writer.writerow([i_episode, eval_return])

            if eval_return > best_eval_returns:
                torch.save(agent.evaluate_net.state_dict(), './outputs/{}/{}_Q{}.pth'.format(features_str, save_env_name, args.seed))
                best_eval_returns = eval_return
				
    csv_file.close()

if __name__ == "__main__":
	gym.register_envs(ale_py)
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default='ALE/Assault-v5')             # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)                 # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--learning_rate", default=2.5e-4, type=float) # Learning rate
	parser.add_argument("--discount", default=0.99, type=float)        # Discount factor
	parser.add_argument("--reward_clip", default=False, type=bool)     # Whether to clip the reward
	parser.add_argument("--double", default=False, type=bool)          # Whether to use double
	parser.add_argument("--dueling", default=False, type=bool)         # Whether to use dueling
	parser.add_argument("--per", default=False, type=bool)             # Whether to use PER
	args = parser.parse_args()

	enabled_features = []
	if args.reward_clip:
		enabled_features.append("RewardClip")
	if args.double:
		enabled_features.append("Double")
	if args.dueling:
		enabled_features.append("Dueling")
	if args.per:
		enabled_features.append("PER")

	features_str = "_".join(enabled_features) if enabled_features else "Vanilla"

	print("---------------------------------------")    
	print(f"Setting: Training DQN, Env: {args.env}, Seed: {args.seed}")
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
     
	env = gym.make(args.env, frameskip=1)
	env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4, terminal_on_life_loss=True)
	env = FrameStack(env, num_stack=4)

	env.reset(seed=args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train(args, device, env, features_str)