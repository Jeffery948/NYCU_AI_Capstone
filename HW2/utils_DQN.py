import gymnasium as gym
import ale_py

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack

class replay_buffer():
    '''
    A deque storing trajectories
    '''
    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer
        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish        
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.
        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left = 2 * parent_index + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left]:
                    parent_index = left
                else:
                    v -= self.tree[left]
                    parent_index = right
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.alpha = 0.5
        self.beta = 0.4  # importance-sampling exponent
        self.beta_increment_per_sampling = 0.001
        self.epsilon = 0.01  # small value to avoid zero priority
        self.capacity = capacity

    def __len__(self):
        return self.tree.n_entries

    def add(self, error, sample):
        clip_error = np.minimum(np.abs(error) + self.epsilon, 1.0)  # Clip the error to avoid too large priority
        priority = clip_error ** self.alpha
        self.tree.add(priority, sample)

    def sample(self, batch_size):
        minibatch = []
        idxs = []
        segment = self.tree.total_priority() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get_leaf(s)
            priorities.append(priority)
            minibatch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total_priority()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= (is_weights.max() + 1e-6)

        return minibatch, idxs, is_weights

    def update(self, idx, error):
        clip_error = np.minimum(np.abs(error) + self.epsilon, 1.0)  # Clip the error to avoid too large priority
        priority = clip_error ** self.alpha
        self.tree.update(idx, priority)
        # Update the priority of the sample at index idx

class QNet(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state
    '''
    def __init__(self, env, dueling=False):
        super(QNet, self).__init__()
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.dueling = dueling

        if not self.dueling:
            self.layer = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, self.action_dim)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
            )
            self.advantage = nn.Linear(512, self.action_dim)
            self.value = nn.Linear(512, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.layer.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                normc_initializer(m.weight, gain=1.0)
                torch.nn.init.zeros_(m.bias)
        if self.dueling:
            for m in self.advantage.modules():
                if isinstance(m, torch.nn.Linear):
                    normc_initializer(m.weight, gain=1.0)
                    torch.nn.init.zeros_(m.bias)
            for m in self.value.modules():
                if isinstance(m, torch.nn.Linear):
                    normc_initializer(m.weight, gain=1.0)
                    torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        """
            Forward pass of the network
            - The input is the state, and the outputs are the corresponding 
              state value
        """
        x = state / 255.0 if state.dtype == torch.uint8 else state
        if self.dueling:
            x = self.layer(x)
            advantage = self.advantage(x)
            value = self.value(x)
            x = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            x = self.layer(x)

        return x

class Agent():
    def __init__(self, env, device, learning_rate=2.5e-4, gamma=0.99, batch_size=32, capacity=200000, double_=False, dueling=False, per=False):
        """
        The agent class for DQN
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            gamma: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.env = env
        self.device = device

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.capacity = capacity
        self.double_ = double_
        self.dueling = dueling
        self.per = per

        self.buffer = replay_buffer(self.capacity) if self.per == False else PrioritizedReplayBuffer(self.capacity)  # the replay buffer
        self.evaluate_net = QNet(self.env, self.dueling).to(self.device)  # the evaluate network
        self.target_net = QNet(self.env, self.dueling).to(self.device)  # the target network

        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=self.learning_rate)

    def select_action(self, state, epsilon=0.05):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the max Q-value)
        """
        with torch.no_grad():
            state = np.array(state)
            state = torch.from_numpy(state).to(self.device).float().unsqueeze(0)
            action = torch.argmax(self.evaluate_net(state)).item() # Choose best action by Q-table
            if np.random.rand() < epsilon: # Decide whether to explore
                action = self.env.action_space.sample()

        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
            Store the transition in the replay buffer
            - The input is the state, action, reward, next_state and done status
        """
        state_tensor = torch.tensor(np.array(state)).unsqueeze(0).to(self.device).float()
        next_state_tensor = torch.tensor(np.array(next_state)).unsqueeze(0).to(self.device).float()

        q_val = self.evaluate_net(state_tensor)[0][action]

        if done:
            q_target = torch.tensor(reward).to(self.device)
        else:
            if self.double_:
                next_action = self.evaluate_net(next_state_tensor).argmax()
                q_target_val = self.target_net(next_state_tensor)[0][next_action]
            else:
                q_target_val = self.target_net(next_state_tensor).max()
            q_target = reward + self.gamma * q_target_val

        # TD error as priority
        error = abs(q_val - q_target).detach().cpu().numpy()  # Convert to numpy for PER
        self.buffer.add(error, (state, action, reward, next_state, done))
    
    def update(self):
        if self.per:
            # Sample with PER: get data, indices, and importance sampling weights
            samples, indices, is_weights = self.buffer.sample(self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = zip(*samples)

            # Convert to torch tensors
            state_batch = torch.tensor(np.array(state_batch)).to(self.device).float()
            action_batch = torch.tensor(np.array(action_batch)).reshape(self.batch_size, 1).to(self.device).long()
            reward_batch = torch.tensor(np.array(reward_batch)).reshape(self.batch_size, 1).to(self.device).float()
            next_state_batch = torch.tensor(np.array(next_state_batch)).to(self.device).float()
            mask_batch = torch.tensor(np.array(mask_batch)).reshape(self.batch_size, 1).to(self.device).float()
            is_weights = torch.tensor(np.array(is_weights)).reshape(self.batch_size, 1).to(self.device).float()
        else:
            # Sample uniformly
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.buffer.sample(self.batch_size)
            state_batch = torch.tensor(np.array(state_batch)).to(self.device).float()
            action_batch = torch.tensor(np.array(action_batch)).reshape(self.batch_size, 1).to(self.device).long()
            reward_batch = torch.tensor(np.array(reward_batch)).reshape(self.batch_size, 1).to(self.device).float()
            next_state_batch = torch.tensor(np.array(next_state_batch)).to(self.device).float()
            mask_batch = torch.tensor(np.array(mask_batch)).reshape(self.batch_size, 1).to(self.device).float()

        # Forward the data to the evaluate net and the target net and make some modifications.
        q_eval = self.evaluate_net(state_batch).gather(1, action_batch)

        if self.double_:
            # Double DQN: use eval net to select action, but target net to evaluate it
            next_q_eval = self.evaluate_net(next_state_batch)
            next_actions = next_q_eval.argmax(dim=1, keepdim=True)  # shape: [batch_size, 1]
            next_q_target = self.target_net(next_state_batch).gather(1, next_actions)
        else:
            # Vanilla DQN: directly take max over target Q
            next_q_target = self.target_net(next_state_batch).max(dim=1, keepdim=True)[0]

        # I use 1 - mask to make sure if episode is done, it don't take additional Q-value
        q_target = reward_batch + self.gamma * next_q_target * (1 - mask_batch)

        # Detaching to prevent backprop through target net
        td_error = (q_eval - q_target).detach().cpu().squeeze().numpy()  

        # Compute loss
        if self.per:
            loss = (is_weights * (q_eval - q_target).pow(2)).mean()
        else:
            loss = F.mse_loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.per:
            for idx, err in zip(indices, td_error):
                self.buffer.update(idx, np.abs(err))

        return loss.item()

    def update_target_net(self):
        """
            Update the target network with the evaluate network
        """
        self.target_net.load_state_dict(self.evaluate_net.state_dict())

def evaluate(agent, env_name, seed, n_episodes=10):
    """
        Test the learned model
    """     
    gym.register_envs(ale_py)
    eval_env = gym.make(env_name, frameskip=1)
    eval_env = AtariPreprocessing(eval_env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    eval_env = FrameStack(eval_env, num_stack=4)
        
    eval_env.reset(seed=seed)
    eval_env.action_space.seed(seed)

    agent.evaluate_net.eval()
    total_return = 0
    max_steps = eval_env.spec.max_episode_steps if eval_env.spec.max_episode_steps is not None else 1500
    
    with torch.no_grad():
        for i_episode in range(n_episodes):
            state = reset_and_fire(eval_env) if env_name.startswith("ALE/") else eval_env.reset()[0]
            ep_reward = 0
            for t in range(max_steps):
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
                if done:
                    break
                state = next_state
            total_return += ep_reward

    agent.evaluate_net.train()
    return total_return / n_episodes

def sanitize_env_name(env_name):
    return env_name.replace("/", "_")

def normc_initializer(weight_tensor, gain=1.0):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L97

    Note that in tensorflow the weight tensor in a linear layer is stored with the
    input dim first and the output dim second. See
    https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/layers/core.py#L1193

    In contrast, in pytorch the output dim is first. See
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

    This means if we want a normc init in pytorch,
    we have to change which dim(s) we normalize on.

    We currently only support normc init for linear layers.
    Performance not guaranteed with other layer weight types.
    """
    with torch.no_grad():
        out = np.random.normal(loc=0.0, scale=1.0, size=weight_tensor.size())
        out = gain * out / np.sqrt(np.sum(np.square(out), axis=1, keepdims=True))
        weight_tensor.copy_(torch.tensor(out))

def reset_and_fire(env):
    """
    Reset Atari env and send a FIRE action to start the game
    """
    state, _ = env.reset()
    
    # Try to FIRE to start the game (if applicable)
    try:
        state, _, terminated, truncated, _ = env.step(1)  # FIRE = 1
        if terminated or truncated:
            state, _ = env.reset()  # 避免直接 GG
    except Exception as e:
        pass  # 如果該環境不支援 FIRE，忽略即可
    
    return state