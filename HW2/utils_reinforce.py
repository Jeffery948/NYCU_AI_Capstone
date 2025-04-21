import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    """
        Implement the policy network
    """
    def __init__(self, env, hidden_size, gamma, device, reward_norm=False, baseline=False, GAE=False, gae_lambda=0.98, regularization=False):
        """
            Initialize the policy network
            - The input is the state, and the output is the action probability distirbution
        """
        super(Actor, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.device = device
        self.reward_norm = reward_norm
        self.baseline = baseline
        self.GAE = GAE
        self.gae_lambda = gae_lambda
        self.regularization = regularization
        self.beta = 0.01 if regularization else 0.0 # regularization coefficient

        self.layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        nn.init.kaiming_normal_(self.layer1.weight)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.kaiming_normal_(self.layer2.weight)
        self.layer3 = nn.Linear(self.hidden_size, self.action_dim)
        nn.init.kaiming_normal_(self.layer3.weight)

        # action & reward & entropy memory
        self.saved_actions = []
        self.rewards = []
        self.entropies = []

    def forward(self, state):
        """
            Forward pass of policy network
            - The input is the state, and the output is the corresponding 
              action probability distirbution
        """

        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        action_prob = F.softmax(self.layer3(x), dim = -1)

        return action_prob

    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
        """
        
        state = torch.tensor(state).to(self.device)
        action_prob = self.forward(state)

        if self.discrete:
            m = Categorical(action_prob)
            action = m.sample()
            self.saved_actions.append((m.log_prob(action), state))
            self.entropies.append(m.entropy())
            return action.item()
        else:
            # For continuous action space, use normal distribution
            mean = action_prob
            std = torch.ones_like(mean) * 0.1  # small std for exploration
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            action = torch.clamp(action, -1.0, 1.0) # clip action to valid range
            self.saved_actions.append((dist.log_prob(action).sum(), state))  # sum for multidim
            return action.detach().cpu().numpy()

    def calculate_loss(self, critic=None):
        """
            Calculate the loss to perform backprop later
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        returns = []

        # Calculate the returns by reversing the rewards
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(self.device).float()

        if critic is not None:
            states = torch.stack([s for (_, s) in saved_actions]).to(self.device)
            values = critic(states).squeeze(-1)
            if self.GAE:
                advantages = GAE(self.gamma, self.gae_lambda)(self.rewards, values.detach())
            else:
                advantages = returns - values.detach()

            if self.reward_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) if advantages.std() > 1e-6 else advantages
        else:
            if self.reward_norm:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8) if returns.std() > 1e-6 else returns
            advantages = returns

        # compute policy loss
        log_probs = torch.stack([log_prob for (log_prob, _) in saved_actions])
        policy_losses = -log_probs * advantages - self.beta * torch.stack(self.entropies)
        policy_loss = policy_losses.sum()

        # compute critic loss if applicable
        if critic is not None:
            critic_loss = F.mse_loss(values, returns)
        else:
            critic_loss = torch.tensor(0.0)

        return policy_loss, critic_loss

    def clear_memory(self):
        # reset rewards, action and entropy buffer
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropies[:]

class Critic(nn.Module):
    """
        Implement the critic network
    """
    def __init__(self, env, hidden_size, device):
        """
            Initialize the critic network
            - The input is the state, and the output is the state value
        """
        super(Critic, self).__init__()
        
        # Extract the dimensionality of state
        self.observation_dim = env.observation_space.shape[0]
        self.hidden_size = hidden_size
        self.device = device

        self.layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        nn.init.kaiming_normal_(self.layer1.weight)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.kaiming_normal_(self.layer2.weight)
        self.layer3 = nn.Linear(self.hidden_size, 1)
        nn.init.kaiming_normal_(self.layer3.weight)

    def forward(self, state):
        """
            Forward pass of critic network
            - The input is the state, and the output is the corresponding 
              state value
        """

        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        state_value = self.layer3(x)

        return state_value

class GAE:
    def __init__(self, gamma, lambda_):
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, rewards, values):
        """
            Calculate the Generalized Advantage Estimation and return the obtained value
        """

        advantages = []
        advantage = 0
        next_value = 0

        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + self.gamma * next_value - v
            advantage = td_error + advantage * self.gamma * self.lambda_
            next_value = v
            advantages.insert(0, advantage)

        advantages = torch.tensor(advantages).to(values.device).float()
        return advantages

def evaluate(agent, env_name, seed, n_episodes=10):
    """
        Test the learned model
    """    
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed)
    eval_env.action_space.seed(seed)

    agent.eval()
    total_return = 0
    max_steps = eval_env.spec.max_episode_steps if eval_env.spec.max_episode_steps is not None else 1500
    
    with torch.no_grad():
        for i_episode in range(n_episodes):
            state, _ = eval_env.reset()
            ep_reward = 0
            for t in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
                if done:
                    break
                state = next_state
            total_return += ep_reward

    agent.train()
    agent.clear_memory()
    return total_return / n_episodes