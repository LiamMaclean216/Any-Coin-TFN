import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
class ActorCritic(nn.Module):
    def __init__(self, params):
        super(ActorCritic, self).__init__()
        self.params = params
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(params['state_dim'], 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, params['action_dim']),
                #nn.Sigmoid()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(params['state_dim'], 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((params['action_dim'],), params['action_std']*params['action_std']).to(params['device'])
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.params['device'])
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = torch.sigmoid(dist.sample())
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.params['device'])
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
class PPO:
    def __init__(self, params):
        
        self.params = params
        
        self.policy = ActorCritic(params).to(self.params['device'])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=params['lr'])#, betas=betas)
        
        self.policy_old = ActorCritic(params).to(self.params['device'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(self.params['device'])
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.params['gamma'] * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.params['device'])
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.params['device']), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.params['device']), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.params['device']).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.params['K_epochs']):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()  
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.params['eps_clip'], 1+self.params['eps_clip']) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())