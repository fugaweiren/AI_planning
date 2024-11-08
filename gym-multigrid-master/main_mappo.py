from torch.distributions.categorical import Categorical
from gym.wrappers import RecordEpisodeStatistics
from gym.vector import SyncVectorEnv
import gym
from gym.envs.registration import register

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import tqdm
import threading
from plot import live_plot, plot_final_results
from gym_multigrid.multigrid import MAX_STEPS
from config import ENV_CLASS, ENV_RULE_SETS
# from external_knowledge import get_expert_actions, get_kg_set
from modified_kg import get_expert_actions, get_kg_set
import argparse
import os
from os.path import join, dirname
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="simple",
                    help="name of the environment (REQUIRED): simple, lava, key")
parser.add_argument("--use_kg", action="store_true", default=True,
                    help="userules")
parser.add_argument("--kg_set", default=2, type=int,
                    help="Ruleset option")
parser.add_argument("--result_dir",  default=join(dirname(os.path.abspath(__file__)), "results/mappo"), type=str,
                    help="Ruleset")
parser.add_argument("--steps", default=1000, type=int,
                    help="Ruleset")
args = parser.parse_args()

env_entrypt = ENV_CLASS[args.env]
USE_KG = args.use_kg
ruleset= ENV_RULE_SETS[args.env][args.kg_set]
scenario = "multigrid-collect-v0"

register(
    id=scenario,
    entry_point=env_entrypt,
)
env = gym.make(scenario)

NUM_ENVS = 1
# envs = SyncVectorEnv([lambda:  gym.make('multigrid-collect-v0') for _ in range(NUM_ENVS)])
NUM_AGENTS = len(env.agents)

envs = env
LEARNING_RATE = 2.5e-4

ROLLOUT_STEPS = MAX_STEPS
NUM_MINI_BATCHES = NUM_EPOCHS = 4
# TOTAL_STEPS = 800000
TOTAL_STEPS = args.steps

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_COEF = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
# RANDOM SEED, DON'T MODIFY
SEED = 2048
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

states = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, NUM_AGENTS) + envs.observation_space.shape).to(device)
actions = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, NUM_AGENTS) + envs.action_space.shape).to(device)
rewards = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, NUM_AGENTS)).to(device)
unnormalized_rewards = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, NUM_AGENTS)).to(device)
dones = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, NUM_AGENTS)).to(device)

logprobs = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, NUM_AGENTS)).to(device)
values = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize the weights and biases of a layer.

    Args:
        layer (nn.Module): The layer to initialize.
        std (float): Standard deviation for orthogonal initialization.
        bias_const (float): Constant value for bias initialization.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)  # Orthogonal initialization
    torch.nn.init.constant_(layer.bias, bias_const)  # Constant bias
    return layer

class ACAgent(nn.Module):
    """Actor-Critic agent using neural networks for policy and value function approximation."""

    def __init__(self):
        """Initialize the Actor-Critic agent with actor and critic networks."""
        super().__init__()

        ### ------------- TASK 1.1 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        actor_input_dim = 7*7*6  # Input dimension for the actor (state dim)
        actor_output_dim = 3  # Output dimension for the actor (number of actions)
        critic_input_dim = NUM_AGENTS*7*7*6   # Input dimension for the critic
        critic_output_dim = 1  # Output dimension for the critic (value estimate)
        ### ------ YOUR CODES END HERE ------- ###

        # Define the actor network
        if USE_KG:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(actor_input_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 64)),
            )
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(actor_input_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, actor_output_dim), std=0.01),  # Final layer with small std for output
            )

        # Define the critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, critic_output_dim), std=1.0),  # Standard output layer for value
        )

        kg_emb_dim = 8
        # self.kg_set = get_kg_set("ball with search strats")
        self.kg_set = get_kg_set(ruleset)
        # self.kg_set = get_kg_set("conflicting rules")


        # Models for KG integration

        self.actor_Q = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, kg_emb_dim)
        )
        self.actor_K = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, kg_emb_dim)
        )
        self.actor_V = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, actor_output_dim)
        )


        # Define expert rules (ours)
        self.expert_K = nn.Embedding(len(self.kg_set), kg_emb_dim)


    def get_value(self, x):
        """Calculate the estimated value for a given state.

        Args:
            x (torch.Tensor): Input state, shape: (batch_size, observation_size)

        Returns:
            torch.Tensor: Estimated value for the state, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.2 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        x = x.flatten(start_dim=-4)
        value = self.critic(x) # Forward pass through the critic network
        ### ------ YOUR CODES END HERE ------- ###
        return value

    def get_probs(self, x):
        """Calculate the action probabilities for a given state.

        Args:
            x (torch.Tensor): Input state, shape: (batch_size, observation_size)

        Returns:
            torch.distributions.Categorical: Categorical distribution over actions.
        """
        ### ------------- TASK 1.3 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        
        if not USE_KG:
            x = x.flatten(start_dim=-3)
            logits = self.actor(x)  # Get logits from the actor network
        else:
            if len(x.shape) == 4:
                x = x.unsqueeze(0) # Add batch dimension
                batch, num_agents, _,_, _ =  x.shape
            else:
                batch, num_agents, _,_,_ =  x.shape
            

            # Find the expert's actions
            expert_actions = get_expert_actions(x, self.kg_set, with_agents=True)# (b x # rules x # actions) or (# rules x # actions)
            num_rules, num_actions = expert_actions.shape[-2:]

            # expert_actions = expert_actions.expand(batch, num_agents, num_rules, num_actions)

            x = x.flatten(start_dim=-3)         # (batch, num_agents, 7*7*6)
            x = self.actor(x)                   # (batch, num_agents, 64)

            # actor_Q(x): rules encoder                 # (batch, num_agents, 8)
            # expert_K.weight: attention weights        # (#rules, 8)
            # expert_actions: Values                    # (batch, num_agents, num_rules, num_actions)


            # expert_Q_norm, expert_Q_norm.unsqueeze(-1)        #(batch, num_agents, ), (batch, num_agents, 1)
            # expert_Kw_norm, expert_Kw_norm.unsqueeze(0)      #(#rules,), (1,#rules)
            # expert_K_norm                                     # ( b x num_agents x # rules x 1)

            expert_K_norm = torch.matmul(torch.linalg.norm(self.actor_Q(x), dim=-1).unsqueeze(-1), torch.linalg.norm(self.expert_K.weight, dim=1).unsqueeze(0)).unsqueeze(-1)# ( b x num_agents x # rules x 1)
            expert_actions = F.softmax(expert_K_norm * expert_actions, dim=-1) # softmax at num_actions's dim

            ## Compute weights
            self.w_i = F.cosine_similarity(self.actor_Q(x), self.actor_K(x), dim=-1).view(batch, num_agents, 1,1) #.view(-1,1,1)# (bx num_agents x1x1) , Self-attention
            self.W_e = F.cosine_similarity(self.expert_K.weight, self.actor_Q(x).unsqueeze(-2), dim=-1).unsqueeze(-1)# (b x num_agents x # rules x 1), Expert Cross-Attention
            
            # torch.sum(torch.exp(self.W_e), dim=1, keepdim=True)       # (b x num_agents x 1 x 1)

            total_exp_w = torch.exp(self.w_i) + torch.sum(torch.exp(self.W_e), dim=-2, keepdim=True)  # (b x num_agents x 1 x 1)
            
            self.w_i = torch.exp(self.w_i) / total_exp_w        # (b x num_agents x 1 x 1)
            self.W_e = torch.exp(self.W_e) / total_exp_w        # (b x num_agents x 1 x 1)
            
            inner_pi = F.softmax(self.actor_V(x), dim=-1)         # (b x num_agents x num_actions)
            logits = self.w_i.view(batch, num_agents, 1) * inner_pi + torch.sum(self.W_e * expert_actions, dim=-2) #(bx num_agents x # rules x 1) x (b x num_agents x # rules x # actions)
            logits = torch.log(logits)
            logits = torch.softmax(logits, dim=-1)


        probs = Categorical(logits=logits)  # Create a categorical distribution from the logits
        ### ------ YOUR CODES END HERE ------- ###
        return probs

    def get_action(self, probs):
        """Sample an action from the action probabilities.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Sampled action, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.4 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        action = probs.sample()  # Sample an action based on the probabilities
        ### ------ YOUR CODES END HERE ------- ###
        return action

    def get_action_logprob(self, probs, action):
        """Compute the log probability of a given action.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.
            action (torch.Tensor): Selected action, shape: (batch_size, 1)

        Returns:
            torch.Tensor: Log probability of the action, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.5 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        logprob = probs.log_prob(action)  # Calculate log probability of the sampled action
        ### ------ YOUR CODES END HERE ------- ###
        return logprob

    def get_entropy(self, probs):
        """Calculate the entropy of the action distribution.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Entropy of the distribution, shape: (batch_size, 1)
        """
        return probs.entropy()  # Return the entropy of the probabilities

    def get_action_logprob_entropy(self, x):
        """Get action, log probability, and entropy for a given state.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            tuple: (action, logprob, entropy)
                - action (torch.Tensor): Sampled action.
                - logprob (torch.Tensor): Log probability of the action.
                - entropy (torch.Tensor): Entropy of the action distribution.
        """
        probs = self.get_probs(x)  # Get the action probabilities
        action = self.get_action(probs)  # Sample an action   #KG action=> 1x4 batch x agents
        logprob = self.get_action_logprob(probs, action)  # Compute log probability of the action
        entropy = self.get_entropy(probs)  # Compute entropy of the action distribution
        return action, logprob, entropy  # Return action, log probability, and entropy

BATCH_SIZE = ROLLOUT_STEPS * NUM_ENVS
MINI_BATCH_SIZE = BATCH_SIZE // NUM_MINI_BATCHES
NUM_ITERATIONS = TOTAL_STEPS // BATCH_SIZE

def get_deltas(rewards, values, next_values, next_nonterminal, gamma):
    """Compute the temporal difference (TD) error.

    Args:
        rewards (torch.Tensor): Rewards at each time step, shape: (batch_size,).
        values (torch.Tensor): Predicted values for each state, shape: (batch_size,).
        next_values (torch.Tensor): Predicted value for the next state, shape: (batch_size,).
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Computed TD errors, shape: (batch_size,).
    """
    ### -------------- TASK 2 ------------ ###
    ### ----- YOUR CODES START HERE ------ ###
    deltas = rewards + gamma*next_values*next_nonterminal - values
    ### ------ YOUR CODES END HERE ------- ###
    return deltas

def get_ratio(logprob, logprob_old):
    """Compute the probability ratio between the new and old policies.

    This function calculates the ratio of the probabilities of actions under
    the current policy compared to the old policy, using their logarithmic values.

    Args:
        logprob (torch.Tensor): Log probability of the action under the current policy,
                                shape: (batch_size,).
        logprob_old (torch.Tensor): Log probability of the action under the old policy,
                                    shape: (batch_size,).

    Returns:
        torch.Tensor: The probability ratio of the new policy to the old policy,
                      shape: (batch_size,).
    """
    ### ------------ TASK 3.1.1 ---------- ###
    ### ----- YOUR CODES START HERE ------ ###
    logratio = logprob - logprob_old  # Compute the log ratio
    ratio = torch.exp(logratio)  # Exponentiate to get the probability ratio
    ### ------ YOUR CODES END HERE ------- ###
    return ratio

def get_policy_objective(advantages, ratio, clip_coeff=CLIP_COEF):
    """Compute the clipped surrogate policy objective.

    This function calculates the policy objective using the advantages and the
    probability ratio, applying clipping to stabilize training.

    Args:
        advantages (torch.Tensor): The advantage estimates, shape: (batch_size,).
        ratio (torch.Tensor): The probability ratio of the new policy to the old policy,
                             shape: (batch_size,).
        clip_coeff (float, optional): The clipping coefficient for the policy objective.
                                       Defaults to CLIP_COEF.

    Returns:
        torch.Tensor: The computed policy objective, a scalar value.
    """
    ### ------------ TASK 3.1.2 ---------- ###
    ### ----- YOUR CODES START HERE ------ ###
    policy_objective1 = ratio*advantages   # Calculate the first policy loss term
    policy_objective2 = torch.clamp(ratio, min=(1-clip_coeff), max=(1+clip_coeff))*advantages  # Calculate the clipped policy loss term
    policy_objective = torch.minimum(policy_objective1, policy_objective2).mean(dim=(0,1))  # Take the minimum and average over the batch
    ### ------ YOUR CODES END HERE ------- ###
    # policy_objective = policy_objective1.mean(dim=(0,1))
    return policy_objective

def get_value_loss(values, values_old, returns):
    """Compute the combined value loss with clipping.

    This function calculates the unclipped and clipped value losses
    and returns the maximum of the two to stabilize training.

    Args:
        values (torch.Tensor): Predicted values from the critic, shape: (batch_size, 1).
        values_old (torch.Tensor): Old predicted values from the critic, shape: (batch_size, 1).
        returns (torch.Tensor): Computed returns for the corresponding states, shape: (batch_size, 1).

    Returns:
        torch.Tensor: The combined value loss, a scalar value.
    """
    ### ------------- TASK 3.2 ----------- ###
    ### ----- YOUR CODES START HERE ------ ###

    values = values.unsqueeze(1).expand(-1, returns.shape[1])
    values_old = values_old.unsqueeze(1).expand(-1, returns.shape[1])

    value_loss_unclipped = 0.5*(values - returns)**2  # Calculate unclipped value loss

    value_loss_clipped = 0.5*((values_old + torch.clamp(values-values_old, min=-CLIP_COEF, max=CLIP_COEF)-returns))**2   # Calculate clipped value loss

    value_loss = torch.max(value_loss_unclipped,value_loss_clipped).mean(dim=(0,1)) # Average over the batch, Number of Agents
    ### ------ YOUR CODES END HERE ------- ###
    return value_loss  # Return the final combined value loss

def get_entropy_objective(entropy):
    """Compute the entropy objective.

    This function calculates the average entropy of the action distribution,
    which encourages exploration by penalizing certainty.

    Args:
        entropy (torch.Tensor): Entropy values for the action distribution, shape: (batch_size,).

    Returns:
        torch.Tensor: The computed entropy objective, a scalar value.
    """
    return entropy.mean()  # Return the average entropy

def get_total_loss(policy_objective, value_loss, entropy_objective, value_loss_coeff=VALUE_LOSS_COEF, entropy_coeff=ENTROPY_COEF):
    """Compute the total loss for the actor-critic agent.

    This function combines the policy objective, value loss, and entropy objective
    into a single loss value for optimization. It applies coefficients to scale
    the contribution of the value loss and entropy objective.

    Args:
        policy_objective (torch.Tensor): The policy objective, a scalar value.
        value_loss (torch.Tensor): The computed value loss, a scalar value.
        entropy_objective (torch.Tensor): The computed entropy objective, a scalar value.
        value_loss_coeff (float, optional): Coefficient for scaling the value loss. Defaults to VALUE_LOSS_COEF.
        entropy_coeff (float, optional): Coefficient for scaling the entropy loss. Defaults to ENTROPY_COEF.

    Returns:
        torch.Tensor: The total computed loss, a scalar value.
    """
    ### ------------- TASK 3.3 ----------- ###
    ### ----- YOUR CODES START HERE ------ ###
    total_loss = -policy_objective + value_loss_coeff*value_loss - entropy_coeff*entropy_objective  # Combine losses
    ### ------ YOUR CODES END HERE ------- ###
    return total_loss




ENABLE_LIVE_PLOT= False
ENABLE_LIVE_ENV_RENDER= False
TRAIN = False

# Initialize global step counter and reset the environment
global_step = 0
initial_state, _ = envs.reset()
state = torch.Tensor(initial_state).to(device)
done = torch.zeros(NUM_ENVS).to(device)

if TRAIN:
    agent = ACAgent().to(device)
    max_reward, min_reward = env.balls_reward[0], (-0.1-1)*MAX_STEPS 
    # Set up progress tracking
    progress_bar = tqdm.tqdm(range(1, NUM_ITERATIONS + 1), postfix={'Total Rewards': 0})
    actor_loss_history = []
    critic_loss_history = []
    entropy_objective_history = []

    reward_history = []
    episode_history = []

    n_agents = env.agents.__len__()
    data_to_plot = {}
    for agent_idx in range(n_agents):
        # data_to_plot[f'Agent {agent_idx} Actor Loss'] = []
        # data_to_plot[f'Agent {agent_idx} Critic Loss'] = []
        data_to_plot[f'Agent {agent_idx} Reward'] = []

    data_to_plot['Total Reward'] =  reward_history
    data_to_plot['Shared Actor Policy Loss'] =  actor_loss_history
    data_to_plot['Shared Critic Loss'] =  critic_loss_history
    data_to_plot['Entropy Loss'] =  entropy_objective_history
    plot_update_counter = [0]

    if ENABLE_LIVE_PLOT:
        # Start live_plot in a separate thread
        plot_thread = threading.Thread(target=live_plot, args=(data_to_plot,plot_update_counter))
        plot_thread.daemon = True  # Ensures the thread will exit when the main program ends
        plot_thread.start()

    # Initialize the optimizer for the agent's parameters
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    for iteration in progress_bar:

        # Adjust the learning rate using a linear decay
        fraction_completed = 1.0 - (iteration - 1.0) / NUM_ITERATIONS
        current_learning_rate = fraction_completed * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = current_learning_rate

        # Perform rollout to gather experience
        for step in range(0, ROLLOUT_STEPS):
            if ENABLE_LIVE_ENV_RENDER:
                envs.render(mode='human', highlight=True)
                assert envs.step_count == step
            global_step += NUM_ENVS
            states[step] = state
            dones[step] = done

            with torch.no_grad():
                # Get action, log probability, and entropy from the agent
                action, log_probability, _ = agent.get_action_logprob_entropy(state)
                value = agent.get_value(state)
                values[step] = value.flatten()

            action = action if not USE_KG else action[0]
            actions[step] = action 
            logprobs[step] = log_probability if not USE_KG else log_probability[0]
            
            # Execute action in the environment
            next_state, reward, done, info = envs.step(action.cpu().numpy())
            normalized_reward = (reward - min_reward) / (max_reward - min_reward)  # Normalize the reward
            rewards[step] = torch.tensor(normalized_reward).to(device).view(-1)
            unnormalized_rewards[step] = torch.tensor(reward).to(device).view(-1)
            state = torch.Tensor(next_state).to(device)
            done = torch.Tensor([done]).to(device)

            if step == ROLLOUT_STEPS-1:
                done = torch.Tensor([True]).to(device)
            if done.item() == True:
                episodic_reward = unnormalized_rewards.sum().item()
                reward_history.append(episodic_reward)
                episode_history.append(global_step)
                progress_bar.set_postfix({'Total Rewards': episodic_reward})
                resert_state, _ = env.reset()
                state =  torch.Tensor(resert_state).to(device)
                break
                
            for agent_idx, r in zip(range(n_agents),reward):
                data_to_plot[f'Agent {agent_idx} Reward'].append(r)

        # Calculate advantages and returns
        with torch.no_grad():
            next_value = agent.get_value(state).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)

            last_gae_lambda = 0
            for t in reversed(range(ROLLOUT_STEPS)):
                if t == ROLLOUT_STEPS - 1:
                    next_non_terminal = 1.0 - done
                    next_value = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]

                # Compute delta using the utility function
                delta = get_deltas(rewards[t], values[t], next_value, next_non_terminal, gamma=GAMMA)
                advantages[t] = last_gae_lambda = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lambda
            returns = advantages + values

        # Flatten the batch data for processing
        batch_states = states.reshape((-1,  NUM_AGENTS) + envs.observation_space.shape)
        batch_logprobs = logprobs.reshape(-1, NUM_AGENTS)
        batch_actions = actions.reshape((-1, NUM_AGENTS) + envs.action_space.shape)
        batch_advantages = advantages.reshape(-1, NUM_AGENTS)
        batch_returns = returns.reshape(-1, NUM_AGENTS)
        batch_values = values.reshape(-1)

        # Shuffle the batch data to break correlation between samples
        batch_indices = np.arange(BATCH_SIZE)
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_objective = 0

        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(batch_indices)
            for start in range(0, BATCH_SIZE, MINI_BATCH_SIZE):
                # Get the indices for the mini-batch
                end = start + MINI_BATCH_SIZE
                mini_batch_indices = batch_indices[start:end]

                mini_batch_advantages = batch_advantages[mini_batch_indices]
                # Normalize advantages to stabilize training
                mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (mini_batch_advantages.std() + 1e-8)

                # Compute new probabilities and values for the mini-batch
                new_probabilities = agent.get_probs(batch_states[mini_batch_indices])
                new_log_probability = agent.get_action_logprob(new_probabilities, batch_actions.long()[mini_batch_indices])
                entropy = agent.get_entropy(new_probabilities)
                new_value = agent.get_value(batch_states[mini_batch_indices])

                # Calculate the policy loss
                ratio = get_ratio(new_log_probability, batch_logprobs[mini_batch_indices])
                policy_objective = get_policy_objective(mini_batch_advantages, ratio, clip_coeff=CLIP_COEF)
                policy_loss = -policy_objective

                # Calculate the value loss
                value_loss = get_value_loss(new_value.view(-1), batch_values[mini_batch_indices], batch_returns[mini_batch_indices])

                # Calculate the entropy loss
                entropy_objective = get_entropy_objective(entropy)

                # Combine losses to get the total loss
                total_loss = get_total_loss(policy_objective, value_loss, entropy_objective, value_loss_coeff=VALUE_LOSS_COEF, entropy_coeff=ENTROPY_COEF)

                optimizer.zero_grad()
                total_loss.backward()
                # Clip the gradient to stabilize training
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                total_actor_loss += policy_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy_objective += entropy_objective.item()

        actor_loss_history.append(total_actor_loss // NUM_EPOCHS)
        critic_loss_history.append(total_critic_loss // NUM_EPOCHS)
        entropy_objective_history.append(total_entropy_objective // NUM_EPOCHS)

        
        data_to_plot['Total Reward'] =  reward_history
        data_to_plot['Shared Actor Policy Loss'] =  actor_loss_history
        data_to_plot['Shared Critic Loss'] =  critic_loss_history
        data_to_plot['Entropy Loss'] =  entropy_objective_history
        plot_update_counter[0]+=1

        # Clear previous episode rollouts
        states = torch.zeros_like(states).to(device)
        actions = torch.zeros_like(actions).to(device)
        rewards = torch.zeros_like(rewards).to(device)
        unnormalized_rewards = torch.zeros_like(unnormalized_rewards).to(device)
        logprobs = torch.zeros_like(logprobs).to(device)
        values = torch.zeros_like(values).to(device)
    
    KG_STR = f"_USE_KG_{ruleset}" if USE_KG else ""

    FOLDER_NAME = f"steps_{TOTAL_STEPS}_ngames_{len(data_to_plot['Total Reward'])}{KG_STR}"
    ENV_FOLDER = join(args.result_dir, args.env)
    
    
    results_directory = join(ENV_FOLDER,FOLDER_NAME)
    os.makedirs(results_directory, exist_ok=True)
    print(f"Written to {results_directory}")

    
    torch.save(agent.state_dict(), join(results_directory,"mappo_agent_model.pth"))

    with open(join(results_directory,"experiment1.pickle"), 'wb') as handle:
        pickle.dump(data_to_plot, handle)

    plot_final_results(data_to_plot, save_path= join(results_directory,"final_plot.png"))
    
    with open(join(results_directory,"logs"), "w") as f:
        f.write(f"Training takes {progress_bar.format_dict['elapsed']}")


else:
    # For qualitative
    initial_state, _ = envs.reset()
    state = torch.Tensor(initial_state).to(device)
    shared_agent = ACAgent().to(device)
    # state_dict = torch.load("gym-multigrid-master/results/mappo/no_clip_epoch_800K_noKG/mappo_agent_model.pth")
    state_dict = torch.load("mappo_agent_model.pth")
    shared_agent.load_state_dict(state_dict)
    done = False
    progress_bar = tqdm.tqdm(range(1, NUM_ITERATIONS + 1))
    for iteration in progress_bar:
        envs.render(mode='human', highlight=True)
        with torch.no_grad():
            # Get action, log probability, and entropy from the agent
            action, _, _ = shared_agent.get_action_logprob_entropy(state)
            action = action[0] if len(action.shape) ==2 else action
            next_state, reward, done, info = envs.step(action.cpu().numpy())
            state = torch.Tensor(next_state).to(device)
        if done:
            initial_state, _ = envs.reset()
            state = torch.Tensor(initial_state).to(device)
    