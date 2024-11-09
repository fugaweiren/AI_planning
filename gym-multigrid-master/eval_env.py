import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from modified_kg import get_expert_actions

import gym
from gym.envs.registration import register
import tqdm

from plot import live_plot, plot_final_results
from gym_multigrid.multigrid import MAX_STEPS
from config import ENV_CLASS, ENV_RULE_SETS

from modified_kg import get_expert_actions, get_kg_set
import argparse
import os
from os.path import join, dirname
import pickle
import numpy as np
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="simple",
                    help="name of the environment (REQUIRED): simple, lava, key")
parser.add_argument("--use_kg", action="store_true", default=False,
                    help="userules")
parser.add_argument("--kg_set", default=0, type=int,
                    help="Ruleset option")
parser.add_argument("--load_model_path",  default="expert", type=str,
                    help="model path, but for COMA: Model directories")
parser.add_argument("--model_type", default="expert", type=str,
                    help="expert, MAPPO, COMA, random")
parser.add_argument("--result_dir",  default=join(dirname(os.path.abspath(__file__)), "results_eval"), type=str,
                    help="Ruleset")
parser.add_argument("--viz_dir",  default=join(dirname(os.path.abspath(__file__)), "results_eval_viz"), type=str,
                    help="Ruleset")
parser.add_argument("--steps", default=1000, type=int,
                    help="eval_steps")
parser.add_argument("--vision_dim", default=7, type=int,
                    help="Ruleset")
args = parser.parse_args()

env_entrypt = ENV_CLASS[args.env]
ruleset= ENV_RULE_SETS[args.env][args.kg_set]
scenario = "multigrid-collect-v0"

if args.kg_set == "lava2":
    args.vision_dim = 5

register(
    id=scenario,
    entry_point=env_entrypt,
)
env = gym.make(scenario)
NUM_ENVS = 1
NUM_AGENTS = len(env.agents)
USE_KG = args.use_kg
TOTAL_STEPS = args.steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.kg_set = get_kg_set(ruleset)


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

    def get_probs(self, x):

        
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
            expert_actions = get_expert_actions(x, self.kg_set, with_agents=True)

            x = x.flatten(start_dim=-3)
            x = self.actor(x)
            expert_K_norm = torch.matmul(torch.linalg.norm(self.actor_Q(x), dim=-1).unsqueeze(-1), torch.linalg.norm(self.expert_K.weight, dim=1).unsqueeze(0)).unsqueeze(-1)# ( b x num_agents x # rules x 1)
            expert_actions = F.softmax(expert_K_norm * expert_actions, dim=-1)

            ## Compute weights
            self.w_i = F.cosine_similarity(self.actor_Q(x), self.actor_K(x), dim=-1).view(batch, num_agents, 1,1)
            self.W_e = F.cosine_similarity(self.expert_K.weight, self.actor_Q(x).unsqueeze(-2), dim=-1).unsqueeze(-1)
            

            total_exp_w = torch.exp(self.w_i) + torch.sum(torch.exp(self.W_e), dim=-2, keepdim=True)  
            
            self.w_i = torch.exp(self.w_i) / total_exp_w       
            self.W_e = torch.exp(self.W_e) / total_exp_w       
            
            inner_pi = F.softmax(self.actor_V(x), dim=-1)
            logits = self.w_i.view(batch, num_agents, 1) * inner_pi + torch.sum(self.W_e * expert_actions, dim=-2)
            logits = torch.log(logits)
            logits = torch.softmax(logits, dim=-1)


        probs = Categorical(logits=logits)
        return probs
    
    def get_actions(self, x):
        with torch.no_grad():
            actions = self.get_probs(x).sample()
          

        return actions


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        if USE_KG:
            self.fc1 = nn.Linear(state_dim, 64)
            self.fc2 = nn.Linear(64, 64)
        else:
            self.fc1 = nn.Linear(state_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, action_dim)

        kg_emb_dim = 8
        self.kg_set = get_kg_set(set=ruleset)

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
            nn.Linear(64, action_dim)
        )

        self.expert_K = nn.Embedding(len(self.kg_set), kg_emb_dim)

    def forward(self, x):
        if USE_KG:
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            expert_actions = get_expert_actions(x, self.kg_set)
            num_rules, num_actions = expert_actions.shape[-2:]

            x = x.flatten(start_dim=-3)
            x = F.relu(self.fc2(F.relu(self.fc1(x))))

            expert_K_norm = torch.matmul(torch.linalg.norm(self.actor_Q(x), dim=-1).unsqueeze(-1), torch.linalg.norm(self.expert_K.weight, dim=1).unsqueeze(0)).unsqueeze(-1)
            expert_actions = F.softmax(expert_K_norm * expert_actions, dim=-1)

            self.w_i = F.cosine_similarity(self.actor_Q(x), self.actor_K(x), dim=-1).view(-1, 1, 1)
            self.W_e = F.cosine_similarity(self.expert_K.weight, self.actor_Q(x).unsqueeze(-2), dim=-1).unsqueeze(-1)

            total_exp_w = torch.exp(self.w_i) + torch.sum(torch.exp(self.W_e), dim=-2, keepdim=True)
            self.w_i = torch.exp(self.w_i) / total_exp_w
            self.W_e = torch.exp(self.W_e) / total_exp_w

            inner_pi = F.softmax(self.actor_V(x), dim=-1)
            logits = self.w_i.view(1, 1) * inner_pi + torch.sum(self.W_e * expert_actions, dim=-2)
            logits = torch.log(logits)

            dist = Categorical(logits=logits)
            return torch.softmax(logits, dim=-1), dist
        else:
            x = x.view(-1, self.fc1.in_features)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            logits = self.fc3(x)
            return F.softmax(logits, dim=-1), Categorical(logits=logits)

class COMAAgentWrapper:
    def __init__(self, model_dir):
        state_dim = 7*7*6 
        action_dim = 3
        self.actors =[]
        for file in os.listdir(model_dir):
            if (args.env=='simple' and "actor" in file and 'irrelevant' and '20000' in file) or (args.env =='simple' and "actor" in file and "30000" in file) or (args.env == "key" and "actor" in file and "10000" in file) or (args.env == "lava" and USE_KG == False and "actor" in file and "10000" in file)\
                or (args.env == "lava" and USE_KG == True and "actor" in file and "30000" in file):
                actor = Actor(state_dim, action_dim).to(device)
                actor.load_state_dict(torch.load(join(model_dir, file)))
                self.actors.append(actor)
        assert len(self.actors) == NUM_AGENTS
    
    def get_actions(self, obs):
        actions = []
        for i in range(NUM_AGENTS):
            with torch.no_grad():
                pi, probs = self.actors[i](obs[i])
                action = probs.sample()
                actions.append(action)
        
        return torch.Tensor(actions).to(device)

class RandomPolicy:
    def get_actions(self, obs):
        return torch.randint(low=0, high=3, size=(NUM_AGENTS,)).to(device)

class Expert:
    def __init__(self):
        self.kg_set = get_kg_set(ruleset)
    def get_actions(self, obs):
        expert_actions = get_expert_actions(obs, self.kg_set)
        x = torch.sum(expert_actions, dim=-2)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        return dist.sample().to(device)


initial_state, _ = env.reset()
state = torch.Tensor(initial_state)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if USE_KG:
    if args.model_type == "expert":
        shared_agent = Expert()
    elif args.model_type == "MAPPO":
        shared_agent = ACAgent().to(device)
        shared_agent.load_state_dict(torch.load(args.load_model_path))
    elif args.model_type == "COMA":
        shared_agent = COMAAgentWrapper(args.load_model_path)
    else:
        print("UNKNOWN model type")
        assert False
else:
    if args.model_type == "MAPPO":
        shared_agent = ACAgent().to(device)
        shared_agent.load_state_dict(torch.load(args.load_model_path))
    elif args.model_type == "COMA":
        shared_agent = COMAAgentWrapper(args.load_model_path)
    else:
        #print("UNKNOWN model type")
        #assert False
        shared_agent = RandomPolicy()

done = False
progress_bar = tqdm.tqdm(range(1, TOTAL_STEPS + 1))

data_to_plot = {}
for agent_idx in range(NUM_AGENTS):
    data_to_plot[f'Agent {agent_idx} Reward'] = []

data_to_plot['Total Reward'] = []
data_to_plot['Num Agents died'] = [] 
data_to_plot['Num Wall Hits'] = []
SAVE_FRAMES = 1
ENABLE_LIVE_ENV_RENDER = True
initial_state, _ = env.reset()
rewards = torch.zeros((MAX_STEPS, NUM_AGENTS)).to(device)
step = 0
frames = []
state = torch.tensor(initial_state,dtype=torch.float).to(device)
for iteration in progress_bar:
    if ENABLE_LIVE_ENV_RENDER:
        img = env.render(mode='human', highlight=True)
        if SAVE_FRAMES:
            frames.append(img)
    
    with torch.no_grad():
        action = shared_agent.get_actions(state)
        action = action[0] if len(action.shape) ==2 else action
        next_state, reward, done, info = env.step(action.cpu().numpy())
        state = torch.tensor(next_state, dtype=torch.float).to(device)
        if step == MAX_STEPS-1:
            done = True
        rewards[step] = torch.tensor(reward).to(device)
    

    if done:
        initial_state, info = env.reset()
        state = torch.tensor(initial_state, dtype=torch.float).to(device)

        SAVE_FRAMES =0
        episodic_reward = rewards.sum().item()
        progress_bar.set_postfix({'Total Rewards': episodic_reward})
        
        for agent_idx in range(NUM_AGENTS):
            data_to_plot[f'Agent {agent_idx} Reward'].append(rewards[:,agent_idx].sum().item())
        data_to_plot['Total Reward'].append(episodic_reward)
        data_to_plot['Num Agents died'].append(info["number of agents died:"])
        data_to_plot['Num Wall Hits'].append(info["amount of times agents hit wall:"])
        step =0
        rewards = torch.zeros_like(rewards)
    
    step +=1

KG_STR = f"_USEKG_{ruleset}" if USE_KG else ""
FOLDER_NAME = f"steps_{TOTAL_STEPS}_ngames_{len(data_to_plot['Total Reward'])}{KG_STR}"
ENV_FOLDER = join(args.result_dir, args.env, args.model_type)
results_directory = join(ENV_FOLDER,FOLDER_NAME)

os.makedirs(results_directory, exist_ok=True)

with open(join(results_directory,f"{args.env}_{args.model_type}_{FOLDER_NAME}_eval.pickle"), 'wb') as handle:
    pickle.dump(data_to_plot, handle)

plot_final_results(data_to_plot, save_path= join(results_directory,"eval_plot.png"))

with open(join(results_directory,"eval_stats"), "w+") as f:
    for agent_idx in range(NUM_AGENTS):
        f.write(f"Mean_reward for Agent {agent_idx}: {np.mean(data_to_plot[f'Agent {agent_idx} Reward'])}\n")
    f.write(f"Mean_reward per episode: {np.mean(data_to_plot['Total Reward'])}\n")
    f.write(f"Mean deaths per episode: {np.mean(data_to_plot['Num Agents died'])}")
    f.write(f"Mean wall hits per episode: {np.mean(data_to_plot['Num Wall Hits'])}")


gif_filename = join(results_directory, f'{args.env}_{args.model_type}_{FOLDER_NAME}_animation.gif')
imageio.mimsave(gif_filename, frames, fps=15)  # Adjust fps as needed

ENV_FOLDER = join(args.viz_dir, args.env)
os.makedirs(ENV_FOLDER, exist_ok=True)
with open(join(ENV_FOLDER, f"{args.env}_{args.model_type}_{FOLDER_NAME}_eval.pickle"), 'wb') as handle:
    pickle.dump(data_to_plot, handle)

print(f"Written to {results_directory}")