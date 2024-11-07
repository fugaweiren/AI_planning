import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
# from utils import DictToListWrapper, live_plot
from gym.wrappers import RecordEpisodeStatistics
from gym.vector import SyncVectorEnv
import gym
from gym.envs.registration import register
import tqdm
from plot import live_plot
import random
import time
import threading

# from external_knowledge import get_expert_actions, get_kg_set

from modified_kg import get_expert_actions, get_kg_set


scenario = "multigrid-collect-v0"

register(
    id=scenario,
    entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
    # entry_point='gym_multigrid.envs:CollectGame1HEnv10x10',
)
env = gym.make('multigrid-collect-v0')

# NUM_ENVS = 4
NUM_ENVS = 1
# envs = SyncVectorEnv([lambda:  gym.make('multigrid-collect-v0') for _ in range(NUM_ENVS)])
NUM_AGENTS = 4
USE_KG = True
STATE_DIM = 7*7*6
ACTION_DIM = 3
envs = env
LR_A = 0.0001
LR_C = 0.005

target_update_steps = 10

NUM_MINI_BATCHES = NUM_EPOCHS = 4
TOTAL_STEPS = 800000

GAMMA = 0.99

# RANDOM SEED, DON'T MODIFY
SEED = 2048
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Memory:
    def __init__(self, agent_num, action_dim):
        self.agent_num = agent_num
        self.action_dim = action_dim

        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(agent_num)]
        self.reward = []
        self.done = [[] for _ in range(agent_num)]

    def get(self):
        actions = torch.tensor(self.actions).to(device)
        observations = self.observations

        pi = []
        for i in range(self.agent_num):
            pi.append(torch.cat(self.pi[i]).view(len(self.pi[i]), self.action_dim))

        reward = torch.tensor(self.reward).to(device)
        done = self.done

        return actions, observations, pi, reward, done

    def clear(self):
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(self.agent_num)]
        self.reward = []
        self.done = [[] for _ in range(self.agent_num)]


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
        self.kg_set = get_kg_set(set="ball with search strats")

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
            nn.Linear(64, action_dim)
        )

        # Define expert rules (ours)
        self.expert_K = nn.Embedding(len(self.kg_set), kg_emb_dim)
    def forward(self, x):
        if USE_KG:
            #if len(x.shape) == 4:
            #    x = x.unsqueeze(0) # Add batch dimension
            #    batch, num_agents, _,_, _ =  x.shape
            #else:
            #    batch, num_agents, _,_,_ =  x.shape

            # Find the expert's actions
            if len(x.shape) == 3:
                x = x.unsqueeze(0) # Add batch dimension
            expert_actions = get_expert_actions(x, self.kg_set)# (b x # rules x # actions) or (# rules x # actions)
            num_rules, num_actions = expert_actions.shape[-2:]

            # expert_actions = expert_actions.expand(batch, num_agents, num_rules, num_actions)

            x = x.flatten(start_dim=-3)         # (batch, 7*7*6)
            # x = x.type(torch.FloatTensor)
            x = F.relu(self.fc2(F.relu(self.fc1(x)))) # (batch, 64)

            # actor_Q(x): rules encoder                 # (batch, num_agents, 8)
            # expert_K.weight: attention weights        # (#rules, 8)
            # expert_actions: Values                    # (batch, num_agents, num_rules, num_actions)


            # expert_Q_norm, expert_Q_norm.unsqueeze(-1)        #(batch, num_agents, ), (batch, num_agents, 1)
            # expert_Kw_norm, expert_Kw_norm.unsqueeze(0)      #(#rules,), (1,#rules)
            # expert_K_norm                                     # ( b x num_agents x # rules x 1)

            expert_K_norm = torch.matmul(torch.linalg.norm(self.actor_Q(x), dim=-1).unsqueeze(-1), torch.linalg.norm(self.expert_K.weight, dim=1).unsqueeze(0)).unsqueeze(-1)# ( b x num_agents x # rules x 1)
            expert_actions = F.softmax(expert_K_norm * expert_actions, dim=-1) # softmax at num_actions's dim

            ## Compute weights
            self.w_i = F.cosine_similarity(self.actor_Q(x), self.actor_K(x), dim=-1).view(-1,1,1) #.view(-1,1,1)# (bx num_agents x1x1) , Self-attention
            self.W_e = F.cosine_similarity(self.expert_K.weight, self.actor_Q(x).unsqueeze(-2), dim=-1).unsqueeze(-1)# (b x num_agents x # rules x 1), Expert Cross-Attention
            
            # torch.sum(torch.exp(self.W_e), dim=1, keepdim=True)       # (b x num_agents x 1 x 1)

            total_exp_w = torch.exp(self.w_i) + torch.sum(torch.exp(self.W_e), dim=-2, keepdim=True)  # (b x num_agents x 1 x 1)
            
            self.w_i = torch.exp(self.w_i) / total_exp_w        # (b x num_agents x 1 x 1)
            self.W_e = torch.exp(self.W_e) / total_exp_w        # (b x num_agents x 1 x 1)
            
            inner_pi = F.softmax(self.actor_V(x), dim=-1)         # (b x num_agents x num_actions)
            logits = self.w_i.view(1, 1) * inner_pi + torch.sum(self.W_e * expert_actions, dim=-2) #(bx num_agents x # rules x 1) x (b x num_agents x # rules x # actions)
            # logits = logits
            
            dist = Categorical(logits=logits)

            return torch.softmax(logits, dim=-1), dist
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            logits = self.fc3(x)
            return F.softmax(logits, dim=-1),  Categorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super(Critic, self).__init__()

        input_dim = 1 + state_dim * agent_num + agent_num

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class COMA:
    def __init__(self, agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps):
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma

        self.target_update_steps = target_update_steps

        self.memory = Memory(agent_num, action_dim)

        self.actors = [Actor(state_dim, action_dim).to(device) for _ in range(agent_num)]
        self.critic = Critic(agent_num, state_dim, action_dim).to(device) 

        self.critic_target = Critic(agent_num, state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_a) for i in range(agent_num)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.count = 0
    
    def get_probs(self, results):
        
        return results

    def get_actions(self, observations):
        # observations = torch.tensor(observations)
        actions = []

        for i in range(self.agent_num):
            pi, probs = self.get_probs(self.actors[i](observations[i]))
            action = probs.sample()

            self.memory.pi[i].append(pi)
            actions.append(action)

        self.memory.observations.append(observations)
        self.memory.actions.append(actions)

        return actions

    def train(self):
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer

        actions, observations, pi, reward, done = self.memory.get()


        info = {}
        for i in range(self.agent_num):

            info[i] = {}
            info[i]["agent_loss"] = 0
            info[i]["critic_loss"] = 0

            # train actor

            input_critic = self.build_input_critic(i, observations, actions)
            Q_target = self.critic_target(input_critic).detach()

            action_taken = actions[:, i].reshape(-1, 1)

            baseline = torch.sum(pi[i] * Q_target, dim=1).detach()
            Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze()
            advantage = Q_taken_target - baseline

            log_pi = torch.log(torch.gather(pi[i], dim=1, index=action_taken).squeeze())

            actor_loss = - torch.mean(advantage * log_pi)

            actor_optimizer[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
            actor_optimizer[i].step()

            # train critic

            Q = self.critic(input_critic)

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()

            # TD(0)
            r = torch.zeros(len(reward[:, i])).to(device)
            for t in range(len(reward[:, i])):
                if done[i][t]:
                    r[t] = reward[:, i][t]
                else:
                    r[t] = reward[:, i][t] + self.gamma * Q_taken_target[t + 1]

            critic_loss = torch.mean((r - Q_taken) ** 2)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            critic_optimizer.step()


            info[i]["agent_loss"] = actor_loss.item()
            info[i]["critic_loss"] = critic_loss.item()

        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(agents.critic.state_dict())
            self.count = 0
        else:
            self.count += 1

        self.memory.clear()

        return info

    def build_input_critic(self, agent_id, observations, actions):
        batch_size = len(observations)

        ids = (torch.ones(batch_size) * agent_id).view(-1, 1).to(device)

        observations = torch.cat(observations).view(batch_size, self.state_dim * self.agent_num)
        # input_critic = torch.cat([observations.type(torch.float32), actions.type(torch.float32)], dim=-1)
        input_critic = torch.cat([observations.type(torch.float32).to(device), actions.type(torch.float32).to(device)], dim=-1)
        input_critic = torch.cat([ids, input_critic], dim=-1)

        return input_critic

agents = COMA(NUM_AGENTS, STATE_DIM, ACTION_DIM, LR_C, LR_A, GAMMA, target_update_steps)

obs,_ = envs.reset()

episode_reward = 0
episodes_reward = []

# training loop
ENABLE_LIVE_PLOT= False
ENABLE_LIVE_ENV_RENDER= True
N_GAMES = 10000
PRINT_INTERVAL = 500
episode = 0
step_per_episode = 0
best_score = 0

n_agents = env.agents.__len__()
#  Set up progress tracking
progress_bar = tqdm.tqdm(range(1, N_GAMES + 1), postfix={'Total Rewards': 0})
data_to_plot = {}
for agent_idx in range(n_agents):
    data_to_plot[f'Agent {agent_idx} Actor Loss'] = []
    data_to_plot[f'Agent {agent_idx} Critic Loss'] = []
    data_to_plot[f'Agent {agent_idx} Reward'] = []
data_to_plot["Total Reward"] =[]
plot_update_counter = [0]

if ENABLE_LIVE_PLOT:
    # # Start live_plot in a separate thread
    plot_thread = threading.Thread(target=live_plot, args=(data_to_plot,plot_update_counter))
    plot_thread.daemon = True  # Ensures the thread will exit when the main program ends
    plot_thread.start()



# while episode < n_episodes:
for i, iteration in enumerate(progress_bar):
    if ENABLE_LIVE_ENV_RENDER:
        # time.sleep(0.5)
        env.render(mode='human', highlight=True)
    
    obs = torch.tensor(obs, dtype=torch.float).to(device)
    actions = agents.get_actions(obs)
    next_obs, reward, done_n, _ = env.step(actions)
    agents.memory.reward.append(reward)
    for i in range(NUM_AGENTS):
        agents.memory.done[i].append(done_n)

    
    episode_reward += sum(reward)
    obs = next_obs
    
    for agent_idx in range(n_agents):
        data_to_plot[f'Agent {agent_idx} Reward'].append(reward[agent_idx])

    if step_per_episode == env.max_steps:
        done_n = True

    if done_n:
        progress_bar.set_postfix({'Total Rewards': episode_reward})
        episodes_reward.append(episode_reward)
        data_to_plot["Total Reward"] = episodes_reward
        
        step_per_episode = 0
        episode_reward = 0

        episode += 1

        obs, _ = env.reset()

        if episode % 10 == 0:
            info = agents.train()
            if info is not None:
                for agent_idx in range(n_agents):
                    data_to_plot[f'Agent {agent_idx} Actor Loss'].append(info[agent_idx]["agent_loss"])
                    data_to_plot[f'Agent {agent_idx} Critic Loss'].append(info[agent_idx]["critic_loss"])
                    

        if episode % 100 == 0:
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")

        plot_update_counter[0]+=1
    step_per_episode +=1

    
    # live_plot(data_to_plot, plot_update_counter)


import pickle
with open('experiment1.pickle', 'wb') as handle:
    pickle.dump(data_to_plot, handle)
