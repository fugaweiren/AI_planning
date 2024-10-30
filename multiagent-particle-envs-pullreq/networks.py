import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
# from .external_knowledge import get_expert_actions, get_kg_set


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims) # 249+4*8
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state.flatten(start_dim=1), action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()


        # Set Dimension
        action_space_n = 3
        kg_emb_dim = 8

        self.chkpt_file = os.path.join(chkpt_dir, name)
        
        # State Embedding (which is the actor base)
        self.fc1 = nn.Linear(input_dims, fc1_dims) # x64
        self.fc2 = nn.Linear(fc1_dims, fc2_dims) # x64

        # # Models for KG integration
        # self.actor_Q = nn.Sequential(
        #     # self.fc1, self.fc2,
        #     nn.Tanh(),
        #     nn.Linear(64, kg_emb_dim)
        # )
        # self.actor_K = nn.Sequential(
        #     # self.fc1, self.fc2,
        #     nn.Tanh(),
        #     nn.Linear(64, kg_emb_dim)
        # )
        # self.actor_V = nn.Sequential(
        #     # self.fc1, self.fc2,
        #     nn.Tanh(),
        #     nn.Linear(64, action_space_n)
        # )
        # # Define expert rules (ours)
        # self.expert_K = nn.Embedding(len(self.kg_set), kg_emb_dim)

        # # KG Set
        # self.kg_set = get_kg_set(kg_set_name)
        # print("Using the knowledge set: {}".format(kg_set_name))

        # MADDPG 
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state.flatten(start_dim=1)))
        x = F.relu(self.fc2(x)) #state repesentation

        # # Find the expert's actions
        # expert_actions = get_expert_actions(state, self.kg_set, env_name = self.env_name)# (b x # rules x # actions) or (# rules x # actions)


        # # actor_Q(x): rules encoder
        # # expert_K: attention weights
        # # expert_actions: Values
        # ## trick: scale the expert_actions to make its entropy learnable
        # expert_K_norm = T.matmul(T.linalg.norm(self.actor_Q(x), dim=1).unsqueeze(1), T.linalg.norm(self.expert_K.weight, dim=1).unsqueeze(0)).unsqueeze(2)# ( b x # rules x 1)
        # expert_actions = F.softmax(expert_K_norm * expert_actions, dim=2) 

        # ## Compute weights
        # self.w_i = F.cosine_similarity(self.actor_Q(x), self.actor_K(x), dim=1).view(-1,1,1)# (b) , Self-attention
        # self.W_e = F.cosine_similarity(self.expert_K.weight, self.actor_Q(x).unsqueeze(1), dim=2).unsqueeze(2)# (b x # rules), Expert Cross-Attention
        
        # total_exp_w = T.exp(self.w_i) + T.sum(T.exp(self.W_e), dim=1, keepdim=True)
        
        # self.w_i = T.exp(self.w_i) / total_exp_w
        # self.W_e = T.exp(self.W_e) / total_exp_w
        
        # inner_pi = F.softmax(self.actor_V(x),dim=1)
        # x = self.w_i.view(-1,1) * inner_pi + T.sum(self.W_e * expert_actions, dim=1) #(b x # rules x 1) x (b # rules x # actions)
        # pi = T.softmax(x, dim=1)
        ## sample dist, when doing pre-softmax
        # dist = Categorical(logits=T.log(x))

        # This is MADPPG
        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

