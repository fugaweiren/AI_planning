import torch
import torch.nn.functional as F

def get_kg_set(env_name=""):


    kg_set = [
        "go to the ball",
    ]



    return kg_set


actions = {
    "Left":0,
    "Right":1,
    "Forward":2
}

def get_expert_actions(obs, expert_rules, with_agents=False ,env_name=""):
    agent_pos = (obs.shape[-3]//2, obs.shape[-2]-1)
    
    if len(obs.shape) == 5 and with_agents:
        # obs => 
        expert_actions = torch.zeros((obs.shape[0], obs.shape[1], len(expert_rules), len(actions)), device='cuda:0')
        img = obs[:,:,:,:,0]
    elif  len(obs.shape) == 4 and with_agents:
        expert_actions = torch.zeros((obs.shape[0], obs.shape[1], len(expert_rules), len(actions)), device='cuda:0')
        img = obs[:,:,:,0]
    else:
        expert_actions = torch.zeros((obs.shape[0], len(expert_rules), len(actions)), device='cuda:0')
        img = obs[:,:,:,0]  # batch x 7 x 7x 6

    # colors = obs.image[:,:,:,1]
    # states = obs.image[:,:,:,2]


    def convert_pos_to_dir_actions(rule_id, start, goals, ids_to_remove=None):
        if with_agents:

            for agent_idx in range(obs.shape[1]):

                goal = goals[:, agent_idx].nonzero()

                img_id_meet_condition = goal[:,0][(start[0] < goal[:, 1]).int().nonzero()]
                expert_actions[img_id_meet_condition, agent_idx, rule_id,actions["Right"]] = 1
                
                img_id_meet_condition = goal[:,0][(start[0] > goal[:, 1]).int().nonzero()]
                expert_actions[img_id_meet_condition,agent_idx, rule_id,actions["Left"]] = 1
                
                img_id_meet_condition = goal[:,0][(start[1] > goal[:, 2]).int().nonzero()]
                expert_actions[img_id_meet_condition, agent_idx ,rule_id,actions["Forward"]] = 1

                if ids_to_remove is not None:
                    expert_actions[ids_to_remove ,rule_id,actions["Right"]] = 0
                    expert_actions[ids_to_remove,rule_id,actions["Left"]] = 0
                    expert_actions[ids_to_remove,rule_id,actions["Forward"]] = 0

        else:
            
            img_id_meet_condition = goals[:,0][(start[0] < goals[:, 1]).int().nonzero()]
            expert_actions[img_id_meet_condition,rule_id,actions["Right"]] = 1
            
            img_id_meet_condition = goals[:,0][(start[0] > goals[:, 1]).int().nonzero()]
            expert_actions[img_id_meet_condition,rule_id,actions["Left"]] = 1
            
            img_id_meet_condition = goals[:,0][(start[1] > goals[:, 2]).int().nonzero()]
            expert_actions[img_id_meet_condition,rule_id,actions["Forward"]] = 1

            if ids_to_remove is not None:
                expert_actions[ids_to_remove ,rule_id,actions["Right"]] = 0
                expert_actions[ids_to_remove,rule_id,actions["Left"]] = 0
                expert_actions[ids_to_remove,rule_id,actions["Forward"]] = 0

    if "go to the ball" in expert_rules:
        rule_id = expert_rules.index("go to the ball")
        goal_pos = (img == 6).nonzero() if not with_agents else (img == 6)
        convert_pos_to_dir_actions(rule_id, agent_pos, goal_pos)


    return expert_actions # (b x # rules x # actions) or  # (b x # num_agents x # rules x # actions) 


def expert_behaviors_by_env(env_name='all'):
    expert_rules = get_kg_set(env_name)
    def expert_behaviors(obs):
        expert_actions = get_expert_actions(obs, expert_rules, env_name=env_name)# (b x # rules x # actions) or (# rules x # actions)
        x = torch.sum(expert_actions, dim=1) #(b x # actions)
        return F.softmax(x, dim=1)
    return lambda x: expert_behaviors(x)
