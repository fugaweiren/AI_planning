import torch
import torch.nn.functional as F


actions = {
    "Left":0,
    "Right":1,
    "Forward":2,
}
def get_kg_set(set=""):
    if set == 'ball only':                      # ball Environment for 2 agents
        kg_set = [
            "go to the ball",
            ]

    elif set == "ball with search strats":      # Right wall hugging
        kg_set = [
            "go to the ball",
            "turn left when detect front wall",
            "move forward if right wall detected",
            "turn left when detect front wall and right wall"
            ]
    
    elif set == "key only":             # Key Environment for 2 agents
        kg_set = [
        "go to the key",
        "go to the door",
        ]

    elif set == "ball + key":           # Key Environment
            kg_set = [
            "go to the ball",
            "go to the key",
            "go to the door",
            ]
    
    elif set =='lava + ball':           # Lava Environment
        kg_set = [
            "go to the ball",
            "do not hit lava",
            ]
        
    elif set == "conflicting rules":    # Right wall hugging & Left wall hugging
        kg_set = [
            "go to the ball",
            "turn left when detect front wall",
            "turn left when detect front wall and right wall",
            "move forward if right wall detected",
            "turn right when detect front wall",
            "turn right when detect front wall and right wall"
            "move forward if left wall detected",
            ]


    elif set == "irrelevant rules":                 # Try on simple
        kg_set = [
            "go to the ball",
            "go to the key",
            "go to the door",
            "do not hit lava",
            "turn left when detect front wall",
            ]                       

    else:
        kg_set = [
            "go to the ball",
            "go to the key",
            "go to the door",
            "do not hit lava",
            "turn left when detect front wall",
            "move forward if between two wall",
            "do not hit lava",
            "turn left when detect front wall",
            "turn left when detect front wall and right wall",
            "move forward if between two wall"]
    return kg_set

def get_expert_actions(obs, expert_rules, with_agents=False , env_name=""):
    agent_pos = (obs.shape[1]//2, obs.shape[2]-1)
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
    # colors = obs[:,:,:,1]
    # states = obs[:,:,:,2]



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


        else:
            
            img_id_meet_condition = goals[:,0][(start[0] < goals[:, 1]).int().nonzero()]
            expert_actions[img_id_meet_condition,rule_id,actions["Right"]] = 1
            
            img_id_meet_condition = goals[:,0][(start[0] > goals[:, 1]).int().nonzero()]
            expert_actions[img_id_meet_condition,rule_id,actions["Left"]] = 1
            
            img_id_meet_condition = goals[:,0][(start[1] > goals[:, 2]).int().nonzero()]
            expert_actions[img_id_meet_condition,rule_id,actions["Forward"]] = 1

       
    def prevent_actions(rule_id, start, goals):
        if with_agents:
            for agent_idx in range(obs.shape[1]):

                expert_actions[:, agent_idx, rule_id,actions["Right"]] = 1
                expert_actions[:, agent_idx, rule_id,actions["Left"]] = 1
                expert_actions[:, agent_idx, rule_id,actions["Forward"]] = 1
                
                img_id_meet_condition = goals[:,0][torch.logical_and(start[0] - goals[:,1] == -1, start[1] == goals[:,2]).int().nonzero()]
                expert_actions[img_id_meet_condition, agent_idx, rule_id, actions["Right"]] = 0
                
                img_id_meet_condition = goals[:,0][torch.logical_and(start[0] - goals[:,1] == 1,  start[1] == goals[:,2]).int().nonzero()]
                expert_actions[img_id_meet_condition, agent_idx, rule_id, actions["Left"]] = 0
                
                img_id_meet_condition = goals[:,0][torch.logical_and(start[1] - goals[:,2] == 1,  start[0] == goals[:,1]).int().nonzero()]
                expert_actions[img_id_meet_condition, agent_idx, rule_id, actions["Forward"]] = 0
        
        else:
                expert_actions[:,rule_id,actions["Right"]] = 1
                expert_actions[:,rule_id,actions["Left"]] = 1
                expert_actions[:,rule_id,actions["Forward"]] = 1
                
                img_id_meet_condition = goals[:,0][torch.logical_and(start[0] - goals[:,1] == -1, start[1] == goals[:,2]).int().nonzero()]
                expert_actions[img_id_meet_condition,rule_id,actions["Right"]] = 0
                
                img_id_meet_condition = goals[:,0][torch.logical_and(start[0] - goals[:,1] == 1,  start[1] == goals[:,2]).int().nonzero()]
                expert_actions[img_id_meet_condition,rule_id,actions["Left"]] = 0
                
                img_id_meet_condition = goals[:,0][torch.logical_and(start[1] - goals[:,2] == 1,  start[0] == goals[:,1]).int().nonzero()]
                expert_actions[img_id_meet_condition,rule_id,actions["Forward"]] = 0
    
    if "go to the ball" in expert_rules:
        rule_id = expert_rules.index("go to the ball")
        goal_pos = (img == 6).nonzero()
        convert_pos_to_dir_actions(rule_id, agent_pos, goal_pos)
    
    if "go to the key" in expert_rules:
        rule_id = expert_rules.index("go to the key")
        key_pos = (img == 5).nonzero()
        convert_pos_to_dir_actions(rule_id, agent_pos, key_pos)
        
    if "go to the door" in expert_rules:
        rule_id = expert_rules.index("go to the door")
        door_pos = (img == 4).nonzero()
        convert_pos_to_dir_actions(rule_id, agent_pos, door_pos)
    

    if "do not hit lava" in expert_rules:
        rule_id = expert_rules.index("do not hit lava")
        prevent_pos = (img == 9).nonzero()
        prevent_actions(rule_id, agent_pos, prevent_pos)

    if "do not hit wall" in expert_rules:
        rule_id = expert_rules.index("do not hit wall")
        prevent_pos = (img == 2).nonzero()
        prevent_actions(rule_id, agent_pos, prevent_pos)

    # if "do not hit lava and wall" in expert_rules:
    #     rule_id = expert_rules.index("do not hit lava and wall")
    #     prevent_pos = torch.logical_or(img == 2,img ==9).nonzero()
    #     prevent_actions(rule_id, agent_pos, prevent_pos)
    
    if "turn left when detect front wall" in expert_rules:
        rule_id = expert_rules.index("turn left when detect front wall")
        wall_pos = (img==2).nonzero()
        img_id_meet_condition = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]).int() * (wall_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        
        if with_agents:
            for agent_idx in range(obs.shape[1]):
                expert_actions[img_id_meet_condition, agent_idx, rule_id, actions["Left"]] = 1
        else:
            expert_actions[img_id_meet_condition,rule_id,actions["Left"]] = 1

    if "turn right when detect front wall" in expert_rules:
        rule_id = expert_rules.index("turn right when detect front wall")
        wall_pos = (img==2).nonzero()
        img_id_meet_condition = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]).int() * (wall_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        
        if with_agents:
            for agent_idx in range(obs.shape[1]):
                expert_actions[img_id_meet_condition, agent_idx, rule_id, actions["Right"]] = 1
        else:
            expert_actions[img_id_meet_condition,rule_id,actions["Right"]] = 1
    
    if "turn left when detect front wall and right wall" in expert_rules:
        # wall_pos[:,1] => If same row as agents, means it is in the forward dircetion of agent, (Left = -1, Right = +1)
        # wall_pos[:,2] => If same column as agents, means it is in the left(-1)/right(+1) dircetion of agent,(Foward = -1, Behind = +1)
        rule_id = expert_rules.index("turn left when detect front wall and right wall")
        img_id_meet_condition_front = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]).int() * (wall_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        img_id_meet_condition_right = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]+1).int() * (wall_pos[:,2] == agent_pos[1]).int()).nonzero()]
        right_corner = img_id_meet_condition_front[torch.isin(img_id_meet_condition_front, img_id_meet_condition_right)]
        
        
        if with_agents:
            for agent_idx in range(obs.shape[1]):
                expert_actions[right_corner, agent_idx,rule_id, actions["Left"]] = 1
        else:
            expert_actions[right_corner, rule_id, actions["Left"]] = 1

    if "turn right when detect front wall and left wall" in expert_rules:
        # wall_pos[:,1] => If same row as agents, means it is in the forward dircetion of agent, (Left = -1, Right = +1)
        # wall_pos[:,2] => If same column as agents, means it is in the left(-1)/right(+1) dircetion of agent,(Foward = -1, Behind = +1)
        rule_id = expert_rules.index("turn right when detect front wall and left wall")
        img_id_meet_condition_front = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]).int() * (wall_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        img_id_meet_condition_left = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]-1).int() * (wall_pos[:,2] == agent_pos[1]).int()).nonzero()]
        right_corner = img_id_meet_condition_front[torch.isin(img_id_meet_condition_front, img_id_meet_condition_left)]
        
        
        if with_agents:
            for agent_idx in range(obs.shape[1]):
                expert_actions[right_corner,agent_idx, rule_id, actions["Right"]] = 1
        else:
            expert_actions[right_corner, rule_id, actions["Right"]] = 1


    if "move forward if between two wall" in expert_rules:
        rule_id = expert_rules.index("move forward if between two wall")
        wall_pos = (img==2).nonzero()
        img_id_meet_conditionA = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]+1).int() * (wall_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        img_id_meet_conditionB = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]-1).int() * (wall_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        between_wall = img_id_meet_conditionA[torch.isin(img_id_meet_conditionA, img_id_meet_conditionB)]
        expert_actions[between_wall,rule_id,actions["Forward"]] = 1

        if with_agents:
            for agent_idx in range(obs.shape[1]):
                expert_actions[right_corner, agent_idx,rule_id, actions["Forward"]] = 1
        else:
            expert_actions[right_corner, rule_id, actions["Forward"]] = 1

        
    if "avoid corridor" in expert_rules:
        rule_id = expert_rules.index("avoid corridor")
        wall_pos = (img==2).nonzero()
        img_id_meet_conditionA = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]).int() * (wall_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        img_id_meet_conditionB = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]-1).int() * (wall_pos[:,2] == agent_pos[1]).int()).nonzero()]
        right_corridor = img_id_meet_conditionA[torch.isin(img_id_meet_conditionA, img_id_meet_conditionB)]
        
        img_id_meet_conditionC = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]+1).int() * (wall_pos[:,2] == agent_pos[1]).int()).nonzero()]
        left_corridor = img_id_meet_conditionA[torch.isin(img_id_meet_conditionA, img_id_meet_conditionC)]
        
        
        if with_agents:
            for agent_idx in range(obs.shape[1]):
                expert_actions[right_corridor,agent_idx, rule_id,actions["Left"]] = 1
                expert_actions[left_corridor, agent_idx, rule_id,actions["Right"]] = 1
        else:
            expert_actions[right_corridor,rule_id,actions["Left"]] = 1
            expert_actions[left_corridor,rule_id,actions["Right"]] = 1

    if "move forward if right wall detected" in expert_rules:
        rule_id = expert_rules.index("move forward if right wall detected")
        wall_pos = (img==2).nonzero()
        img_id_meet_condition = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]+1).int() * (wall_pos[:,2] == agent_pos[1]).int()).nonzero()]
        
        if with_agents:
            for agent_idx in range(obs.shape[1]):
                expert_actions[img_id_meet_condition, agent_idx, rule_id, actions["Forward"]] = 1
        else:
            expert_actions[img_id_meet_condition,rule_id,actions["Forward"]] = 1


    if "move forward if left wall detected" in expert_rules:
        rule_id = expert_rules.index("move forward if left wall detected")
        wall_pos = (img==2).nonzero()
        img_id_meet_condition = wall_pos[:,0][((wall_pos[:,1] == agent_pos[0]-1).int() * (wall_pos[:,2] == agent_pos[1]).int()).nonzero()]
        
        if with_agents:
            for agent_idx in range(obs.shape[1]):
                expert_actions[img_id_meet_condition, agent_idx, rule_id, actions["Forward"]] = 1
        else:
            expert_actions[img_id_meet_condition,rule_id,actions["Forward"]] = 1
    return expert_actions # (b x # rules x # actions)


def expert_behaviors_by_env(env_name='all'):
    expert_rules = get_kg_set(env_name)
    def expert_behaviors(obs):
        expert_actions = get_expert_actions(obs, expert_rules, env_name=env_name)# (b x # rules x # actions) or (# rules x # actions)
        print(expert_actions)
        x = torch.sum(expert_actions, dim=1) #(b x # actions)
        return F.softmax(x, dim=1)
    return lambda x: expert_behaviors(x)
