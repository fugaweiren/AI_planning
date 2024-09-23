class KnowledgeGraph:
    def __init__(self):
        self.agent_position = (1,1)
        self.goal_position = (7,7)
        self.KG = nx.DiGraph()
        self.env = None
    all_policies = {
        # Move to Goal
        'move_towards_goal' : lambda agent_pos, obs: 'move_forward' if obs['front_clear'] else None,
        'turn_left_if_blocked' : lambda agent_pos, obs: 'move_left' if not obs['front_clear'] and not obs['right_clear'] else None,
        'turn_right_if_blocked' : lambda agent_pos, obs: 'move_right' if not obs['front_clear'] and not obs['left_clear'] else None,
        # Key
        #'pickup_key': lambda agent_pos, obs: 'pickup' if obs['key_nearby'] else None,
        'open_door' : lambda agent_pos, obs: 'unlock' if obs['door_nearby'] else None,
        'find_key' : lambda agent_pos, obs: 'move_to_key' if obs['key_nearby'] else None,
    }
    obs = {
        'front_clear' : False,
        'right_clear' : False,
        'left_clear' : False,
        'key_nearby' : False,
        'door_nearby' : False
    }

    def set_agent_position(self, position, front_clear=False, left_clear = False, right_clear = False):
        self.agent_position = position
        self.obs['front_clear'] = front_clear
        self.obs['right_clear'] = right_clear
        self.obs['left_clear'] = left_clear
        
        self.KG.add_node('agent', position = position, front_clear = front_clear, left_clear = left_clear, right_clear = right_clear)
 
    def set_goal_position(self, position):
        self.KG.add_node('goal', position = position)
        self.KG.add_edge('agent', 'goal', action='move_towards_goal')

    def set_obstacle(self, position):
        self.KG.add_node('obstacle', position = position, type='Wall')
        self.KG.add_edge('agent', 'obstacle', relation = 'avoid')
        
    def show_agent_position(self):
        print(self.agent_position)

    def apply_policies(self):
        for policy_name, policy_function in self.all_policies.items():
            action = policy_function(self.agent_position, self.obs)
           # if action == 'move_forward':
           #     return env.actions.forward 
           # elif action == 'move_left':
           #     return env.actions.left
           # elif action == 'move_right':
           #     return env.actions.right
        return "move_forward"

    def update_knowledge_graph(env, obs):
        self.env = env
        self.agent_position = self.env.agent_pos
        self.obs = obs
        
        if self.obs.get('key_nearby'):
            self.KG.add_node('key', position=obs['key_position'])
            self.add_edge(self.agent_position, 'key', relation='collect')
            
        