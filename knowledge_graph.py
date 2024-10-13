class KnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.env = None
    def initialize(self):
        self.G.add_node("agent", position=(1, 1), front_block=True, left_block=False, right_block=False, direction=0)
        self.G.add_node("goal", position=(self.env.width - 1, self.env.height - 1))
        self.G.add_node("obstacle", position=self.get_wall_coord())

        self.G.add_edge("agent", "goal", action="move_forward")

    def agent_check_surrounding(self):
      if isinstance(self.env.grid.get(self.env.agent_pos[0]+1, self.env.agent_pos[1]) , gym_minigrid.minigrid.Wall):
        self.update_wall_coord(self.env.agent_pos[0]+1, self.env.agent_pos[1])
      if isinstance(self.env.grid.get(self.env.agent_pos[0]+1, self.env.agent_pos[1]-1) , gym_minigrid.minigrid.Wall):
        self.update_wall_coord(self.env.agent_pos[0]+1, self.env.agent_pos[1]-1)
      if isinstance(self.env.grid.get(self.env.agent_pos[0]+1, self.env.agent_pos[1]+1) , gym_minigrid.minigrid.Wall):
        self.update_wall_coord(self.env.agent_pos[0]+1, self.env.agent_pos[1]+1)
      if isinstance(self.env.grid.get(self.env.agent_pos[0]-1, self.env.agent_pos[1]) , gym_minigrid.minigrid.Wall):
        self.update_wall_coord(self.env.agent_pos[0]-1, self.env.agent_pos[1])
      if isinstance(self.env.grid.get(self.env.agent_pos[0]-1, self.env.agent_pos[1]-1) , gym_minigrid.minigrid.Wall):
        self.update_wall_coord(self.env.agent_pos[0]-1, self.env.agent_pos[1]-1)
      if isinstance(self.env.grid.get(self.env.agent_pos[0]-1, self.env.agent_pos[1]+1) , gym_minigrid.minigrid.Wall):
        self.update_wall_coord(self.env.agent_pos[0]-1, self.env.agent_pos[1]+1)
      if isinstance(self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]-1) , gym_minigrid.minigrid.Wall):
        self.update_wall_coord(self.env.agent_pos[0], self.env.agent_pos[1]-1)
      if isinstance(self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]+1) , gym_minigrid.minigrid.Wall):
        self.update_wall_coord(self.env.agent_pos[0], self.env.agent_pos[1]+1)
      if self.G.nodes["agent"]["direction"] == 0:
        self.G.nodes["agent"]["front_block"] = (self.env.agent_pos[0] + 1, self.env.agent_pos[1]) in self.G.nodes['obstacle']['position']
        self.G.nodes["agent"]["left_block"] = (self.env.agent_pos[0], self.env.agent_pos[1] - 1) in self.G.nodes['obstacle']['position']
        self.G.nodes["agent"]["right_block"] = (self.env.agent_pos[0], self.env.agent_pos[1] + 1) in self.G.nodes['obstacle']['position']
        print(self.env.grid.get(self.env.agent_pos[0]+1, self.env.agent_pos[1]))
      elif self.G.nodes["agent"]["direction"] == 1:
        self.G.nodes["agent"]["front_block"] = (self.env.agent_pos[0], self.env.agent_pos[1] + 1) in self.G.nodes['obstacle']['position']
        self.G.nodes["agent"]["left_block"] = (self.env.agent_pos[0] + 1, self.env.agent_pos[1]) in self.G.nodes['obstacle']['position']
        self.G.nodes["agent"]["right_block"] = (self.env.agent_pos[0] - 1, self.env.agent_pos[1]) in self.G.nodes['obstacle']['position']
        print(self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]+1))
      elif self.G.nodes["agent"]["direction"] == 2:
        self.G.nodes["agent"]["front_block"] = (self.env.agent_pos[0] - 1, self.env.agent_pos[1]) in self.G.nodes['obstacle']['position']
        self.G.nodes["agent"]["left_block"] = (self.env.agent_pos[0], self.env.agent_pos[1] + 1) in self.G.nodes['obstacle']['position']
        self.G.nodes["agent"]["right_block"] = (self.env.agent_pos[0], self.env.agent_pos[1] - 1) in self.G.nodes['obstacle']['position']
        print(self.env.grid.get(self.env.agent_pos[0]-1, self.env.agent_pos[1]))
      else:
        self.G.nodes["agent"]["front_block"] = (self.env.agent_pos[0], self.env.agent_pos[1] - 1) in self.G.nodes['obstacle']['position']
        self.G.nodes["agent"]["left_block"] = (self.env.agent_pos[0] - 1, self.env.agent_pos[1]) in self.G.nodes['obstacle']['position']
        self.G.nodes["agent"]["right_block"] = (self.env.agent_pos[0] + 1, self.env.agent_pos[1]) in self.G.nodes['obstacle']['position']
        print(self.env.grid.get(self.env.agent_pos[0], self.env.agent_pos[1]-1))
      print(self.G.nodes['agent'])

    def update_knowledge_graph(self, env):
      # Update agent position
      self.env = env
      self.G.nodes["agent"]["position"] = (env.agent_pos[0], env.agent_pos[1])
      self.G.nodes["agent"]["direction"] = env.agent_dir

      # Update agent's surroundings (front_clear, left_clear, right_clear)
      self.agent_check_surrounding()
      # 0 - |>
      # 1 - V
      # 2 - <|
      # 3 - ^

      # Optionally update edges if relationships between entities have changed
      if not self.G.nodes["agent"]["front_block"] and not self.G.nodes["agent"]["right_block"] and not self.G.nodes["agent"]["left_block"]:
        if self.G.nodes["agent"]["direction"] == 0 and (self.env.agent_pos[0] + 1, self.env.agent_pos[1] - 1) in self.G.nodes['obstacle']['position'] and (self.env.agent_pos[0] + 1, self.env.agent_pos[1] + 1) in self.G.nodes['obstacle']['position']:
            self.G.add_edge("agent", "goal", action="move_forward")
        elif self.G.nodes["agent"]["direction"] == 1 and (self.env.agent_pos[0] + 1, self.env.agent_pos[1] + 1) in self.G.nodes['obstacle']['position'] and (self.env.agent_pos[0] - 1, self.env.agent_pos[1] + 1) in self.G.nodes['obstacle']['position']:
            self.G.add_edge("agent", "goal", action="move_forward")

        elif abs(self.env.agent_pos[0] - self.G.nodes['goal']['position'][0]) < abs(self.env.agent_pos[1] - self.G.nodes['goal']['position'][1]):
          if self.G.nodes["agent"]["direction"] == 0:
            self.G.add_edge("agent", "goal", action="move_forward")
          elif self.G.nodes['agent']['direction'] == 1:
            self.G.add_edge("agent", "goal", action="turn_left")
          elif self.G.nodes['agent']['direction'] == 2:
            self.G.add_edge("agent", "goal", action="turn_left")
          else:
            self.G.add_edge("agent", "goal", action="turn_right")
        elif abs(self.env.agent_pos[0] - self.G.nodes['goal']['position'][0]) > abs(self.env.agent_pos[1] - self.G.nodes['goal']['position'][1]):
          if self.G.nodes["agent"]["direction"] == 0:
            self.G.add_edge("agent", "goal", action="turn_right")
          elif self.G.nodes['agent']['direction'] == 1:
            self.G.add_edge("agent", "goal", action="move_forward")
          elif self.G.nodes['agent']['direction'] == 2:
            self.G.add_edge("agent", "goal", action="turn_left")
          else:
            self.G.add_edge("agent", "goal", action="turn_right")
        else:
          if self.env.agent_pos[0] < self.G.nodes['goal']['position'][0]:
            if self.G.nodes["agent"]["direction"] == 0:
              self.G.add_edge("agent", "goal", action="move_forward")
            elif self.G.nodes["agent"]["direction"] == 1:
              self.G.add_edge("agent", "goal", action="turn_left")
            elif self.G.nodes["agent"]["direction"] == 3:
              self.G.add_edge("agent", "goal", action="turn_right")

      elif self.G.nodes["agent"]["front_block"] and not self.G.nodes["agent"]["right_block"] and not self.G.nodes["agent"]["left_block"]:
        #if self.env.agent_pos[0] < self.G.nodes['goal']['position'][0]:
        if self.G.nodes["agent"]["direction"] == 0:
          self.G.add_edge("agent", "goal", action="turn_right")
        elif self.G.nodes['agent']['direction'] == 1:
          self.G.add_edge("agent", "goal", action="turn_left")
        elif self.G.nodes['agent']['direction'] == 2:
          self.G.add_edge("agent", "goal", action="turn_left")
        else:
          self.G.add_edge("agent", "goal", action="turn_right")


      elif self.G.nodes["agent"]["front_block"] and self.G.nodes["agent"]["left_block"] and not self.G.nodes["agent"]["right_block"]:
          self.G.add_edge("agent", "goal", action="turn_right")

      elif self.G.nodes["agent"]["front_block"] and not self.G.nodes["agent"]["left_block"] and self.G.nodes["agent"]["right_block"]:
        if self.G.nodes["agent"]["direction"] == 3:
          if (self.env.agent_pos[0] + 1, self.env.agent_pos[1] +1) not in self.G.nodes['obstacle']['position']:
            self.G.add_edge("agent", "goal", action="move_forward")

        elif abs(self.env.agent_pos[0] - self.G.nodes['goal']['position'][0]) < abs(self.env.agent_pos[1] - self.G.nodes['goal']['position'][1]):
          self.G.add_edge("agent", "goal", action="turn_right")
        elif abs(self.env.agent_pos[0] - self.G.nodes['goal']['position'][0]) > abs(self.env.agent_pos[1] - self.G.nodes['goal']['position'][1]):
          self.G.add_edge("agent", "goal", action="turn_left")
        else:
          if self.G.nodes["agent"]["direction"] == 0:
            self.G.add_edge("agent", "goal", action="turn_right")
          if self.G.nodes["agent"]["direction"] == 2:
            self.G.add_edge("agent", "goal", action="turn_left")

      elif not self.G.nodes["agent"]["front_block"] and self.G.nodes["agent"]["left_block"] and not self.G.nodes["agent"]["right_block"]:
        if self.G.nodes["agent"]["direction"] == 2 and (self.env.agent_pos[0] - 1, self.env.agent_pos[1] +1) not in self.G.nodes['obstacle']['position']:
            self.G.add_edge("agent", "goal", action="move_forward")
        elif self.G.nodes["agent"]["direction"] == 1 and (self.env.agent_pos[0] + 1, self.env.agent_pos[1] +1) not in self.G.nodes['obstacle']['position']:
            self.G.add_edge("agent", "goal", action="move_forward")
        elif self.G.nodes["agent"]["direction"] == 1 and (self.env.agent_pos[0] + 1, self.env.agent_pos[1] +1) in self.G.nodes['obstacle']['position']:
            self.G.add_edge("agent", "goal", action="move_forward")

        elif self.G.nodes["agent"]["direction"] == 0:
          if (self.env.agent_pos[0] + 1, self.env.agent_pos[1] +1) not in self.G.nodes['obstacle']['position']:
            self.G.add_edge("agent", "goal", action="move_forward")
        elif abs(self.env.agent_pos[0] - self.G.nodes['goal']['position'][0]) < abs(self.env.agent_pos[1] - self.G.nodes['goal']['position'][1]):
          self.G.add_edge("agent", "goal", action="move_forward")
        elif abs(self.env.agent_pos[0] - self.G.nodes['goal']['position'][0]) > abs(self.env.agent_pos[1] - self.G.nodes['goal']['position'][1]):
          self.G.add_edge("agent", "goal", action="turn_right")
        else:
          self.G.add_edge("agent", "goal", action="move_forward")


      elif not self.G.nodes["agent"]["front_block"] and not self.G.nodes["agent"]["left_block"] and self.G.nodes["agent"]["right_block"]:
        if self.G.nodes["agent"]["direction"] == 0:
          self.G.add_edge("agent", "goal", action="move_forward")
        elif self.G.nodes["agent"]["direction"] == 1:
          self.G.add_edge("agent", "goal", action="move_forward")
        elif self.G.nodes["agent"]["direction"] == 2:
          self.G.add_edge("agent", "goal", action="turn_left")
        elif self.G.nodes["agent"]["direction"] == 3:
          self.G.add_edge("agent", "goal", action="move_forward")

      elif not self.G.nodes["agent"]["front_block"] and self.G.nodes["agent"]["left_block"] and self.G.nodes["agent"]["right_block"]:
        self.G.add_edge("agent", "goal", action="move_forward")
      
      elif self.G.nodes["agent"]["front_block"] and self.G.nodes["agent"]["left_block"] and self.G.nodes["agent"]["right_block"]:
        self.G.add_edge("agent", "goal", action="turn_right")
      elif not self.G.nodes["agent"]["right_block"]:
        self.G.add_edge("agent", "goal", action="turn_right")
      elif not self.G.nodes["agent"]["left_block"]:
        self.G.add_edge("agent", "goal", action="turn_left")

      elif not self.G.nodes["agent"]["front_block"]:
        self.G.add_edge("agent", "goal", action="turn_right")

      return self.G

    def get_action(self):
      if self.G.edges[('agent', 'goal')]['action'] == 'move_forward':
        return env.actions.forward
      elif self.G.edges[('agent', 'goal')]['action'] == 'turn_right':
        return env.actions.right
      elif self.G.edges[('agent', 'goal')]['action'] == 'turn_left':
        return env.actions.left

    def get_wall_coord(self):
      rows = self.env.height
      cols = self.env.width

      outer_grid = []
      outer_grid.extend([(0, col) for col in range(cols)])
      outer_grid.extend([(rows - 1, col) for col in range(cols)])
      outer_grid.extend([(row, 0) for row in range(1, rows - 1)])
      outer_grid.extend([(row, cols - 1) for row in range(1, rows - 1)])
      return outer_grid

    def update_wall_coord(self, x, y):
      if(x,y) not in self.G.nodes['obstacle']['position']:
         self.G.nodes['obstacle']['position'].append((x,y))


# env = FlatObsWrapper(gym.make('MiniGrid-SimpleCrossingS9N1-v0', render_mode="rgb_array"))
# env.reset()
# KG =KnowledgeGraph ()
# KG.env = env
# KG.initialize()
# before_img = env.render()

# ImageConcate = before_img

# for i in range(30):
#   KG.update_knowledge_graph(env)
#   action = KG.get_action()
#   obs, reward, terminated, truncated, info  = env.step(action)
#   rendered_img = env.render()

#   ImageConcate = np.concatenate([ImageConcate, rendered_img], 1)
#   if terminated:
#     break

# env.close()
# plt.imshow(ImageConcate);
# #show_video()
# plt.axis('off')
# plt.show()

