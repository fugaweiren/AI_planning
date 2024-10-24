from minigrid.core.world_object import WorldObj
from minigrid.core.world_object import Lava, Floor
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle, point_in_triangle
from minigrid.core.constants import COLORS
from operator import add
import random
import EscapeRoom_Env.config as cf

class Enemy(WorldObj):
    def __init__(self, name, color="red"):
        super().__init__(name, color)
        self.direction = cf.DOWN  # 0: Up, 1: Right, 2: Down, 3: Left

    def set_direction(self, direction):
        """ Set the direction of the enemy. """
        self.direction = direction

    def move(self, env):
        """ Move the enemy in its current facing direction. """
        # This will be overridden by subclasses
        raise NotImplementedError

    def attack(self, agent_pos):
        """ Check if the enemy can attack the agent. """
        # This will be overridden by subclasses
        raise NotImplementedError
    

class Helldog(Enemy):
    def __init__(self, color="red"):
        super().__init__("ball", color)
        self.lava_timer = 0  # Timer for lava left behind
        self.alert_pos = (0,0)
        self.lava_pos = []
        self.alert = False
        


    def render(self,img):
        #Render helldog as a arrow facing different directions 

        if self.direction == cf.UP:
            fill_coords(img, point_in_triangle((0.5, 0.8), (0.3, 0.2), (0.7, 0.2)), COLORS[self.color])
        
        elif self.direction == cf.RIGHT:
            fill_coords(img, point_in_triangle((0.8, 0.5), (0.2, 0.3), (0.2, 0.7)), COLORS[self.color])

        elif self.direction == cf.DOWN:
            fill_coords(img, point_in_triangle((0.5, 0.2), (0.3, 0.8), (0.7, 0.8)), COLORS[self.color])

        elif self.direction == cf.LEFT:
            fill_coords(img, point_in_triangle((0.2, 0.5), (0.8, 0.3), (0.8, 0.7)), COLORS[self.color])

    def move(self, env):
        old_pos = (self.cur_pos[0],self.cur_pos[1])

        if self.lava_timer > 0:
            self.lava_timer -= 1
        else:
            #reset lava
            for lava in range(len(self.lava_pos)):
                env.grid.set(*self.lava_pos[lava],None)
            self.lava_pos = []

            # Move 2 steps in a random direction
            self.direction = random.randint(0,3)
            dx, dy = self.get_direction_delta()
            new_pos_1 = (self.cur_pos[0] + dx, self.cur_pos[1] + dy)
            new_pos_2 = (self.cur_pos[0] + 2 * dx, self.cur_pos[1] + 2 * dy)

            # Check if the position is free for both moves
            if env.grid.get(*new_pos_1) is None and env.grid.get(*new_pos_2) is None:
                # Set the current position to None (removing the Helldog)
                env.grid.set(*self.cur_pos, Floor())

                # Move to the new position
                self.cur_pos = new_pos_2
                env.grid.set(*self.cur_pos, self)

                # Leave lava at the first position and old position
                env.grid.set(*new_pos_1, Lava())
                env.grid.set(*old_pos, Lava())
                self.lava_pos.append(new_pos_1)
                self.lava_pos.append(old_pos)

                # Set the lava timer for 3 rounds
                self.lava_timer = cf.lava_timer

            elif env.grid.get(*new_pos_1) is None:
                # Set the current position to None (removing the Helldog)
                env.grid.set(*self.cur_pos, None)

                # Move to the new position
                self.cur_pos = new_pos_1
                env.grid.set(*self.cur_pos, self)
                

                # Leave lava at the first position
                env.grid.set(*old_pos, Lava())
                self.lava_pos.append(old_pos)

                # Set the lava timer for 3 rounds
                self.lava_timer = cf.lava_timer

    def get_direction_delta(self):
        """ Return the x and y deltas for the current direction. """
        if self.direction == cf.UP:   
            return (0, 1)
        elif self.direction == cf.RIGHT:  
            return (1, 0)
        elif self.direction == cf.DOWN:  
            return (0, -1)
        elif self.direction == cf.LEFT:  
            return (-1, 0)

    def attack(self, env,agent_pos):
        #Remove old alert pos
        if self.alert_pos != (0,0) and self.alert_pos != agent_pos:
            env.grid.set(*self.alert_pos,None)
            env.grid.set_color(*self.alert_pos,None)
            self.alert = False

        if abs(self.cur_pos[0] - agent_pos[0]) <= 1 and abs(self.cur_pos[1] - agent_pos[1]) <= 1:
            if agent_pos != self.alert_pos:
                env.grid.set(agent_pos[0],agent_pos[1], Floor(color = cf.ALERT))
                env.grid.set_color(agent_pos[0],agent_pos[1],cf.ALERT)
                self.alert_pos = agent_pos
                self.alert = True
            else:
                env.grid.set_color(agent_pos[0],agent_pos[1],None)
                env.grid.set(self.alert_pos[0],self.alert_pos[1],Floor(color =cf.ATTACK))
                env.grid.set_color(agent_pos[0],agent_pos[1],cf.ATTACK)
                self.alert = False
                
        