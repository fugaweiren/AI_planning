from minigrid.core.world_object import WorldObj
from minigrid.utils import fill_coords, point_in_circle
from minigrid.core.constants import COLORS
from minigrid.core.grid import Lava

class Enemy(WorldObj):
    def __init__(self, name, color="red"):
        super().__init__(name, color)
        self.direction = 0  # 0: Up, 1: Right, 2: Down, 3: Left

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
        super().__init__("helldog", color)
        self.lava_timer = 0  # Timer for lava left behind

    def render(self, img):
        # Render the Helldog as a red circle
        fill_coords(img, point_in_circle(0.5, 0.5, 0.3), COLORS[self.color])

    def move(self, env):
        if self.lava_timer > 0:
            self.lava_timer -= 1
        else:
            #Remove the lava floor 
            

            # Move 2 steps in the current direction
            dx, dy = self.get_direction_delta()
            new_pos_1 = (self.cur_pos[0] + dx, self.cur_pos[1] + dy)
            new_pos_2 = (self.cur_pos[0] + 2 * dx, self.cur_pos[1] + 2 * dy)

            # Check if the position is free for both moves
            if env.grid.get(*new_pos_1) is None and env.grid.get(*new_pos_2) is None:
                # Set the current position to None (removing the Helldog)
                env.grid.set(*self.cur_pos, None)

                # Move to the new position
                self.cur_pos = new_pos_2
                env.grid.set(*self.cur_pos, self)

                # Leave lava at the first position
                env.grid.set(*new_pos_1, Lava())

                # Set the lava timer for 3 rounds
                self.lava_timer = 3

    def get_direction_delta(self):
        """ Return the x and y deltas for the current direction. """
        if self.direction == 0:    # Up
            return (0, 1)
        elif self.direction == 1:  # Right
            return (1, 0)
        elif self.direction == 2:  # Down
            return (0, -1)
        elif self.direction == 3:  # Left
            return (-1, 0)

    def attack(self, agent_pos):
        if self.lava_timer == 0:  # Only attack if not leaving lava
            if abs(self.cur_pos[0] - agent_pos[0]) <= 1 and abs(self.cur_pos[1] - agent_pos[1]) <= 1:
                print("Helldog attacks the agent!")

class Archer(Enemy):
    def __init__(self, color="green"):
        super().__init__("archer", color)

    def render(self, img):
        # Render the Archer as a green square
        fill_coords(img, point_in_circle(0.5, 0.5, 0.3), COLORS[self.color])

    def move(self, env):
        # Archer does not move
        pass

    def attack(self, agent_pos):
        if self.is_in_sight(agent_pos):
            print("Archer attacks the agent from a distance!")

    def is_in_sight(self, agent_pos):
        """ Check if the agent is in the line of sight (same row or column). """
        return (self.cur_pos[0] == agent_pos[0]) or (self.cur_pos[1] == agent_pos[1])

class Fireball(Enemy):
    def __init__(self, color="orange"):
        super().__init__("fireball", color)
        self.direction = 1  # Initially move right
        self.steps = 0  # Count steps for back and forth movement

    def render(self, img):
        # Render the Fireball as an orange circle
        fill_coords(img, point_in_circle(0.5, 0.5, 0.3), COLORS[self.color])

    def move(self, env):
        dx, dy = self.get_direction_delta()
        new_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dy)

        # Move only if the next position is free
        if env.grid.get(*new_pos) is None:
            env.grid.set(*self.cur_pos, None)  # Remove from current position
            self.cur_pos = new_pos
            env.grid.set(*self.cur_pos, self)  # Set new position

            # Leave a lava trail behind
            env.grid.set(*(self.cur_pos[0] - dx, self.cur_pos[1] - dy), Lava())
            self.steps += 1

            # Alternate direction every 2 steps
            if self.steps % 2 == 0:
                self.direction = (self.direction + 1) % 2  # Toggle direction (0: back, 1: forth)

    def get_direction_delta(self):
        """ Return the direction of movement (right or left). """
        if self.direction == 0:  # Move left
            return (-1, 0)
        elif self.direction == 1:  # Move right
            return (1, 0)

    def attack(self, agent_pos):
        # Fireball does not attack, but could implement logic if needed
        pass

# In your custom environment file
from custom_minigrid_env.custom_enemies import Helldog, Archer, Fireball

def _gen_grid(self, width, height):
    self.grid = CustomGrid(width, height)
    
    # Create and place enemies
    helldog = Helldog(color="red")
    helldog.set_direction(0)  # Example direction (up)
    self.grid.set(5, 5, helldog)
    
    archer = Archer(color="green")
    archer.set_direction(1)  # Example direction (right)
    self.grid.set(7, 5, archer)
    
    fireball = Fireball(color="orange")
    fireball.set_direction(1)  # Example direction (right)
    self.grid.set(3, 5, fireball)
