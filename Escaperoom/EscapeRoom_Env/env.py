from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.constants import COLOR_TO_IDX
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Ball, Floor
from minigrid.minigrid_env import MiniGridEnv
from operator import add

from EscapeRoom_Env.grid import ColorGrid
from EscapeRoom_Env.Characters import Helldog
import EscapeRoom_Env.config as cf

import random

WALL = 'grey'
FLOOR = 'none'
ATTACK = "red"

class LockedRoom:
    def __init__(self, top, size, doorPos):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(topX + 1, topX + sizeX - 1, topY + 1, topY + sizeY - 1)

class EscapeRoomEnv(MiniGridEnv):
    """
    ## Description

    The environment has six rooms, one of which is locked. The agent receives
    a textual mission string as input, telling it which room to go to in order
    to get the key that opens the locked room. It then has to go into the locked
    room in order to reach the final goal. This environment is extremely
    difficult to solve with vanilla reinforcement learning alone.

    ## Mission Space

    "get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal"

    {lockedroom_color}, {keyroom_color}, and {door_color} can be "red", "green",
    "blue", "purple", "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-LockedRoom-v0`

    """
    def __init__(
        self,
        size=19, 
        num_guardians=3,
        max_steps: int | None = None, 
        **kwargs):
        self.size = size

        if max_steps is None:
            max_steps = 10 * size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES] * 3,
        )

        self.num_guardians = num_guardians 

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(lockedroom_color: str, keyroom_color: str, door_color: str):
        return (
            f"get the {lockedroom_color} key from the {keyroom_color} room,"
            f" unlock the {door_color} door and go to the goal"
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = ColorGrid(width, height)

        # Generate the surrounding walls
        WallThickness = 1
        for i in range(0, width):
            for row in range(0,WallThickness):
                self.grid.set(i, row, Wall())
                self.grid.set(i, height - row - 1, Wall())

                #Record the color of the grid
                self.grid.set_color(i,row,WALL)
                self.grid.set_color(i,height - row -1,WALL)

        for j in range(0, height):
            for col in range(0,WallThickness):
                self.grid.set(col, j, Wall())
                self.grid.set(width - col - 1, j, Wall())

                self.grid.set_color(col,j,WALL)
                self.grid.set_color(width-col-1, j, WALL)

        # Hallway walls
        Hallway_width = (width -4*WallThickness) // 10 * 3 + width % 2 #Place the hallway at center that all rooms have same width
        lWallIdx = (width - Hallway_width) // 2 
        rWallIdx = lWallIdx + Hallway_width
        for j in range(0, height):
            self.grid.set(lWallIdx, j, Wall())
            self.grid.set(rWallIdx, j, Wall())

            self.grid.set_color(lWallIdx,j,WALL)
            self.grid.set_color(rWallIdx,j,WALL)

        self.rooms = []


        # Room splitting walls
        for n in range(0, 3):
            j = n * (height // 3)
            for i in range(0, lWallIdx):
                self.grid.set(i, j, Wall())
                self.grid.set_color(i,j,WALL)
            for i in range(rWallIdx, width):
                self.grid.set(i, j, Wall())

            roomW = lWallIdx + 1
            roomH = height // 3 + 1
            self.rooms.append(LockedRoom((0, j), (roomW, roomH), (lWallIdx, j + 3)))
            self.rooms.append(LockedRoom((rWallIdx, j), (roomW, roomH), (rWallIdx, j + 3))
            )

        self.occupiedRoom = []
        self.occupiedPos = []

        # Choose one random room to be locked
        lockedRoom = self._rand_elem(self.rooms)
        lockedRoom.locked = True
        goalPos = lockedRoom.rand_pos(self)
        self.grid.set(*goalPos, Goal())

        self.occupiedRoom.append(lockedRoom) #Record occupied room 
        self.occupiedPos.append(goalPos)     #Record occupied Pos

        # Generate lava wall beside the goal 
        #self.grid.vert_wall(goalPos[0]+1,goalPos[1],roomH - goalPos[1]%roomH-3, Lava)

        # Assign the door colors
        colors = set(COLOR_NAMES)
        for room in self.rooms:
            color = self._rand_elem(sorted(colors))
            colors.remove(color)
            room.color = color
            if room.locked:
                self.grid.set(*room.doorPos, Door(color, is_locked=True))
            else:
                self.grid.set(*room.doorPos, Door(color))

        # Select a random room to contain the key
        while True:
            keyRoom = self._rand_elem(self.rooms)
            if keyRoom != lockedRoom:
                break
        keyPos = keyRoom.rand_pos(self)
        self.grid.set(*keyPos, Key(lockedRoom.color))

        self.occupiedRoom.append(keyRoom) 
        self.occupiedPos.append(keyPos)     

        # Randomize the player start position and orientation
        self.agent_pos = self.place_agent(
            top=(lWallIdx, 0), size=(rWallIdx - lWallIdx, height)
        )

    
         # Create and place enemies
        self.helldog_list = []

        '''Place in the hallway'''
        for helldog in range (cf.num_helldogs):
            self.helldog_list.append(Helldog(color = "red"))
            self.place_obj(self.helldog_list[helldog], top=(lWallIdx, 0), size=(Hallway_width, height), max_tries=100)
        
        '''Place in rooms'''
        
        for i in range(4):
            dog_count = 0
            while True:
                dogroom = self._rand_elem(self.rooms)
                if dogroom not in self.occupiedRoom:
                    if dog_count <= 2:
                        helldog = Helldog(color = "blue")
                        self.helldog_list.append(helldog)
                        dog_pos = dogroom.rand_pos(self)
                        self.put_obj(helldog,*dog_pos)
                        dog_count += 1
                    else:
                        self.occupiedRoom.append(dogroom)
                    break
                    
            
            
            
        # Generate the mission string
        self.mission = (
            "get the %s key from the %s room, "
            "unlock the %s door and "
            "go to the goal"
        ) % (lockedRoom.color, keyRoom.color, lockedRoom.color)


    def step(self,action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Update obstacle positions
        for helldog in self.helldog_list:
            if helldog.alert != True:
                helldog.move(self)
            helldog.attack(self,self.agent_pos)
        
        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != "goal" and front_cell.type != "door" and front_cell.type !="key" and front_cell.type !="wall"

        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info
        
        elif self.grid.get_color(*self.agent_pos) == cf.ATTACK:
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info

    

