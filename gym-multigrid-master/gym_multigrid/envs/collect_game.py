# from gym_multigrid.multigrid import *

# class CollectGameEnv(MultiGridEnv):
#     """
#     Environment in which the agents have to collect the balls
#     """

#     def __init__(
#         self,
#         size=10,
#         width=None,
#         height=None,
#         num_balls=[],
#         agents_index = [],
#         balls_index=[],
#         balls_reward=[],
#         zero_sum = False,
#         view_size=7

#     ):
#         self.num_balls = num_balls
#         self.balls_index = balls_index
#         self.balls_reward = balls_reward
#         self.zero_sum = zero_sum
        
#         self.ball_counts= num_balls[0]
        

#         self.world = World

#         agents = []
#         for i in agents_index:
#             agents.append(Agent(self.world, i, view_size=view_size))

#         super().__init__(
#             grid_size=size,
#             width=width,
#             height=height,
#             max_steps= 10000,
#             # Set this to True for maximum speed
#             see_through_walls=False,
#             agents=agents,
#             agent_view_size=view_size
#         )



#     def _gen_grid(self, width, height):
#         self.grid = Grid(width, height)

#         # Generate the surrounding walls
#         self.grid.horz_wall(self.world, 0, 0)
#         self.grid.horz_wall(self.world, 0, height-1)
#         self.grid.vert_wall(self.world, 0, 0)
#         self.grid.vert_wall(self.world, width-1, 0)

#         for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
#             for i in range(number):
#                 self.place_obj(Ball(self.world, index, reward))

#         # Randomize the player start position and orientation
#         for a in self.agents:
#             self.place_agent(a)

#     def _handle_wall(self,i, rewards, reward=-0.1):
#         """
#         Compute the reward if hit wall
#         """
#         for j,a in enumerate(self.agents):
#             if a.index==i or a.index==0:
#                 rewards[j]+=reward
#             if self.zero_sum:
#                 if a.index!=i or a.index==0:
#                     rewards[j] -= reward
#     def _handle_still(self,i, rewards, reward=-0.1):
#         """
#         Compute the reward if stay still
#         """
#         print("agents still")
#         for j,a in enumerate(self.agents):
#             if a.index==i or a.index==0:
#                 rewards[j]+=reward
#             if self.zero_sum:
#                 if a.index!=i or a.index==0:
#                     rewards[j] -= reward

#     def _reward(self, i, rewards, reward=1):
#         """
#         Compute the reward to be given upon success
#         """
#         # for j,a in enumerate(self.agents):
#         #     if a.index==i or a.index==0:
#         #         rewards[j]+=reward
#         #     if self.zero_sum:
#         #         if a.index!=i or a.index==0:
#         #             rewards[j] -= reward
#         rewards += reward

#     def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
#         if fwd_cell:
#             if fwd_cell.can_pickup():
#                 if fwd_cell.index in [0, self.agents[i].index]:
#                     fwd_cell.cur_pos = np.array([-1, -1])
#                     self.grid.set(*fwd_pos, None)
#                     self._reward(i, rewards, fwd_cell.reward)

#     def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
#         pass
    

#     def step(self, actions):
#         obs, rewards, done, info = MultiGridEnv.step(self, actions)
#         return obs, rewards, done, info


# class CollectGame4HEnv10x10N2(CollectGameEnv):
#     def __init__(self):
#         super().__init__(size=10,
#         num_balls=[5],
#         agents_index = [1,2,3,4],
#         balls_index=[0],
#         balls_reward=[10],
#         zero_sum=False)
        
# class CollectGame1HEnv10x10(CollectGameEnv):
#     def __init__(self):
#         super().__init__(size=10,
#         num_balls=[5],
#         agents_index = [1],
#         balls_index=[0],
#         balls_reward=[1],
#         zero_sum=True)



from gym_multigrid.multigrid import *
import time

class CollectGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        num_lava=[],
        lava_index=[],
        agents_index = [],
        balls_index=[],
        key_index=[],
        balls_reward=[],
        lava_reward=[],
        zero_sum = False,
        view_size=7

    ):
        self.num_balls = num_balls
        self.num_lava = num_lava
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum
        self.lava_reward=lava_reward
        self.ball_counts= num_balls[0]
        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))
            # if agents_index:
            #     agents[i-1].terminated = False


        self.alive_agents_count = len(agents)
        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )



    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
            for i in range(number):
                self.place_obj(Ball(self.world, index, reward))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)
        

        

    def _handle_wall(self,i, rewards, reward=-1):
        """
        Compute the reward if hit wall
        """
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward
    
    def _handle_still(self,i, rewards, reward=-0.1):
        """
        Compute the reward if stay still
        """
        print("agents still")
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        # for j,a in enumerate(self.agents):
        #     if a.index==i or a.index==0:
        #         rewards[j]+=reward
        #     if self.zero_sum:
        #         if a.index!=i or a.index==0:
        #             rewards[j] -= reward
        rewards += reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, 1, self.agents[i].index]:
                    if self.agents[i].carrying is None and fwd_cell.type == "key":
                        self.agents[i].carrying = fwd_cell
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self._reward(i, rewards, fwd_cell.reward)



    ### For killing agents
    # def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell): 
    #     if fwd_cell:
    #         self.agents[i].terminated = True
    #         self.grid.set(*fwd_pos, None)

        
    

    def step(self, actions):
        
        obs, rewards, done, info = MultiGridEnv.step(self, actions)

        # Terminate if all agetns die
        alive_check = 0
        for a in self.agents:
            if a.terminated:
                alive_check += 1
            if alive_check == len(self.agents):
                done = True

        if done:
            for a in self.agents:
                a.terminated = False
        
    
        return obs, rewards, done, info

class CollectGame4HEnv10x10N2(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[5],
        agents_index = [1,2,3,4],
        balls_index=[0],
        balls_reward=[50],
        zero_sum=False)

class CollectGame4HEnv10x10N2Lava(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[5],
        agents_index = [1,2,3,4],
        balls_index=[0],
        num_lava=[3],
        lava_reward=[-10],
        balls_reward=[50],
        zero_sum=False)
        

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
            for i in range(number):
                self.place_obj(Ball(self.world, index, reward))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

        # Generate lava at random pos
        lava_count = self.num_lava[0] #remaining lava tiles need to be placed
        while(lava_count != 0):
            for number, reward in zip(self.num_lava, self.lava_reward):
                for i in range(lava_count):
                    self.place_obj(Lava(self.world, reward))
                    lava_count -= 1

class CollectGame1HEnv10x10(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[5],
        agents_index = [1],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=True)

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
    
class KeyCollectGame4HEnv10x10N2(CollectGameEnv):
    def __init__(self,
                 Roomsize = 7,
                 ball_count = 4,
                 lava_count = 2): 
        
        self.roomW = Roomsize-1
        self.roomH = Roomsize-1

        super().__init__(
            size = None,
            width = Roomsize *2,
            height = Roomsize,
            num_balls=[ball_count],
            num_lava=[lava_count],
            key_index=[1], 
            agents_index = [1,2,3,4],
            balls_index=[0],
            balls_reward=[10],
            lava_reward=[-10],
            zero_sum=False)
        
        
    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Generate room-splitting walls
        wall_pos = self.roomW
        self.grid.vert_wall(self.world, wall_pos, 0)

        #Generate rooms
        self.rooms = []
        doorPos = np.random.randint(1,self.roomH)
        self.rooms.append(LockedRoom((0, 0), (self.roomW, self.roomH), 
                                     (wall_pos, doorPos)))
        
        self.rooms.append(LockedRoom((wall_pos,0),(self.roomW,self.roomH),(wall_pos,doorPos)))
        
        # Choose one random room to be locked
        lockedRoom = self._rand_elem(self.rooms)
        lockedRoom.locked = True

        # Assign the door color
        colors = set(COLOR_NAMES)
        color = self._rand_elem(sorted(colors))
        colors.remove(color)
        lockedRoom.color = color
        self.grid.set(*lockedRoom.doorPos,Door(self.world, color, is_locked=True))

        # Generate key in another room at a random pos
        while True:
            keyRoom = self._rand_elem(self.rooms)
            if keyRoom != lockedRoom:
                break
        self.place_obj(Key(self.world,lockedRoom.color),keyRoom.top,keyRoom.size)
        
        # Randomize the balls location
        for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
            for i in range(number):
                self.place_obj(Ball(self.world, index, reward))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a,keyRoom.top,keyRoom.size)

        # Generate lava at random pos
        lava_count = self.num_lava[0] #remaining lava tiles need to be placed
        while(lava_count != 0):
            for number, reward in zip(self.num_lava, self.lava_reward):
                for i in range(lava_count):
                    self.place_obj(Lava(self.world, reward))
                    lava_count -= 1

            # Clear the door area
            nearby_cell = []
            for room in self.rooms:
                nearby_cell.append((room.doorPos[0]+1,room.doorPos[1]))
                nearby_cell.append((room.doorPos[0]-1,room.doorPos[1]))
                nearby_cell.append((room.doorPos[0]+1,room.doorPos[1]+1))
                nearby_cell.append((room.doorPos[0]+1,room.doorPos[1]-1))
                nearby_cell.append((room.doorPos[0]-1,room.doorPos[1]-1))
                nearby_cell.append((room.doorPos[0]-1,room.doorPos[1]+1))
                nearby_cell.append((room.doorPos[0]+2,room.doorPos[1]))
                nearby_cell.append((room.doorPos[0]-2,room.doorPos[1]))
            for cell in nearby_cell:
                cell_type = self.grid.get(*cell)
                if cell_type != None:
                    if cell_type.type == "lava":
                        self.grid.set(*cell, None)
                        lava_count += 1
        