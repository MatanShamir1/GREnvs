import itertools as itt

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Floor
from minigrid.minigrid_env import MiniGridEnv


class CustomColorEnv(MiniGridEnv):

    def __init__(self, plan : list = [], size=13, num_crossings=4, obstacle_type=Lava, start_pos = (3, 1), goal_pos =(1, 1), **kwargs):
        self.size = size
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type

        if obstacle_type == Lava:
            mission_space = MissionSpace(
                mission_func=lambda: "avoid the lava and get to the green goal square"
            )
        else:
            mission_space = MissionSpace(
                mission_func=lambda: "find the opening and get to the green goal square"
            )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=False,
            **kwargs
        )
        self.goal_x, self.goal_y = goal_pos
        self.start_pos = start_pos
        self.plan = plan

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array(self.start_pos)
        self.agent_dir = 0

        # Place a goal square in the specified location
        self.put_obj(Goal(), self.goal_x, self.goal_y)
        if self.size == 13:
            # room
            self.put_obj(self.obstacle_type(), 7, 1)
            self.put_obj(self.obstacle_type(), 7, 2)
            self.put_obj(self.obstacle_type(), 7, 3)
            self.put_obj(self.obstacle_type(), 7, 4)
            
            self.put_obj(self.obstacle_type(), 11, 4)
            self.put_obj(self.obstacle_type(), 10, 4)
            self.put_obj(self.obstacle_type(), 9, 4)
            # end_of_room
            
            self.put_obj(self.obstacle_type(), 8, 11)
            self.put_obj(self.obstacle_type(), 8, 10)
            self.put_obj(self.obstacle_type(), 8, 9)
            self.put_obj(self.obstacle_type(), 8, 8)

            self.put_obj(self.obstacle_type(), 2, 8)
            self.put_obj(self.obstacle_type(), 3, 8)
            self.put_obj(self.obstacle_type(), 4, 8)
            self.put_obj(self.obstacle_type(), 5, 8)
        
        else:
            self.put_obj(self.obstacle_type(), 2, 1)
            self.put_obj(self.obstacle_type(), 2, 2)
            self.put_obj(self.obstacle_type(), 2, 3)
            self.put_obj(self.obstacle_type(), 2, 4)
            
            self.put_obj(self.obstacle_type(), 5, 7)
            self.put_obj(self.obstacle_type(), 5, 6)
            self.put_obj(self.obstacle_type(), 5, 5)
            self.put_obj(self.obstacle_type(), 5, 4)
            self.put_obj(self.obstacle_type(), 5, 3)
                
            self.mission = (
                "avoid the lava and get to the green goal square"
                if self.obstacle_type == Lava
                else "find the opening and get to the green goal square"
            )
            
        # draw the plan
        for x,y in self.plan:
            if x == self.goal_x and y == self.goal_y:
                continue
            elif (x,y) == self.start_pos:
                self.grid.set(x, y, Floor(color='yellow'))
            else:
                self.grid.set(x, y, Floor(color='purple'))
                
