import itertools as itt

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv

class DynamicGoalCrossingCustom13Env(MiniGridEnv):

    """
    ### Description

    Depending on the `obstacle_type` parameter:
    - `Lava` - The agent has to reach the green goal square on the other corner
        of the room while avoiding rivers of deadly lava which terminate the
        episode in failure. Each lava stream runs across the room either
        horizontally or vertically, and has a single crossing point which can be
        safely used; Luckily, a path to the goal is guaranteed to exist. This
        environment is useful for studying safety and safe exploration.
    - otherwise - Similar to the `LavaCrossing` environment, the agent has to
        reach the green goal square on the other corner of the room, however
        lava is replaced by walls. This MDP is therefore much easier and maybe
        useful for quickly testing your algorithms.

    ### Mission Space
    Depending on the `obstacle_type` parameter:
    - `Lava` - "avoid the lava and get to the green goal square"
    - otherwise - "find the opening and get to the green goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    S: size of the map SxS.
    N: number of valid crossings across lava or walls from the starting position
    to the goal

    - Wall :
        - `MiniGrid-SimpleCrossingS9N1-v0`
        - `MiniGrid-SimpleCrossingS9N2-v0`
        - `MiniGrid-SimpleCrossingS9N3-v0`
        - `MiniGrid-SimpleCrossingS11N5-v0`
        - `MiniGrid-SimpleCrossingS13N3-v0`

    """

    def __init__(self, size=13, num_crossings=4, obstacle_type=Lava, start_pos = (3, 1), goal_pos =(1, 1), **kwargs):
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
            self.put_obj(self.obstacle_type(), 5, 4)
            self.put_obj(self.obstacle_type(), 5, 3)

        self.mission = (
            "get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
