from gymnasium.envs.registration import register
from minigrid import minigrid_env, wrappers
from minigrid.core import roomgrid
from minigrid.core.world_object import Wall, Lava


def register_minigrid_envs():
    # Dyanmic Goal Empty Custom 13
    # base goals:
    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0", # for base
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (11, 1), "obstacle_type": Wall, "start_pos": (1, 1)},
    )
    
    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0", # for dynamic
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (11, 11), "obstacle_type": Wall, "start_pos": (1, 1)},
    )

    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0", # for base
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (1, 11), "obstacle_type": Wall, "start_pos": (1, 1)},
    )

    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0", # for base
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (7, 11), "obstacle_type": Wall, "start_pos": (1, 1)},
    )
    # end of base goals
    
    #dynamic goals:
    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x3-v0", # for dynamic
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (11, 3), "obstacle_type": Wall, "start_pos": (1, 1)},
    )
    
    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x5-v0", # for dynamic
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (11, 5), "obstacle_type": Wall, "start_pos": (1, 1)},
    )
    
    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x8-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (11, 8), "obstacle_type": Wall, "start_pos": (1, 1)},
    )

    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0", # for base
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (8, 1), "obstacle_type": Wall, "start_pos": (1, 1)},
    )

    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x1-v0", # for dynamic
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (6, 1), "obstacle_type": Wall, "start_pos": (1, 1)},
    )

    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x7-v0", # for dynamic, to confuse with 9x1(within same room)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (1, 7), "obstacle_type": Wall, "start_pos": (1, 1)},
    )
    
    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-5x9-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (5, 9), "obstacle_type": Wall, "start_pos": (1, 1)},
    )

    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (10, 6), "obstacle_type": Wall, "start_pos": (1, 1)},
    )

    register(
        id="MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 13, "num_crossings": 4, "goal_pos": (6, 9), "obstacle_type": Wall, "start_pos": (1, 1)},
    )
    # end of dynamic goals

    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-7x7-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (7, 7), "obstacle_type": Lava},
    )

    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-1x7-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (1, 7), "obstacle_type": Lava},
    )

    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-7x1-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (7, 1), "obstacle_type": Lava},
    )

    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (1, 3), "obstacle_type": Lava},
    )
    
    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (6, 5), "obstacle_type": Lava},
    )
    
    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (4, 7), "obstacle_type": Lava},
    )
    
    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (2, 5), "obstacle_type": Lava},
    )
    
    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (5, 2), "obstacle_type": Lava},
    )
    
    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (4, 5), "obstacle_type": Lava},
    )
    
    register(
        id="MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0", # for dynamic, to confuse with 9x1 (outside the room of 9x1)
        entry_point="gr_envs.minigrid_scripts.envs:DynamicGoalCrossingCustom13Env",
        kwargs={"size": 9, "num_crossings": 2, "goal_pos": (1, 1), "obstacle_type": Lava},
    )
    

register_minigrid_envs()
# python experiments.py --recognizer graml --domain point_maze --task L2 --partial_obs_type continuing --point_maze_env obstacles --collect_stats --inference_same_seq_len
