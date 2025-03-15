import gymnasium
from gymnasium.envs.registration import register
from gr_envs.maze_scripts.envs.maze.generate_maze import gen_empty_env, gen_four_rooms_env, gen_maze_with_obstacles
from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.maze import maps


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""
    ### MAZE SPECIAL ENVS ###
    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        for width, height in [(11, 11)]:
            for start_x, start_y in [(1, 1)]:
                register(
                    id=f"PointMaze-FourRoomsEnv{suffix}-{width}x{height}-Goals-9x1-1x9-9x9",
                    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                    kwargs= {
                        "reward_type": reward_type,
                        "maze_map": gen_four_rooms_env(width, height, [(start_x, start_y)], [(1, 9), (9, 1), (9, 9)]),
                        "continuing_task": False
                    },
                    max_episode_steps=900,
                )
            for goal_x, goal_y in [(1, 9), (9, 1), (9, 9), (7, 3), (3, 7), (6, 4), (4, 6), (3, 3), (6, 6), (4, 4),
                                   (3, 4), (7, 7), (6, 7), (8, 8), (7, 4), (4, 7), (6, 3), (3, 6), (5, 5), (5, 1),
                                   (1, 5), (8, 2), (2, 8), (3, 4), (4, 3)]:
                register(
                    id=f"PointMaze-EmptyEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}",
                    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                    kwargs= {
                        "reward_type": reward_type,
                        "maze_map": gen_empty_env(width, height, [(start_x, start_y)], [(goal_x, goal_y)]),
                        "continuing_task": False
                    },
                    max_episode_steps=900,
                )
                register(
                    id=f"PointMaze-FourRoomsEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}",
                    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                    kwargs= {
                        "reward_type": reward_type,
                        "maze_map": gen_four_rooms_env(width, height, [(start_x, start_y)], [(goal_x, goal_y)]),
                        "continuing_task": False
                    },
                    max_episode_steps=900,
                )
                register(
                    id=f"PointMaze-ObstaclesEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}",
                    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                    kwargs= {
                        "reward_type": reward_type,
                        "maze_map": gen_maze_with_obstacles(11, 11, [(1, 1)], [(goal_x, goal_y)], [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]),
                        "continuing_task": False
                    },
                    max_episode_steps=900,
                )
    ### END OF MAZE SPECIAL ENVS ###
    
    ### kitchen special envs ###
    register(
        id="FrankaKitchen-v1",
        entry_point="gymnasium_robotics.envs.franka_kitchen:KitchenEnv",
        max_episode_steps=1000, # default is 280, but since this is a sparse reward environment, better increase it so the agent can 
    )
    ### END OF kitchen special envs ###

    def _merge(a, b):
        a.update(b)
        return a

register_robotics_envs()