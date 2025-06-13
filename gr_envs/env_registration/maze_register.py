import gymnasium as gym
from gymnasium.envs.registration import register, registry
from gymnasium.envs.registration import registry

from gr_envs.wrappers.point_maze_wrapper import PointMazeWrapper, PointMazeGoalList
from gr_envs.wrappers.point_maze_wrapper import (
    gen_empty_env,
    gen_four_rooms_env,
    gen_maze_with_obstacles,
)


def make_point_maze_wrapped_env(**kwargs):
    """
    Factory function to be used as entry_point in register calls.
    Creates a point maze environment and wraps it with PointMazeWrapper.
    """
    # Extract goal parameter if provided
    goal_state = kwargs.pop("goal_state", None)  # Single goal state
    goal_states = kwargs.pop("goal_states", None)  # Multiple goal states

    # Extract the base environment ID parameter
    base_env_id = kwargs.pop("base_env_id")

    # Create the base environment
    env = gym.make(base_env_id, **kwargs)

    # Create goal list based on what was provided
    if goal_states is not None:
        goals = PointMazeGoalList(goal_states)
    elif goal_state is not None:
        goals = PointMazeGoalList([goal_state])
    else:
        goals = None

    # Wrap the environment
    return PointMazeWrapper(env=env, goal=goals)


def point_maze_register():
    """Register all environment ID's to Gymnasium."""
    ### MAZE SPECIAL ENVS ###
    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        for width, height in [(11, 11)]:
            for start_x, start_y in [(1, 1)]:
                # Register Four Rooms with multiple goals
                base_env_id = f"PointMazeBase-FourRoomsEnv{suffix}-{width}x{height}-Goals-9x1-1x9-9x9"
                if base_env_id not in registry:
                    register(
                        id=base_env_id,
                        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                        kwargs={
                            "reward_type": reward_type,
                            "maze_map": gen_four_rooms_env(
                                width,
                                height,
                                [(start_x, start_y)],
                                [(1, 9), (9, 1), (9, 9)],
                            ),
                            "continuing_task": False,
                        },
                        max_episode_steps=900,
                    )

                # Register wrapped version
                env_id = (
                    f"PointMaze-FourRoomsEnv{suffix}-{width}x{height}-Goals-9x1-1x9-9x9"
                )
                if env_id not in registry:
                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs={
                            "base_env_id": base_env_id,
                            "goal_states": [(1, 9), (9, 1), (9, 9)],
                        },
                        max_episode_steps=900,
                    )

            # Loop through individual goals
            for goal_x, goal_y in [
                (1, 9),
                (9, 1),
                (9, 9),
                (7, 3),
                (3, 7),
                (6, 4),
                (4, 6),
                (3, 3),
                (6, 6),
                (4, 4),
                (3, 4),
                (7, 7),
                (6, 7),
                (8, 8),
                (7, 4),
                (4, 7),
                (6, 3),
                (3, 6),
                (5, 5),
                (5, 1),
                (1, 5),
                (8, 2),
                (2, 8),
                (4, 3),
            ]:
                # Register base Empty environment
                base_env_id = f"PointMazeBase-EmptyEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}"
                if base_env_id not in registry:
                    register(
                        id=base_env_id,
                        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                        kwargs={
                            "reward_type": reward_type,
                            "maze_map": gen_empty_env(
                                width, height, [(start_x, start_y)], [(goal_x, goal_y)]
                            ),
                            "continuing_task": False,
                        },
                        max_episode_steps=900,
                    )

                # Register wrapped Empty environment
                env_id = f"PointMaze-EmptyEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}"
                if env_id not in registry:
                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs={
                            "base_env_id": base_env_id,
                            "goal_state": (goal_x, goal_y),
                        },
                        max_episode_steps=900,
                    )

                # Register base Four Rooms environment
                base_env_id = f"PointMazeBase-FourRoomsEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}"
                if base_env_id not in registry:
                    register(
                        id=base_env_id,
                        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                        kwargs={
                            "reward_type": reward_type,
                            "maze_map": gen_four_rooms_env(
                                width, height, [(start_x, start_y)], [(goal_x, goal_y)]
                            ),
                            "continuing_task": False,
                        },
                        max_episode_steps=900,
                    )

                # Register wrapped Four Rooms environment
                env_id = f"PointMaze-FourRoomsEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}"
                if env_id not in registry:
                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs={
                            "base_env_id": base_env_id,
                            "goal_state": (goal_x, goal_y),
                        },
                        max_episode_steps=900,
                    )

                # Register base Obstacles environment
                base_env_id = f"PointMazeBase-ObstaclesEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}"
                if base_env_id not in registry:
                    register(
                        id=base_env_id,
                        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                        kwargs={
                            "reward_type": reward_type,
                            "maze_map": gen_maze_with_obstacles(
                                width,
                                height,
                                [(start_x, start_y)],
                                [(goal_x, goal_y)],
                                [
                                    (2, 2),
                                    (2, 3),
                                    (2, 4),
                                    (3, 2),
                                    (3, 3),
                                    (3, 4),
                                    (4, 2),
                                    (4, 3),
                                    (4, 4),
                                ],
                            ),
                            "continuing_task": False,
                        },
                        max_episode_steps=900,
                    )

                # Register wrapped Obstacles environment
                env_id = f"PointMaze-ObstaclesEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}"
                if env_id not in registry:
                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs={
                            "base_env_id": base_env_id,
                            "goal_state": (goal_x, goal_y),
                        },
                        max_episode_steps=900,
                    )

            # Register additional multi-goal environments
            multi_goal_sets = [
                [(1, 1), (9, 9), (5, 5)],  # Diagonal and center
                [(3, 3), (3, 7), (7, 3), (7, 7)],  # Four corners of inner area
                [(2, 2), (2, 8), (8, 2), (8, 8)],  # Four corners
            ]

            for maze_type in ["empty", "four_rooms", "obstacles"]:
                capitalized_maze_type = "".join(
                    [part.capitalize() for part in maze_type.split("_")]
                )

                for goal_set in multi_goal_sets:
                    # Create a string representation of the goals for the ID
                    goal_str = "-".join([f"{g[0]}x{g[1]}" for g in goal_set])

                    # Register base multi-goal environment
                    base_env_id = f"PointMazeBase-{capitalized_maze_type}Env{suffix}-{width}x{height}-MultiGoals-{goal_str}"
                    if base_env_id not in registry:
                        # Generate the appropriate maze map based on type
                        if maze_type == "obstacles":
                            obstacles_list = [
                                (2, 2),
                                (2, 3),
                                (2, 4),
                                (3, 2),
                                (3, 3),
                                (3, 4),
                                (4, 2),
                                (4, 3),
                                (4, 4),
                            ]
                            maze_map = gen_maze_with_obstacles(
                                width,
                                height,
                                [(start_x, start_y)],
                                goal_set,
                                obstacles_list,
                            )
                        elif maze_type == "four_rooms":
                            maze_map = gen_four_rooms_env(
                                width, height, [(start_x, start_y)], goal_set
                            )
                        else:  # empty
                            maze_map = gen_empty_env(
                                width, height, [(start_x, start_y)], goal_set
                            )

                        register(
                            id=base_env_id,
                            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                            kwargs={
                                "reward_type": reward_type,
                                "maze_map": maze_map,
                                "continuing_task": False,
                            },
                            max_episode_steps=900,
                        )

                    # Register wrapped multi-goal environment
                    env_id = f"PointMaze-{capitalized_maze_type}Env{suffix}-{width}x{height}-MultiGoals-{goal_str}"
                    if env_id not in registry:
                        register(
                            id=env_id,
                            entry_point=make_point_maze_wrapped_env,
                            kwargs={
                                "base_env_id": base_env_id,
                                "goal_states": goal_set,
                            },
                            max_episode_steps=900,
                        )

            # Register goal-conditioned environments for each maze type
            for maze_type in ["empty", "four_rooms", "obstacles"]:
                capitalized_maze_type = "".join(
                    [part.capitalize() for part in maze_type.split("_")]
                )

                # Generate all possible goal positions (excluding obstacles and edges)
                all_goal_states = []
                obstacles_list = [
                    (2, 2),
                    (2, 3),
                    (2, 4),
                    (3, 2),
                    (3, 3),
                    (3, 4),
                    (4, 2),
                    (4, 3),
                    (4, 4),
                ]

                for x in range(1, width - 1):
                    for y in range(1, height - 1):
                        pos = (x, y)
                        # Skip obstacles for obstacle environments
                        if maze_type == "obstacles" and pos in obstacles_list:
                            continue
                        all_goal_states.append(pos)

                # Register base goal-conditioned environment
                base_env_id = f"PointMazeBase-{capitalized_maze_type}Env{suffix}-{width}x{height}-GoalConditioned"
                if base_env_id not in registry:
                    # Generate maze map based on type
                    if maze_type == "obstacles":
                        maze_map = gen_maze_with_obstacles(
                            width,
                            height,
                            [(start_x, start_y)],
                            all_goal_states,
                            obstacles_list,
                        )
                    elif maze_type == "four_rooms":
                        maze_map = gen_four_rooms_env(
                            width, height, [(start_x, start_y)], all_goal_states
                        )
                    else:  # empty
                        maze_map = gen_empty_env(
                            width, height, [(start_x, start_y)], all_goal_states
                        )

                    register(
                        id=base_env_id,
                        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                        kwargs={
                            "reward_type": reward_type,
                            "maze_map": maze_map,
                            "continuing_task": False,
                        },
                        max_episode_steps=900,
                    )

                # Register wrapped goal-conditioned environment
                env_id = f"PointMaze-{capitalized_maze_type}Env{suffix}-{width}x{height}-GoalConditioned"
                if env_id not in registry:
                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs={
                            "base_env_id": base_env_id,
                            "goal_states": all_goal_states,
                        },
                        max_episode_steps=900,
                    )


# Register all environments
point_maze_register()
