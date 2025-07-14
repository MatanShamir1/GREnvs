import gymnasium
from gymnasium.envs.registration import register, registry

from gr_envs.wrappers.point_maze_wrapper import (
    PointMazeWrapper,
    PointMazeGoalList,
    gen_empty_env,
    gen_four_rooms_env,
    gen_maze_with_obstacles,
)


def make_point_maze_wrapped_env(**kwargs):
    """Factory function for creating a wrapped point maze environment."""
    # Extract maze parameters
    maze_type = kwargs.pop("maze_type", "empty")
    env_id = kwargs.pop("env_id", "PointMazeEnv")
    entry_point = kwargs.pop(
        "entry_point", "gymnasium_robotics.envs.maze.point_maze:PointMazeEnv"
    )
    width = kwargs.pop("width", 11)
    height = kwargs.pop("height", 11)
    initial_states = kwargs.pop("initial_states", [(1, 1)])
    reward_type = kwargs.pop("reward_type", "sparse")
    continuing_task = kwargs.pop("continuing_task", False)
    max_episode_steps = kwargs.pop("max_episode_steps", 900)
    render_mode = kwargs.pop("render_mode", None)

    # Extract goal parameters
    goal_state = kwargs.pop("goal_state", None)  # Single goal state
    goal_states = kwargs.pop("goal_states", None)  # Multiple goal states

    # Default obstacles for obstacle maze type
    default_obstacles = [
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
    obstacles = kwargs.pop("obstacles", default_obstacles)

    # Determine maze generation function based on maze type
    if maze_type == "empty":
        maze_map_func = gen_empty_env
    elif maze_type == "four_rooms":
        maze_map_func = gen_four_rooms_env
    elif maze_type == "obstacles":
        maze_map_func = gen_maze_with_obstacles
    else:
        raise ValueError(f"Unknown maze type: {maze_type}")

    # Create goal list from provided goals
    if goal_states is not None:
        all_goals = goal_states
    elif goal_state is not None:
        all_goals = [goal_state]
    else:
        all_goals = [(9, 9)]  # Default goal if none specified

    # Generate maze map
    if maze_type == "obstacles":
        maze_map = maze_map_func(width, height, initial_states, all_goals, obstacles)
    else:
        maze_map = maze_map_func(width, height, initial_states, all_goals)

    # Create environment kwargs
    env_kwargs = {
        "reward_type": reward_type,
        "maze_map": maze_map,
        "continuing_task": continuing_task,
    }

    if render_mode:
        env_kwargs["render_mode"] = render_mode

    register(
        id=f"Base{env_id}",
        entry_point=entry_point,
        kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
    )

    # Create the base environment
    env = gymnasium.make(
        f"Base{env_id}",
        **env_kwargs,
    )

    # Create goal list for the wrapper
    goals = PointMazeGoalList(all_goals)

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
                env_id = (
                    f"PointMaze-FourRoomsEnv{suffix}-{width}x{height}-Goals-9x1-1x9-9x9"
                )
                if env_id not in registry:
                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs={
                            "env_id": env_id,
                            "entry_point": "gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                            "maze_type": "four_rooms",
                            "width": width,
                            "height": height,
                            "initial_states": [(start_x, start_y)],
                            "goal_states": [(1, 9), (9, 1), (9, 9)],
                            "reward_type": reward_type,
                            "continuing_task": False,
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
                # Register Empty environment
                env_id = f"PointMaze-EmptyEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}"
                if env_id not in registry:
                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs={
                            "maze_type": "empty",
                            "env_id": env_id,
                            "entry_point": "gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                            "width": width,
                            "height": height,
                            "initial_states": [(start_x, start_y)],
                            "goal_state": (goal_x, goal_y),
                            "reward_type": reward_type,
                            "continuing_task": False,
                        },
                        max_episode_steps=900,
                    )

                # Register Four Rooms environment
                env_id = f"PointMaze-FourRoomsEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}"
                if env_id not in registry:
                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs={
                            "maze_type": "four_rooms",
                            "env_id": env_id,
                            "entry_point": "gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                            "width": width,
                            "height": height,
                            "initial_states": [(start_x, start_y)],
                            "goal_state": (goal_x, goal_y),
                            "reward_type": reward_type,
                            "continuing_task": False,
                        },
                        max_episode_steps=900,
                    )

                # Register Obstacles environment
                env_id = f"PointMaze-ObstaclesEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}"
                if env_id not in registry:
                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs={
                            "maze_type": "obstacles",
                            "env_id": env_id,
                            "entry_point": "gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                            "width": width,
                            "height": height,
                            "initial_states": [(start_x, start_y)],
                            "goal_state": (goal_x, goal_y),
                            "reward_type": reward_type,
                            "continuing_task": False,
                            "obstacles": [
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
                        },
                        max_episode_steps=900,
                    )

            # Register additional multi-goal environments
            multi_goal_sets = [
                [(2, 2), (9, 9), (5, 5)],  # Diagonal and center
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

                    # Register multi-goal environment
                    env_id = f"PointMaze-{capitalized_maze_type}Env{suffix}-{width}x{height}-MultiGoals-{goal_str}"
                    if env_id not in registry:
                        kwargs = {
                            "maze_type": maze_type,
                            "env_id": env_id,
                            "entry_point": "gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                            "width": width,
                            "height": height,
                            "initial_states": [(start_x, start_y)],
                            "goal_states": goal_set,
                            "reward_type": reward_type,
                            "continuing_task": False,
                        }

                        if maze_type == "obstacles":
                            kwargs["obstacles"] = [
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

                        register(
                            id=env_id,
                            entry_point=make_point_maze_wrapped_env,
                            kwargs=kwargs,
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
                        if x == start_x and y == start_y:
                            continue
                        all_goal_states.append(pos)

                # Register goal-conditioned environment
                env_id = f"PointMaze-{capitalized_maze_type}Env{suffix}-{width}x{height}"
                if env_id not in registry:
                    kwargs = {
                        "maze_type": maze_type,
                        "env_id": env_id,
                        "width": width,
                        "height": height,
                        "initial_states": [(start_x, start_y)],
                        "goal_states": all_goal_states,
                        "reward_type": reward_type,
                        "continuing_task": True,  # Goal-conditioned environments are continuing
                    }

                    if maze_type == "obstacles":
                        kwargs["obstacles"] = obstacles_list

                    register(
                        id=env_id,
                        entry_point=make_point_maze_wrapped_env,
                        kwargs=kwargs,
                        max_episode_steps=900,
                    )


# Register all environments
point_maze_register()
