import gymnasium as gym
from gymnasium_robotics.envs.maze.maps import R, G, C
from gymnasium.envs.registration import register


def gen_empty_env(width, height, initial_states, goal_states):
    # Create an empty environment matrix with walls (1) around the edges
    env = [
        [
            1 if x == 0 or x == width - 1 or y == 0 or y == height - 1 else 0
            for x in range(width)
        ]
        for y in range(height)
    ]

    # Place initial states (R) and goal states (G)
    for x, y in initial_states:
        if 0 < x < width - 1 and 0 < y < height - 1 and env[y][x] == 0:
            env[y][x] = R
    for x, y in goal_states:
        if 0 < x < width - 1 and 0 < y < height - 1:
            if env[y][x] == 0:
                env[y][x] = G
            elif env[y][x] == R:
                env[y][x] = C

    return env


def gen_four_rooms_env(width, height, initial_states, goal_states):
    # Create an empty environment matrix with walls (1) around the edges
    env = [
        [
            1 if x == 0 or x == width - 1 or y == 0 or y == height - 1 else 0
            for x in range(width)
        ]
        for y in range(height)
    ]

    # Add walls for the four rooms structure
    for y in range(1, height - 1):
        env[y][width // 2] = 1 if y != height // 4 and y != height * 3 // 4 else 0
    for x in range(1, width - 1):
        env[height // 2][x] = 1 if x != width // 4 and x != width * 3 // 4 else 0

    # Place initial states (R) and goal states (G)
    for x, y in initial_states:
        if 0 < x < width - 1 and 0 < y < height - 1 and env[y][x] == 0:
            env[y][x] = R
    for x, y in goal_states:
        if 0 < x < width - 1 and 0 < y < height - 1:
            if env[y][x] == 0:
                env[y][x] = G
            elif env[y][x] == R:
                env[y][x] = C

    return env


def gen_maze_with_obstacles(width, height, initial_states, goal_states, obstacles):
    # Create an empty environment matrix with walls (1) around the edges
    env = [
        [
            1 if x == 0 or x == width - 1 or y == 0 or y == height - 1 else 0
            for x in range(width)
        ]
        for y in range(height)
    ]

    # Place obstacles (1)
    for x, y in obstacles:
        env[y][x] = 1

    # Place initial states (R) and goal states (G)
    for x, y in initial_states:
        if 0 < x < width - 1 and 0 < y < height - 1 and env[y][x] == 0:
            env[y][x] = R
    for x, y in goal_states:
        if 0 < x < width - 1 and 0 < y < height - 1:
            if env[y][x] == 0:
                env[y][x] = G
            elif env[y][x] == R:
                env[y][x] = C

    return env


def point_maze_register():
    """Register all environment ID's to Gymnasium."""
    ### MAZE SPECIAL ENVS ###
    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        for width, height in [(11, 11)]:
            for start_x, start_y in [(1, 1)]:
                register(
                    id=f"PointMaze-FourRoomsEnv{suffix}-{width}x{height}-Goals-9x1-1x9-9x9",
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
                (3, 4),
                (4, 3),
            ]:
                register(
                    id=f"PointMaze-EmptyEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}",
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
                register(
                    id=f"PointMaze-FourRoomsEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}",
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
                register(
                    id=f"PointMaze-ObstaclesEnv{suffix}-{width}x{height}-Goal-{goal_x}x{goal_y}",
                    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
                    kwargs={
                        "reward_type": reward_type,
                        "maze_map": gen_maze_with_obstacles(
                            11,
                            11,
                            [(1, 1)],
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
