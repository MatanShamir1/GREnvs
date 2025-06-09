from typing import List, Tuple, Optional

import gymnasium
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.error import NameNotFound

from gr_envs.wrappers.goal_wrapper import GoalRecognitionWrapper, Goal
from gr_envs.env_registration.point_maze_register import (
    gen_empty_env,
    gen_four_rooms_env,
    gen_maze_with_obstacles,
)


class PointMazeGoalList(Goal):
    """Goal representation for Point Maze environments."""

    def __init__(self, goal_states: List[Tuple[int, int]]):
        """
        Initialize a PointMazeGoalList with possible goal states.

        Args:
            goal_states: List of (x, y) coordinates representing goal positions
        """
        assert isinstance(goal_states, list) and all(
            isinstance(g, tuple) and len(g) == 2 for g in goal_states
        ), "goal_states must be a list of (x, y) coordinate tuples"

        self.goal_states = goal_states
        self.current_goal = self._sample_goal()

    def _sample_goal(self) -> Tuple[int, int]:
        """Sample a random goal from the list of goal states."""
        return self.goal_states[np.random.randint(len(self.goal_states))]

    def get(self) -> Tuple[int, int]:
        """Return the current goal."""
        return self.current_goal

    def reset(self) -> None:
        """Reset by sampling a new goal."""
        self.current_goal = self._sample_goal()


class PointMazeWrapper(GoalRecognitionWrapper):
    """Wrapper for Point Maze environments to support goal recognition."""

    def __init__(
        self,
        env: gymnasium.Env,
        goal: Optional[PointMazeGoalList] = None,
    ):
        """
        Initialize a Point Maze wrapper.

        Args:
            env: Base environment to wrap
            goal: Optional PointMazeGoalList with goal states
        """
        super().__init__(env, name="point_maze", goal=goal)

    def step(self, action):
        """Step the environment."""
        return self.env.step(action)

    @staticmethod
    def goal_to_str(goal: Tuple[int, int]) -> str:
        """Convert a goal position to a string representation."""
        return f"{goal[0]}x{goal[1]}"


def register_multi_goal_maze_env(
    maze_type: str,
    width: int,
    height: int,
    initial_states: List[Tuple[int, int]],
    goal_states: List[Tuple[int, int]],
    obstacles: List[Tuple[int, int]] = None,
    reward_type: str = "sparse",
    max_episode_steps: int = 900,
    continuing_task: bool = False,
    **kwargs,
):
    """Register a multi-goal point maze environment."""
    suffix = "Dense" if reward_type == "dense" else ""

    # Convert goal_states to string for ID
    goal_str = "-".join([f"{g[0]}x{g[1]}" for g in goal_states])

    # Create the environment ID with multiple goals in the name
    capitalized_maze_type = "".join(
        [part.capitalize() for part in maze_type.split("_")]
    )
    env_id = f"PointMaze-{capitalized_maze_type}Env{suffix}-{width}x{height}-MultiGoals-{goal_str}"

    # Select the maze generation function
    if maze_type == "empty":
        maze_map_func = gen_empty_env
    elif maze_type == "four_rooms":
        maze_map_func = gen_four_rooms_env
    elif maze_type == "obstacles":
        maze_map_func = gen_maze_with_obstacles
        if not obstacles:
            obstacles = []
    else:
        raise ValueError(f"Unknown maze type: {maze_type}")

    # Generate map function arguments
    if maze_type == "obstacles":
        maze_map_args = (width, height, initial_states, goal_states, obstacles)
    else:
        maze_map_args = (width, height, initial_states, goal_states)

    # Prepare kwargs for environment creation
    env_kwargs = {
        "reward_type": reward_type,
        "maze_map": maze_map_func(*maze_map_args),
        "continuing_task": continuing_task,
        **kwargs,  # Include any additional kwargs
    }

    # Register the environment
    register(
        id=env_id,
        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
        kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
    )

    return env_id


def register_goal_conditioned_maze_env(
    maze_type: str,
    width: int,
    height: int,
    initial_states: List[Tuple[int, int]],
    obstacles: List[Tuple[int, int]] = None,
    reward_type: str = "sparse",
    max_episode_steps: int = 900,
    continuing_task: bool = False,
    **kwargs,
):
    """Register a goal-conditioned point maze environment (goals can be anywhere except obstacles)."""
    suffix = "Dense" if reward_type == "dense" else ""

    # Create the environment ID for goal-conditioned environment
    capitalized_maze_type = "".join(
        [part.capitalize() for part in maze_type.split("_")]
    )
    env_id = (
        f"PointMaze-{capitalized_maze_type}Env{suffix}-{width}x{height}-GoalConditioned"
    )

    # Generate all possible goal positions (excluding obstacles and edges)
    all_goal_states = []
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            pos = (x, y)
            # Skip obstacles
            if obstacles and pos in obstacles:
                continue
            # Add as a potential goal
            all_goal_states.append(pos)

    # Select the maze generation function
    if maze_type == "empty":
        maze_map_func = gen_empty_env
    elif maze_type == "four_rooms":
        maze_map_func = gen_four_rooms_env
    elif maze_type == "obstacles":
        maze_map_func = gen_maze_with_obstacles
        if not obstacles:
            obstacles = []
    else:
        raise ValueError(f"Unknown maze type: {maze_type}")

    # Generate map function arguments
    if maze_type == "obstacles":
        maze_map_args = (width, height, initial_states, all_goal_states, obstacles)
    else:
        maze_map_args = (width, height, initial_states, all_goal_states)

    # Prepare kwargs for environment creation
    env_kwargs = {
        "reward_type": reward_type,
        "maze_map": maze_map_func(*maze_map_args),
        "continuing_task": continuing_task,
        **kwargs,  # Include any additional kwargs
    }

    # Register the environment
    register(
        id=env_id,
        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
        kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
    )

    return env_id


def make_point_maze_wrapped(**kwargs):
    """Factory function to create and wrap a point maze environment."""
    # Extract parameters
    env_id = kwargs.pop("env_id", None)
    width = kwargs.pop("width", 11)
    height = kwargs.pop("height", 11)
    initial_states = kwargs.pop("initial_states", [(1, 1)])
    goal_states = kwargs.pop("goal_states", [(9, 9)])
    obstacles = kwargs.pop("obstacles", [])
    maze_type = kwargs.pop("maze_type", "empty")
    reward_type = kwargs.pop("reward_type", "sparse")
    goal_conditioned = kwargs.pop("goal_conditioned", False)
    max_episode_steps = kwargs.pop("max_episode_steps", 900)
    continuing_task = kwargs.pop("continuing_task", False)

    # Keep remaining kwargs to pass to environment creation
    remaining_kwargs = kwargs

    # Check if an env_id was provided, otherwise generate one
    if env_id is None:
        if goal_conditioned:
            # Register a goal-conditioned environment (goals can be anywhere)
            try:
                capitalized_maze_type = "".join(
                    [part.capitalize() for part in maze_type.split("_")]
                )
                env_id = f"PointMaze-{capitalized_maze_type}Env-{width}x{height}-GoalConditioned"
                env = gymnasium.make(env_id, **remaining_kwargs)
            except (NameNotFound, KeyError):
                print(
                    f"Environment {env_id} not found, registering goal-conditioned environment."
                )
                env_id = register_goal_conditioned_maze_env(
                    maze_type=maze_type,
                    width=width,
                    height=height,
                    initial_states=initial_states,
                    obstacles=obstacles,
                    reward_type=reward_type,
                    max_episode_steps=max_episode_steps,
                    continuing_task=continuing_task,
                    **remaining_kwargs,
                )
                env = gymnasium.make(env_id, **remaining_kwargs)
        else:
            # Register a multi-goal environment with specific goals
            try:
                # Create a descriptive ID for multiple goals
                goal_str = "-".join([f"{g[0]}x{g[1]}" for g in goal_states])
                capitalized_maze_type = "".join(
                    [part.capitalize() for part in maze_type.split("_")]
                )
                env_id = f"PointMaze-{capitalized_maze_type}Env-{width}x{height}-MultiGoals-{goal_str}"
                env = gymnasium.make(env_id, **remaining_kwargs)
            except (NameNotFound, KeyError):
                print(
                    f"Environment {env_id} not found, registering multi-goal environment."
                )
                env_id = register_multi_goal_maze_env(
                    maze_type=maze_type,
                    width=width,
                    height=height,
                    initial_states=initial_states,
                    goal_states=goal_states,
                    obstacles=obstacles,
                    reward_type=reward_type,
                    max_episode_steps=max_episode_steps,
                    continuing_task=continuing_task,
                    **remaining_kwargs,
                )
                env = gymnasium.make(env_id, **remaining_kwargs)
    else:
        # Use the provided env_id
        try:
            env = gymnasium.make(env_id, **remaining_kwargs)
        except (NameNotFound, KeyError):
            print(f"Environment {env_id} not found, using default environment.")
            # Fall back to a default environment
            env_id = "PointMaze-EmptyEnv-11x11-Goal-9x9"
            env = gymnasium.make(env_id, **remaining_kwargs)

    # Create goal list
    goals = PointMazeGoalList(goal_states)

    # Create and return wrapped environment
    return PointMazeWrapper(
        env=env,
        goal=goals,
    )
