from typing import List, Tuple, Optional
import numpy as np
import gymnasium
from gymnasium.envs.registration import register
from gymnasium_robotics.envs.maze.maps import R, G, C

from gr_envs.wrappers.goal_wrapper import GoalRecognitionWrapper, Goal


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


class PointMazeGoalList(Goal):
    """Goal representation for Point Maze environments."""

    def __init__(self, goal_states: List[Tuple[int, int]]):
        """
        Initialize a PointMazeGoalList with possible goal states.

        Args:
            goal_states: List of (x, y) coordinates representing goal positions in discrete grid cells
        """
        assert isinstance(goal_states, list) and all(
            isinstance(g, tuple) and len(g) == 2 for g in goal_states
        ), "goal_states must be a list of (x, y) coordinate tuples"

        self.goal_states = goal_states
        self.current_goal = self._sample_goal()
        self.current_goal_idx = 0

    def _sample_goal(self) -> Tuple[int, int]:
        """Sample a random goal from the list of goal states."""
        self.current_goal_idx = np.random.randint(len(self.goal_states))
        return self.goal_states[self.current_goal_idx]

    def get(self) -> Tuple[int, int]:
        """Return the current goal."""
        return self.current_goal

    def reset(self) -> None:
        """Reset by sampling a new goal."""
        self.current_goal = self._sample_goal()

    def is_in_subspace(self, goal: Tuple[int, int]) -> bool:
        """Check if a goal coordinate is in the allowed list of goals"""
        if not isinstance(goal, tuple) or len(goal) != 2:
            return False

        return goal in self.goal_states


class PointMazeWrapper(GoalRecognitionWrapper):
    """Wrapper for Point Maze environments to support dynamic goal recognition."""

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

    @staticmethod
    def goal_to_str(goal: Tuple[int, int]) -> str:
        """Convert a goal position to a string representation."""
        return f"{goal[0]}x{goal[1]}"

    def is_goal_in_subspace(self, goal: Tuple[int, int]) -> bool:
        """Check if a goal is within this wrapper's goal subspace"""
        if self.goal is None:
            return True
        return self.goal.is_in_subspace(goal)
