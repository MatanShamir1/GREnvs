from typing import Optional

import gymnasium
from gymnasium.spaces import Box

from gr_envs.wrappers.goal_wrapper import GoalRecognitionWrapper, Goal
from .hook import get_property_reference
import numpy as np


class PandaGymDesiredGoalList(Goal):
    def __init__(self, goals: Box, goal_array=None):
        assert (
            type(goals) == Box
        ), "PandaGymDesiredGoalList expects a Box space for goals."
        self.goals = goals
        self.current_goal = self.goals.sample()
        # Store explicit goal array if provided
        self.goal_array = goal_array

    def get(self) -> np.ndarray:
        return self.current_goal

    def reset(self) -> None:
        self.current_goal = self.goals.sample()

    def is_in_subspace(self, goal: np.ndarray) -> bool:
        """Check if a goal is within the defined subspace"""
        goal = np.asarray(goal, dtype=self.goals.dtype)
        # First check if we have explicit goal list
        if self.goal_array is not None:
            return any(
                np.allclose(goal, g, rtol=1e-2, atol=1e-2) for g in self.goal_array
            )

        # Otherwise check if it's within the Box constraints
        goal_flat = goal.flatten()
        return all(
            low <= val <= high
            for val, low, high in zip(
                goal_flat, self.goals.low.flatten(), self.goals.high.flatten()
            )
        )


class PandaGymWrapper(GoalRecognitionWrapper):
    GOAL_DIMENSION_SHAPE = (3,)
    GOAL_DTYPE = np.float32
    HOOK_FUNC = "_sample_goal"

    def __init__(
        self,
        env: gymnasium.Env,
        goal: Optional[PandaGymDesiredGoalList],
        action_space: Box,
    ):
        super().__init__(env, name="panda", goal=goal)

        hooked_env = get_property_reference(env, PandaGymWrapper.HOOK_FUNC)
        setattr(hooked_env, PandaGymWrapper.HOOK_FUNC, lambda: self._reset_goals())

        hooked_robot_env = get_property_reference(env, "robot")
        setattr(hooked_robot_env.robot, "action_space", action_space)
        setattr(hooked_robot_env, "action_space", action_space)
        self._hooked_robot_env = hooked_robot_env

    def _reset_goals(self) -> np.ndarray:
        if self.goal is None:
            goal = self.env.np_random.uniform(
                self._hooked_robot_env.task.goal_range_low,
                self._hooked_robot_env.task.goal_range_high,
            )
            return goal
        else:
            self.goal.reset()
            return np.array(
                self.goal.get(), dtype=self.env.observation_space["desired_goal"].dtype
            )

    @staticmethod
    def goal_to_str(goal: np.ndarray) -> str:
        goal_str = "X".join(
            [f"{float(g):.3g}".replace(".", "y").replace("-", "M") for g in goal]
        )
        print(f"Goal string: {goal_str}")
        return goal_str

    def is_goal_in_subspace(self, goal: np.ndarray) -> bool:
        """Check if a goal is within this wrapper's goal subspace"""
        if self.goal is None:
            return True
        return self.goal.is_in_subspace(goal)
