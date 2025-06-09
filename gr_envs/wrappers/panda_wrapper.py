from typing import Optional

import gymnasium
from gymnasium.spaces import Box

from gr_envs.wrappers.goal_wrapper import GoalRecognitionWrapper, Goal
from .hook import get_property_reference
import numpy as np


class PandaGymDesiredGoalList(Goal):
    def __init__(self, goals: Box):
        assert (
            type(goals) == Box
        ), "PandaGymDesiredGoalList expects a Box space for goals."
        self.goals = goals
        self.current_goal = self.goals.sample()

    def get(self) -> np.ndarray:
        return self.current_goal

    def reset(self) -> None:
        self.current_goal = self.goals.sample()


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
