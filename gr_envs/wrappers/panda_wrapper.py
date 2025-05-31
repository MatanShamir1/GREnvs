import random

import gymnasium
from gymnasium.spaces import Box

from gr_envs.wrappers.goal_wrapper import GoalRecognitionWrapper, Goal
from .hook import get_property_reference
import numpy as np

class PandaGymDesiredGoalList(Goal):
	def __init__(self, goals: list[np.ndarray]):
		assert type(goals) == list, f"goals should be with list type, not {type(goals)}"
		self.goals = goals
		self.current_goal = random.choice(self.goals)

	def get(self) -> np.ndarray:
		return self.current_goal

	def reset(self) -> None:
		self.current_goal = random.choice(self.goals)


class PandaGymWrapper(GoalRecognitionWrapper):
	GOAL_DIMENSION_SHAPE = (3,)
	GOAL_DTYPE = np.float32
	HOOK_FUNC = "_sample_goal"
	def __init__(self, env: gymnasium.Env, goal: PandaGymDesiredGoalList, action_space: Box):
		super().__init__(env, name="panda", goal=goal)

		hooked_env = get_property_reference(env, PandaGymWrapper.HOOK_FUNC)
		setattr(
			hooked_env,
			PandaGymWrapper.HOOK_FUNC,
			lambda: self._reset_goals()
		)

		hooked_robot_env = get_property_reference(env, "robot")
		setattr(
			hooked_robot_env.robot,
			"action_space",
			action_space
		)
		setattr(
			hooked_robot_env,
			"action_space",
			action_space
		)
	def _reset_goals(self) -> np.ndarray:
		self.goal.reset()
		return np.array(self.goal.get(), dtype=self.env.observation_space['desired_goal'].dtype)

	@staticmethod
	def goal_to_str(goal: np.ndarray) -> str:
		goal_str = "X".join(
			[f"{float(g):.3g}".replace(".", "y").replace("-", "M") for g in goal]
		)
		print(f"Goal string: {goal_str}")
		return goal_str
