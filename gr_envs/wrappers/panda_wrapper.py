from .goal_wrapper import GoalRecognitionWrapper
from .hook import get_property_reference
import highway_env
import numpy as np

class PandaGymWrapper(GoalRecognitionWrapper):
	GOAL_DIMENSION_SHAPE = (3,)
	GOAL_DTYPE = np.float32
	HOOK_FUNC = "_sample_goal"
	def __init__(self, env, desired_goal):
		super().__init__(env, name="panda")

		assert desired_goal.dtype == PandaGymWrapper.GOAL_DTYPE
		assert desired_goal.shape == PandaGymWrapper.GOAL_DIMENSION_SHAPE

		hooked_env = get_property_reference(env, PandaGymWrapper.HOOK_FUNC)
		setattr(
			hooked_env,
			PandaGymWrapper.HOOK_FUNC,
			lambda: np.array(desired_goal, dtype=self.env.observation_space['desired_goal'].dtype)
		)
