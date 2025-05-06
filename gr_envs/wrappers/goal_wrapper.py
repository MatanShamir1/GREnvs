from abc import abstractmethod
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium
import panda_gym

class GoalRecognitionWrapper(gymnasium.Wrapper):
	def __init__(self, env: gymnasium.Env, name: str):
		super().__init__(env)
		self.name = name
