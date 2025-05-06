import gymnasium
import panda_gym
import numpy as np

class PandaGymWrapper(gymnasium.Wrapper):
	def __init__(self, env, desired_goal):
		super().__init__(env)
		self.desired_goal = np.array(desired_goal, dtype=self.env.observation_space['desired_goal'].dtype)

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		obs['desired_goal'] = self.desired_goal.copy()
		return obs, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		obs['desired_goal'] = self.desired_goal.copy()
		return obs, reward, terminated, truncated, info
