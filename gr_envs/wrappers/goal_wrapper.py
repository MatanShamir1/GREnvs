from __future__ import annotations
import gymnasium

class Goal:
	def get(self) -> int:
		raise NotImplementedError()

	def reset(self) -> None:
		raise NotImplementedError()

class GoalRecognitionWrapper(gymnasium.Wrapper):
	def __init__(self, env: gymnasium.Env, name: str, goal: Goal):
		super().__init__(env)
		self.name = name
		self.goal = goal
