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

    def is_goal_in_subspace(self, goal) -> bool:
        """
        Check if a goal is within this wrapper's goal subspace

        Default implementation returns True if no goal constraints are defined
        """
        if self.goal is None:
            return True

        if hasattr(self.goal, "is_in_subspace"):
            return self.goal.is_in_subspace(goal)

        return True
