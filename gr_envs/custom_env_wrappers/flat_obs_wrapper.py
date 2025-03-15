import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np

class CombineAchievedGoalAndObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CombineAchievedGoalAndObservationWrapper, self).__init__(env)
        
        # Combine 'achieved_goal' dict and 'observation' box
        achieved_goal_space = env.observation_space['achieved_goal']
        desired_goal_space = env.observation_space['desired_goal']
        observation_space = env.observation_space['observation']
        
        # Add 'observation' space to 'achieved_goal' dict
        combined_spaces = {**achieved_goal_space.spaces, **desired_goal_space.spaces, 'observation': observation_space}
        
        # Set the new combined observation space
        self.observation_space = Dict(combined_spaces)
        self.inner_env = env
        self.is_success_once = False

    # this is called from the observationWrapper parent class only if it is used, and right after 'reset', to change the observation. no need to call the parent class func.
    def observation(self, observation):
        # Add the 'observation' to the 'achieved_goal' dict in the observation
        combined_observation = {**observation['achieved_goal'], 'observation': observation['observation']}
        return combined_observation
    
    # need to override the wrapper class's implementation of 'step' exactly the same, and add the required functionality.
    def step(self, action):
        observation, reward, terminated, truncated, info = self.inner_env.step(action)
        if "success" in info.keys(): self.is_success_once |= info["success"]
        elif "is_success" in info.keys(): self.is_success_once |= info["is_success"]
        elif "step_task_completions" in info.keys(): self.is_success_once |= (len(info["step_task_completions"]) == 1)
        return self.observation(observation), reward, terminated, truncated, info
        