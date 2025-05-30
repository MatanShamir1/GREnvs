import gymnasium
import numpy as np
from gymnasium.envs.registration import register
from gr_envs.wrappers import PandaGymWrapper

def make_panda_wrapped(**kwargs):
	assert "desired_goal" in kwargs, f"desired_goal must be in {kwargs=}"
	assert "action_space" in kwargs, f"action_space must be in {kwargs=}"

	desired_goal = kwargs.pop("desired_goal")
	action_space = kwargs.pop("action_space")
	env = gymnasium.make("PandaReach-v3", **kwargs)
	return PandaGymWrapper(env=env, desired_goal=desired_goal, action_space=action_space)


def panda_gym_register():
	for reward_type in ["sparse", "dense"]:
		for control_type in ["ee", "joints"]:
			reward_suffix = "Dense" if reward_type == "dense" else ""
			control_suffix = "Joints" if control_type == "joints" else ""

			register(
				id="PandaMyReach{}{}-v3".format(control_suffix, reward_suffix),
				entry_point="gr_envs.panda_scripts.envs:PandaReachEnv",
				kwargs={"reward_type": reward_type, "control_type": control_type},
				max_episode_steps=101,
			)

			reward_type = "dense"
			control_type = "ee"
			goals = [(-0.5, -0.5, 0.1), (-0.3, -0.3, 0.1), (-0.1, -0.1, 0.1), (-0.5, 0.2, 0.1), (-0.3, 0.2, 0.1), (-0.1, 0.1, 0.1), (0.2, -0.2, 0.1), (0.2, -0.3, 0.1), (0.1, -0.1, 0.1), (0.2, 0.2, 0.1), (0.0, 0.0, 0.1), (0.1, 0.1, 0.1)]
			reward_suffix = "Dense" if reward_type == "dense" else ""
			control_suffix = "Joints" if control_type == "joints" else ""
			for goal in goals:
				goal_str = 'X'.join([str(float(g)).replace(".", "y").replace("-","M") for g in goal])
				register(
					id="PandaMyReach{}{}X{}-v3".format(control_suffix, reward_suffix, goal_str),
					entry_point=make_panda_wrapped,
					kwargs={
						"reward_type": reward_type,
						"control_type": control_type,
						"desired_goal": np.array(goal, dtype=np.float32),
						"action_space": gymnasium.spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)
					},
					max_episode_steps=101,
				)
