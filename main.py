import gymnasium
import numpy as np

from gr_envs.wrappers import PandaGymWrapper

def main():
	env = gymnasium.make("PandaReachDense-v3")
	print(env.reset())

	desired_goal = np.array([0.1, 0.1, 0.1], dtype=np.float32)

	wrapped_env = PandaGymWrapper(gymnasium.make("PandaReachDense-v3"), desired_goal=desired_goal)
	assert np.array_equal(wrapped_env.reset()[0]['desired_goal'], desired_goal)

	for _ in range(10):
		obs, reward, terminated, truncated, info = wrapped_env.step(wrapped_env.action_space.sample())
		assert np.array_equal(obs['desired_goal'], desired_goal)


if __name__ == "__main__":
	main()