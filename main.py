import gymnasium
import numpy as np

from gr_envs.wrappers import PandaGymWrapper
from gr_envs.wrappers.parking_wrapper import ParkingWrapper
from matplotlib import pyplot as plt

def main():
	# PANDA
	desired_goal = np.array([0.1, 0.1, 0.1], dtype=np.float32)

	wrapped_env = PandaGymWrapper(gymnasium.make("PandaReachDense-v3"), desired_goal=desired_goal)

	assert np.array_equal(wrapped_env.reset()[0]['desired_goal'], desired_goal)

	for _ in range(10):
		obs, reward, terminated, truncated, info = wrapped_env.step(wrapped_env.action_space.sample())
		assert np.array_equal(obs['desired_goal'], desired_goal), f"{desired_goal}, {obs['desired_goal']}"

	# Parking
	wrapped_env = ParkingWrapper(
		gymnasium.make("parking-v0", render_mode='rgb_array'),
		n_spots=15,
		goal_index=2,
		heading=np.pi,
		parked_cars=[1, 5, 13, 14, 15, 16]
	)
	wrapped_env.reset()
	plt.imshow(wrapped_env.render())
	plt.show()


if __name__ == "__main__":
	main()