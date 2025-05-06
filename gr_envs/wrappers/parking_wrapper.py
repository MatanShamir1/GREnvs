from .goal_wrapper import GoalRecognitionWrapper
from .hook import get_property_reference

from typing import Optional, List
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
from highway_env.envs import ParkingEnv
import highway_env
import numpy as np


class ParkingWrapper(GoalRecognitionWrapper):
	def __init__(self, env: ParkingEnv, n_spots: int, goal_index: Optional[int], heading: Optional[float] = None, parked_cars: List[int] = None):
		super().__init__(env, name="parking")
		self.n_spots = n_spots
		self.goal_index = goal_index
		self.heading = heading
		self.parked_cars = parked_cars

		assert 0 <= self.goal_index < self.n_spots, f"Goal index {self.goal_index} is out of range [0, {self.n_spots}]"
		assert parked_cars is None or not any(
			(parked_car == self.goal_index or parked_car < 0 or parked_car >= 2 * self.n_spots)
			for parked_car in self.parked_cars
		), f"Parked car is invalid as all parked cars ({self.parked_cars}) must be in a valid spot"

		hooked_env = get_property_reference(env, "_create_road")

		setattr(hooked_env, "_reset", lambda : self.parking_env_reset(hooked_env))
		setattr(hooked_env, "_create_vehicles", lambda : self.parking_env_create_vehicles(hooked_env))

		hooked_env.reset()

	def parking_env_reset(self, hooked_env: ParkingEnv):
		hooked_env._create_road(spots=self.n_spots)
		hooked_env._create_vehicles()

	def _set_goal_parking(self, hooked_env: ParkingEnv):
		lanes = hooked_env.road.network.lanes_list()

		if len(hooked_env.road.network.lanes_list()) > self.goal_index >= 0:  # in case of a goal-directed agent that has a specific goal_index, it is assigned here and added as a landmark.
			goal_idx = self.goal_index
		else:
			goal_idx = hooked_env.np_random.choice(len(lanes))

		lane = lanes[goal_idx]

		self.goal = Landmark(hooked_env.road, lane.position(lane.length / 2, 0), heading=lane.heading)
		for vehicle in hooked_env.controlled_vehicles:
			if hasattr(vehicle, 'goal'):
				pass
			else:
				vehicle.goal = self.goal
		hooked_env.road.objects.append(self.goal)  # depleted after every step, need to add the goal again at every step.

	def _get_heading(self, hooked_env: ParkingEnv) -> float:
		return self.heading if self.heading else 2 * np.pi * hooked_env.np_random.uniform()

	def _set_parked_cars(self, hooked_env: ParkingEnv) -> None:
		for parked_vehicle_spot in self.parked_cars:
			lane = ("a", "b", parked_vehicle_spot) if parked_vehicle_spot < self.n_spots else ("b", "c", self.n_spots - parked_vehicle_spot - 1)
			v = Vehicle.make_on_lane(hooked_env.road, lane, 4, speed=0)
			hooked_env.road.vehicles.append(v)

	def parking_env_create_vehicles(self, hooked_env: ParkingEnv) -> None:
		"""Create some new random vehicles of a given type, and add them on the road."""
		empty_spots = list(hooked_env.road.network.lanes_dict().keys())

		# Controlled vehicles
		hooked_env.controlled_vehicles = []
		for i in range(hooked_env.config["controlled_vehicles"]):
			x0 = (i - hooked_env.config["controlled_vehicles"] // 2) * 10
			vehicle = hooked_env.action_type.vehicle_class(
				hooked_env.road, [x0, 0],
				self._get_heading(hooked_env),
				0
			)
			vehicle.color = VehicleGraphics.EGO_COLOR
			hooked_env.road.vehicles.append(vehicle)
			hooked_env.controlled_vehicles.append(vehicle)
			empty_spots.remove(vehicle.lane_index)

		# hooks
		self._set_goal_parking(hooked_env)
		self._set_parked_cars(hooked_env)

		# Other vehicles
		for i in range(hooked_env.config["vehicles_count"]):
			if not empty_spots:
				continue
			lane_index = empty_spots[hooked_env.np_random.choice(np.arange(len(empty_spots)))]
			v = Vehicle.make_on_lane(hooked_env.road, lane_index, 4, speed=0)
			hooked_env.road.vehicles.append(v)
			empty_spots.remove(lane_index)

		# Walls
		if hooked_env.config["add_walls"]:
			width, height = 70, 42
			for y in [-height / 2, height / 2]:
				obstacle = Obstacle(hooked_env.road, [0, y])
				obstacle.LENGTH, obstacle.WIDTH = (width, 1)
				obstacle.diagonal = np.sqrt(obstacle.LENGTH ** 2 + obstacle.WIDTH ** 2)
				hooked_env.road.objects.append(obstacle)
			for x in [-width / 2, width / 2]:
				obstacle = Obstacle(hooked_env.road, [x, 0], heading=np.pi / 2)
				obstacle.LENGTH, obstacle.WIDTH = (height, 1)
				obstacle.diagonal = np.sqrt(obstacle.LENGTH ** 2 + obstacle.WIDTH ** 2)
				hooked_env.road.objects.append(obstacle)
	# HOOK create_vehicles and create goal and controlled vehicles



