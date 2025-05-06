from .goal_wrapper import GoalRecognitionWrapper
from .hook import get_property_reference

from typing import Optional, List, Any
from gymnasium.core import WrapperObsType
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark
from highway_env.envs import ParkingEnv
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
		self.reset()

	def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
		self.env.reset()
		self._hook()

	def _hook(self):
		hooked_env = get_property_reference(self.env, "controlled_vehicles")
		hooked_env.road.objects = [item for item in hooked_env.road.objects if not isinstance(item, Landmark)]

		assert len(hooked_env.controlled_vehicles) == 1, "This wrapper supports only one controlled vehicle"
		controlled_vehicle = hooked_env.controlled_vehicles[0]
		controlled_vehicle.heading = self.heading or 2 * np.pi * hooked_env.np_random.uniform()

		self._hook_goal(hooked_env)
		self._hook_parked_cars(hooked_env)

	def _hook_goal(self, hooked_env: ParkingEnv) -> None:
		lanes = hooked_env.road.network.lanes_list()
		goal_index = self.goal_index or hooked_env.np_random.choice(len(lanes))
		lane = lanes[goal_index]
		goal = Landmark(hooked_env.road, lane.position(lane.length/2, 0), heading=lane.heading)
		for vehicle in hooked_env.controlled_vehicles:
			vehicle.goal = goal
		hooked_env.road.objects.append(goal)

	def _hook_parked_cars(self, hooked_env: ParkingEnv) -> None:
		for parked_vehicle_spot in self.parked_cars or []:
			lane = ("a", "b", parked_vehicle_spot) if parked_vehicle_spot < self.n_spots - 1 else (
				"b", "c", parked_vehicle_spot - self.n_spots + 1
			)
			v = Vehicle.make_on_lane(hooked_env.road, lane, 4, speed=0)
			hooked_env.road.vehicles.append(v)
