from abc import abstractmethod
from typing import Optional, List

from gymnasium import Env
import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False
        }}
    DEFAULT_N_SPOTS = 14

    def __init__(self, config: dict = None, render_mode: Optional[str] = None, goal_index: int = -1, n_spots: int = DEFAULT_N_SPOTS, parked_cars: List[int] = []) -> None:
        self.observation_type_parking = None
        self.goal_index = goal_index
        self.n_spots = n_spots
        self.parked_cars = parked_cars
        super().__init__(config, render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            "vehicles_count": 0,
            "add_walls": True
        })
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info.update({"is_success": success})
        return info
        
    # reset is called from abstract without goal_idx. this is specifically for env creation for GR generate_observation
    def _reset(self):
        self._create_road(spots=self.n_spots)
        self._create_vehicles()
        # if not hasattr(self, 'goal') or (hasattr(self, 'goal_idx') and self.goal_idx != None): # this is sometimes called from 'step', but the goal already exists. no need to create it again...
        #     self._create_goal_landmark(self.goal_idx)
        # elif hasattr(self, 'goal') and self.goal not in self.road.objects:
        #     self.road.objects.append(self.goal)
        self._create_goal_landmark(self.goal_idx)
        
    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
    ):
        # abstract calls our _reset and we add the goal landmark afterwards. only way I found to fit in the framework and create which goal you want.
        # Assigning a specific goal_idx will make even a goal-conditioned agent get it in the specific episode instead of randomly picking a goal.
        # We want this goal_idx assigning to arrive from 'options' and be assigned for every step of the episode, but 'reset' is called from reset directly and also from 'step'.
        # So if 'options' arrived, we assign, as this is the first 'reset'. In case of being called from 'step', options is not given,
        # but we don't want to change self.goal_idx since it was correctly put in the earlier 'reset'. So we recognize it in the elif by asking if goal_idx exists,
        # and if it does - we leave it as-is: -1 will keep it -1 and use the same goal, and a specified number will keep it too.
        # putting None will pick it randomly.
        # You expect first time else, second time if, third time (from step) elif.
        if options:
            self.goal_idx = options["goal_idx"]
        elif hasattr(self, 'goal_idx'):
            pass
        else:
            self.goal_idx = None
        return super().reset(seed=seed, options=options)

    def _create_road(self, spots) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])
        
    def _create_goal_landmark(self, goal_idx):
        lanes = self.road.network.lanes_list()
        # Goal
        if goal_idx != None: # SHOULD ONLY BE GIVEN FOR A parking-v0 AGENT, NAMELY A GOAL-CONDITIONED AGENT!
            assert self.goal_index == -1 # make sure if a specific goal_idx was given then this is a goal-conditioned agent's env, goal renders randomly every episode.
        elif len(self.road.network.lanes_list()) > self.goal_index >= 0: # in case of a goal-directed agent that has a specific goal_index, it is assigned here and added as a landmark.
            goal_idx = self.goal_index
        else:
            goal_idx = self.np_random.choice(len(lanes))

        lane = lanes[goal_idx]

        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        for vehicle in self.controlled_vehicles:
            if hasattr(vehicle, 'goal'):
                pass
            else:
                vehicle.goal = self.goal
        self.road.objects.append(self.goal) # depleted after every step, need to add the goal again at every step.

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class(self.road, [i*20, 0], 2*np.pi*self.np_random.uniform(), 0)
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        for parked_vehicle_spot in self.parked_cars:
            lane = ("a", "b", parked_vehicle_spot) if parked_vehicle_spot <= self.n_spots else ("b", "c", self.n_spots - parked_vehicle_spot)
            v = Vehicle.make_on_lane(self.road, lane, 4, speed=0)
            self.road.vehicles.append(v)

        # Other vehicles
        for i in range(self.config["vehicles_count"]):
            lane = ("a", "b", i) if self.np_random.uniform() >= 0.5 else ("b", "c", i)
            v = Vehicle.make_on_lane(self.road, lane, 4, speed=0)
            self.road.vehicles.append(v)
        # for v in self.road.vehicles:  # Prevent early collisions
        #     if v is not self.vehicle and (
        #             np.linalg.norm(v.position - self.goal.position) < 20 or
        #             np.linalg.norm(v.position - self.vehicle.position) < 20):
        #         self.road.vehicles.remove(v)

        # Walls
        if self.config["add_walls"]:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)
        reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]


class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


class ParkingEnvParkedVehicles(ParkingEnv):
    def __init__(self):
        super().__init__({"vehicles_count": 10})

        
# class MultiTaskParkingEnv(AbstractEnv, GoalEnv):
#     """
#     A continuous control environment.

#     It implements a reach-type task, where the agent observes their position and speed and must
#     control their acceleration and steering so as to reach a given goal.

#     Credits to Munir Jojo-Verge for the idea and initial implementation.
#     """

#     # For parking env with GrayscaleObservation, the env need
#     # this PARKING_OBS to calculate the reward and the info.
#     # Bug fixed by Mcfly(https://github.com/McflyWZX)
#     PARKING_OBS = {"observation": {
#             "type": "KinematicsGoal",
#             "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
#             "scales": [100, 100, 5, 5, 1, 1],
#             "normalize": False
#         }}
#     DEFAULT_N_SPOTS = 14

#     def __init__(self, config: dict = None, render_mode: Optional[str] = None, goal_indices: List = [-1], n_spots: int = DEFAULT_N_SPOTS, parked_cars: List[int] = []) -> None:
#         self.observation_type_parking = None
#         self.goal_indices = goal_indices
#         self.n_spots = n_spots
#         self.parked_cars = parked_cars
#         super().__init__(config, render_mode)

#     @classmethod
#     def default_config(cls) -> dict:
#         config = super().default_config()
#         config.update({
#             "observation": {
#                 "type": "KinematicsGoal",
#                 "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
#                 "scales": [100, 100, 5, 5, 1, 1],
#                 "normalize": False
#             },
#             "action": {
#                 "type": "ContinuousAction"
#             },
#             "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
#             "success_goal_reward": 0.12,
#             "collision_reward": -5,
#             "steering_range": np.deg2rad(45),
#             "simulation_frequency": 15,
#             "policy_frequency": 5,
#             "duration": 100,
#             "screen_width": 600,
#             "screen_height": 300,
#             "centering_position": [0.5, 0.5],
#             "scaling": 7,
#             "controlled_vehicles": 1,
#             "vehicles_count": 0,
#             "add_walls": True
#         })
#         return config

#     def define_spaces(self) -> None:
#         """
#         Set the types and spaces of observation and action from config.
#         """
#         super().define_spaces()
#         self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])

#     def _info(self, obs, action) -> dict:
#         info = super(ParkingEnv, self)._info(obs, action)
#         if isinstance(self.observation_type, MultiAgentObservation):
#             success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
#         else:
#             obs = self.observation_type_parking.observe()
#             success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
#         info.update({"is_success": success})
#         return info

#     def _reset(self):
#         self._create_road(spots=self.n_spots)
#         self._create_vehicles()

#     def _create_road(self, spots) -> None:
#         """
#         Create a road composed of straight adjacent lanes.

#         :param spots: number of spots in the parking
#         """
#         net = RoadNetwork()
#         width = 4.0
#         lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
#         x_offset = 0
#         y_offset = 10
#         length = 8
#         for k in range(spots):
#             x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
#             net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
#             net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))

#         self.road = Road(network=net,
#                          np_random=self.np_random,
#                          record_history=self.config["show_trajectories"])

#     def _create_vehicles(self) -> None:
#         """Create some new random vehicles of a given type, and add them on the road."""
#         # Controlled vehicles
#         self.controlled_vehicles = []
#         for i in range(self.config["controlled_vehicles"]):
#             vehicle = self.action_type.vehicle_class(self.road, [i*20, 0], 2*np.pi*self.np_random.uniform(), 0)
#             vehicle.color = VehicleGraphics.EGO_COLOR
#             self.road.vehicles.append(vehicle)
#             self.controlled_vehicles.append(vehicle)

#         lanes = self.road.network.lanes_list()
#         # Goal
#         if any(len(self.road.network.lanes_list()) < element for element in goal_index):
#             # we don't support a goal-conditioned agent here: only a multi-task agent with specific goal tasks. Thus we won't generate random goals for an episode if a goal_index of -1 was given.
#             raise NotImplementedError(f"Can't use network lanes larger than {len(self.road.network.lanes_list())}")
#         else:
#             goal_indices = self.goal_indices
#             self.achieved_goals_indices = []
            
#         goal_lanes = [lanes[goal_idx] for goal_idx in goal_indices]

#         self.goals = [Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading) for lane in goal_lanes]
#         for goal in self.goals:
#             self.road.objects.append(goal)

#         for parked_vehicle_spot in self.parked_cars:
#             lane = ("a", "b", parked_vehicle_spot) if parked_vehicle_spot <= self.n_spots else ("b", "c", self.n_spots - parked_vehicle_spot)
#             v = Vehicle.make_on_lane(self.road, lane, 4, speed=0)
#             self.road.vehicles.append(v)

#         # Other vehicles
#         for i in range(self.config["vehicles_count"]):
#             lane = ("a", "b", i) if self.np_random.uniform() >= 0.5 else ("b", "c", i)
#             v = Vehicle.make_on_lane(self.road, lane, 4, speed=0)
#             self.road.vehicles.append(v)
#         # for v in self.road.vehicles:  # Prevent early collisions
#         #     if v is not self.vehicle and (
#         #             np.linalg.norm(v.position - self.goal.position) < 20 or
#         #             np.linalg.norm(v.position - self.vehicle.position) < 20):
#         #         self.road.vehicles.remove(v)

#         # Walls
#         if self.config['add_walls']:
#             width, height = 70, 42
#             for y in [-height / 2, height / 2]:
#                 obstacle = Obstacle(self.road, [0, y])
#                 obstacle.LENGTH, obstacle.WIDTH = (width, 1)
#                 obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
#                 self.road.objects.append(obstacle)
#             for x in [-width / 2, width / 2]:
#                 obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
#                 obstacle.LENGTH, obstacle.WIDTH = (height, 1)
#                 obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
#                 self.road.objects.append(obstacle)

#     def compute_reward(self, achieved_goal: np.ndarray, desired_goals: np.ndarray, info: dict, p: float = 0.5) -> float:
#         """
#         Achievement of each goal is rewarded

#         :param achieved_goal: the goal that was achieved
#         :param desired_goal: the goal that was desired
#         :param dict info: any supplementary information
#         :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
#         :return: the corresponding reward
#         """
#         if any((-np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)) > -self.config["success_goal_reward"] for desired_goal in desired_goals): return 1
#         else: return 0

#     # only relevant for multiagents
#     def _reward(self, action: np.ndarray) -> float:
#         obs = self.observation_type_parking.observe()
#         obs = obs if isinstance(obs, tuple) else (obs,)
#         reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)
#         reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
#         return reward

#     def _is_success(self, achieved_goal: np.ndarray, desired_goals: np.ndarray) -> bool:
#         return self.compute_reward(achieved_goal, desired_goals, {}) > 0

#     def _is_terminated(self) -> bool:
#         """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
#         crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
#         obs = self.observation_type_parking.observe()
#         obs = obs if isinstance(obs, tuple) else (obs,)
#         success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
#         return bool(crashed or success)

#     def _is_truncated(self) -> bool:
#         """The episode is truncated if the time is over."""
#         return self.time >= self.config["duration"]
