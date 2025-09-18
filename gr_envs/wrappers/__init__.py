import warnings

try:
    from gr_envs.wrappers.parking_wrapper import ParkingWrapper, ParkingGoalList
except ImportError:
    warnings.warn("ParkingWrapper or ParkingGoalList could not be imported. Install gr_envs[highway] if needed.", RuntimeWarning)

try:
    from gr_envs.wrappers.panda_wrapper import PandaGymWrapper, PandaGymDesiredGoalList
except ImportError:
    warnings.warn("PandaGymWrapper or PandaGymDesiredGoalList could not be imported. Install gr_envs[panda] if needed.", RuntimeWarning)

try:
    from gr_envs.wrappers.goal_wrapper import GoalRecognitionWrapper
except ImportError:
    warnings.warn("GoalRecognitionWrapper could not be imported.", RuntimeWarning)