from minigrid.envs.blockedunlockpickup import BlockedUnlockPickupEnv
from minigrid.envs.crossing import CrossingEnv
from minigrid.envs.distshift import DistShiftEnv
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.envs.dynamicobstacles import DynamicObstaclesEnv
from minigrid.envs.empty import EmptyEnv
from minigrid.envs.fetch import FetchEnv
from minigrid.envs.fourrooms import FourRoomsEnv
from minigrid.envs.gotodoor import GoToDoorEnv
from minigrid.envs.gotoobject import GoToObjectEnv
from minigrid.envs.keycorridor import KeyCorridorEnv
from minigrid.envs.lavagap import LavaGapEnv
from minigrid.envs.lockedroom import LockedRoom, LockedRoomEnv
from minigrid.envs.memory import MemoryEnv
from minigrid.envs.multiroom import MultiRoom, MultiRoomEnv
from minigrid.envs.obstructedmaze import (
    ObstructedMaze_1Dlhb,
    ObstructedMaze_Full,
    ObstructedMazeEnv,
) # 
from minigrid.envs.playground import PlaygroundEnv
from minigrid.envs.putnear import PutNearEnv
from minigrid.envs.redbluedoors import RedBlueDoorEnv
from minigrid.envs.unlock import UnlockEnv
from minigrid.envs.unlockpickup import UnlockPickupEnv

# Additions not in minigrid:
from gr_libs.minigrid_scripts.envs.dynamic_goal_crossing import DynamicGoalCrossingEnv
from gr_libs.minigrid_scripts.envs.dynamic_goal_crossing_custom import DynamicGoalCrossingCustom13Env
from gr_libs.minigrid_scripts.envs.custom_color import CustomColorEnv
from gr_libs.minigrid_scripts.envs.dynamic_goal_empty import DynamicGoalEmpty
from gr_libs.minigrid_scripts.envs.dynamic_four_rooms import DynamicFourRoomsEnv
