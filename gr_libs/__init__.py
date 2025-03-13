# must import sub-packages so their init.py executed and registers everything.
from . import panda_gym_scripts, minigrid_scripts, maze_scripts, highway_env_scripts
from . import _version
__version__ = _version.get_versions()['version']
