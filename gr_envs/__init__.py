try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"  # fallback if file isn't present

try:
    import gr_envs.minigrid_scripts
except ImportError:
    pass

try:
    import gr_envs.panda_scripts
except ImportError:
    pass

try:
    import gr_envs.highway_scripts
except ImportError:
    pass

try:
    import gr_envs.maze_scripts
except ImportError:
    pass
