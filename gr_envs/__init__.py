from gr_envs.register import register_all_envs


register_all_envs()

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"  # fallback if file isn't present

try:
    import gr_envs.minigrid_scripts
except ImportError:
    pass
