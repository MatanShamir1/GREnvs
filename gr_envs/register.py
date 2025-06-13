from gr_envs.env_registration.panda_register import panda_gym_register
from gr_envs.env_registration.highway_register import register_highway_envs
from gr_envs.env_registration.maze_register import point_maze_register


def register_all_envs():
    try:
        panda_gym_register()
    except Exception as e:
        print(f"Panda-Gym registration failed, {e}")

    try:
        register_highway_envs()
    except Exception as e:
        print(f"Highway-Env registration failed, {e}")

    try:
        point_maze_register()
    except Exception as e:
        print(f"Point-Maze registration failed, {e}")
