def register_all_envs():
    try:
        from gr_envs.env_registration.panda_register import panda_gym_register
        panda_gym_register()
    except Exception as e:
        print(f"Panda-Gym registration failed, {e}. Ignore this warning if you didn't install gr_envs[panda] extra.")

    try:
        from gr_envs.env_registration.highway_register import register_highway_envs
        register_highway_envs()
    except Exception as e:
        print(f"Highway-Env registration failed, {e}. Ignore this warning if you didn't install gr_envs[highway] extra.")

    try:
        from gr_envs.env_registration.maze_register import point_maze_register
        point_maze_register()
    except Exception as e:
        print(f"Point-Maze registration failed, {e}. Ignore this warning if you didn't install gr_envs[maze] extra.")
