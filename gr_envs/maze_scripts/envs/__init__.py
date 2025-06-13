from gr_envs.env_registration.maze_register import point_maze_register

try:
    point_maze_register()
except Exception as e:
    print(f"Point-Maze registration failed, {e}")
