import gymnasium as gym
from gymnasium_robotics.envs.maze.maps import R, G

def gen_empty_env(width, height, initial_states, goal_states):
    # Create an empty environment matrix with walls (1) around the edges
    env = [[1 if x == 0 or x == width-1 or y == 0 or y == height-1 else 0 for x in range(width)] for y in range(height)]

    # Place initial states (R) and goal states (G)
    for x, y in initial_states:
        env[y][x] = R
    for x, y in goal_states:
        env[y][x] = G
    
    return env

def gen_four_rooms_env(width, height, initial_states, goal_states):
    # Create an empty environment matrix with walls (1) around the edges
    env = [[1 if x == 0 or x == width-1 or y == 0 or y == height-1 else 0 for x in range(width)] for y in range(height)]
    
    # Add walls for the four rooms structure
    for y in range(1, height-1):
        env[y][width // 2] = 1 if y != height // 4 and y != height * 3 // 4 else 0
    for x in range(1, width-1):
        env[height // 2][x] = 1 if x != width // 4 and x != width * 3 // 4 else 0
    
    # Place initial states (R) and goal states (G)
    for x, y in initial_states:
        env[y][x] = R
    for x, y in goal_states:
        env[y][x] = G
    
    return env

def gen_maze_with_obstacles(width, height, initial_states, goal_states, obstacles):
    # Create an empty environment matrix with walls (1) around the edges
    env = [[1 if x == 0 or x == width-1 or y == 0 or y == height-1 else 0 for x in range(width)] for y in range(height)]
    
    # Place obstacles (1)
    for x, y in obstacles:
        env[y][x] = 1
    
    # Place initial states (R) and goal states (G)
    for x, y in initial_states:
        env[y][x] = R
    for x, y in goal_states:
        env[y][x] = G
    
    return env

# Example usage
if __name__ == "__main__":
    width, height = 9, 9
    initial_states = [(1, 1)]
    goal_states = [(1, 7)]
    obstacles = [(3, 1), (3, 2), (3, 3)]
    
    empty_env = gen_empty_env(width, height, initial_states, goal_states)
    for row in empty_env:
        print(row)
    
    print()
    
    four_rooms_env = gen_four_rooms_env(width, height, initial_states, goal_states)
    for row in four_rooms_env:
        print(row)
    
    print()
    
    maze_with_obstacles = gen_maze_with_obstacles(width, height, initial_states, goal_states, obstacles)
    for row in maze_with_obstacles:
        print(row)
