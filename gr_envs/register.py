from env_registration.panda_gym_register import panda_gym_register
from env_registration.highway_env_register import register_highway_envs

try:
    panda_gym_register()
except Exception as e:
    print(f"Panda-Gym registration failed, {e}")

try:
    register_highway_envs()
except Exception as e:
    print(f"Highway-Env registration failed, {e}")
