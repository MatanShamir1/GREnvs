from gr_envs.env_registration.highway_register import register_highway_envs

try:
    register_highway_envs()
except Exception as e:
    print(f"Highway-Env registration failed, {e}")
