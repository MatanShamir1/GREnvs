from gr_envs.env_registration.panda_register import panda_gym_register

try:
    panda_gym_register()
except Exception as e:
    print(f"Panda-Gym registration failed, {e}")
