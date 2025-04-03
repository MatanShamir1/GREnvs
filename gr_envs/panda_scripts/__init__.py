import os

from gymnasium.envs.registration import register

# with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
#     __version__ = file_handler.read().strip()

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

#         register(
#             id="PandaReach{}{}-v3".format(control_suffix, reward_suffix),
#             entry_point="panda_gym.envs:PandaReachEnv",
#             kwargs=kwargs,
#             max_episode_steps=50,
#         )

#         register(
#             id="PandaPush{}{}-v3".format(control_suffix, reward_suffix),
#             entry_point="panda_gym.envs:PandaPushEnv",
#             kwargs=kwargs,
#             max_episode_steps=50,
#         )

#         register(
#             id="PandaSlide{}{}-v3".format(control_suffix, reward_suffix),
#             entry_point="panda_gym.envs:PandaSlideEnv",
#             kwargs=kwargs,
#             max_episode_steps=50,
#         )

#         register(
#             id="PandaPickAndPlace{}{}-v3".format(control_suffix, reward_suffix),
#             entry_point="panda_gym.envs:PandaPickAndPlaceEnv",
#             kwargs=kwargs,
#             max_episode_steps=50,
#         )

#         register(
#             id="PandaStack{}{}-v3".format(control_suffix, reward_suffix),
#             entry_point="panda_gym.envs:PandaStackEnv",
#             kwargs=kwargs,
#             max_episode_steps=100,
#         )

#         register(
#             id="PandaFlip{}{}-v3".format(control_suffix, reward_suffix),
#             entry_point="panda_gym.envs:PandaFlipEnv",
#             kwargs=kwargs,
#             max_episode_steps=50,
#         )

        register(
            id="PandaMyReach{}{}-v3".format(control_suffix, reward_suffix),
            entry_point="gr_envs.panda_scripts.envs:PandaMyReachEnv",
            kwargs={**kwargs, "goal_range_negative": -0.5, "goal_range_positive": 0.2},
            max_episode_steps=101,
        )

reward_type = "dense"
control_type = "ee"
goals = [(-0.5, -0.5, 0.1), (-0.3, -0.3, 0.1), (-0.1, -0.1, 0.1), (-0.5, 0.2, 0.1), (-0.3, 0.2, 0.1), (-0.1, 0.1, 0.1), (0.2, -0.2, 0.1), (0.2, -0.3, 0.1), (0.1, -0.1, 0.1), (0.2, 0.2, 0.1), (0.0, 0.0, 0.1), (0.1, 0.1, 0.1)]
reward_suffix = "Dense" if reward_type == "dense" else ""
control_suffix = "Joints" if control_type == "joints" else ""
for goal in goals:
    goal_str = 'X'.join([str(float(g)).replace(".", "y").replace("-","M") for g in goal])
    goal_x, goal_y, goal_z = goal
    kwargs = {"reward_type": reward_type, "control_type": control_type, "goal_x": goal_x, "goal_y": goal_y, "goal_z": goal_z}
    env_id = "PandaMyReach{}{}X{}-v3".format(control_suffix, reward_suffix, goal_str)
    #print(f"{goal}:{env_id}")
    register(
        id=env_id,
        entry_point="gr_envs.panda_scripts.envs:PandaMyReachEnv",
        kwargs=kwargs,
        max_episode_steps=101, 
    )
