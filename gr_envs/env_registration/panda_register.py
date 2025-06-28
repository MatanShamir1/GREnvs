import gymnasium
import numpy as np
from gymnasium.envs.registration import register
from panda_gym import reward_suffix

from gr_envs.wrappers import PandaGymWrapper, PandaGymDesiredGoalList


def make_panda_wrapped(**kwargs):
    assert "action_space" in kwargs, f"action_space must be in {kwargs=}"
    assert "reward_type" in kwargs, f"reward_type must be in {kwargs=}"

    reward_type = kwargs.pop("reward_type")
    reward_suffix = "Dense" if reward_type == "dense" else ""

    goal = kwargs.pop("goal") if "goal" in kwargs else None
    action_space = kwargs.pop("action_space")

    env = gymnasium.make(f"PandaReach{reward_suffix}-v3", **kwargs)
    return PandaGymWrapper(env=env, goal=goal, action_space=action_space)


def panda_gym_register():
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""

            register(
                id="PandaMyReach{}{}-v3".format(control_suffix, reward_suffix),
                entry_point=make_panda_wrapped,
                kwargs={
                    "reward_type": reward_type,
                    "control_type": control_type,
                    "action_space": gymnasium.spaces.Box(
                        low=-0.5, high=0.5, shape=(3,), dtype=np.float32
                    ),
                },
                max_episode_steps=101,
            )

            reward_type = "dense"
            control_type = "ee"
            goals = [
                (-0.5, -0.5, 0.1),
                (-0.3, -0.3, 0.1),
                (-0.1, -0.1, 0.1),
                (-0.5, 0.2, 0.1),
                (-0.3, 0.2, 0.1),
                (-0.1, 0.1, 0.1),
                (0.2, -0.2, 0.1),
                (0.2, -0.3, 0.1),
                (0.1, -0.1, 0.1),
                (0.2, 0.2, 0.1),
                (0.0, 0.0, 0.1),
                (0.1, 0.1, 0.1),
            ]
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            for goal in goals:
                goal_str = "X".join(
                    [str(float(g)).replace(".", "y").replace("-", "M") for g in goal]
                )
                register(
                    id="PandaMyReach{}{}X{}-v3".format(
                        control_suffix, reward_suffix, goal_str
                    ),
                    entry_point=make_panda_wrapped,
                    kwargs={
                        "reward_type": reward_type,
                        "control_type": control_type,
                        "goal": PandaGymDesiredGoalList(
                            gymnasium.spaces.Box(
                                low=np.array(goal, dtype=np.float32),
                                high=np.array(goal, dtype=np.float32),
                                shape=(3,),
                                dtype=np.float32,
                            )
                        ),
                        "action_space": gymnasium.spaces.Box(
                            low=-0.5, high=0.5, shape=(3,), dtype=np.float32
                        ),
                    },
                    max_episode_steps=101,
                )

            # Register Panda environment with a center goal subspace (small area near the center)
            # This is useful for GCAura algorithm that trains on a subspace but adapts to goals outside it
            center_goals = [
                (-0.05, -0.05, 0.1),
                (-0.02, -0.02, 0.1),
                (0.0, 0.0, 0.1),
                (0.02, 0.02, 0.1),
                (0.05, 0.05, 0.1),
                (-0.05, 0.05, 0.1),
                (0.05, -0.05, 0.1),
                (0.0, 0.05, 0.1),
                (0.0, -0.05, 0.1),
                (0.05, 0.0, 0.1),
                (-0.05, 0.0, 0.1),
            ]

            # Create goal space that only covers the center region
            center_goal_space = gymnasium.spaces.Box(
                low=np.array([-0.05, -0.05, 0.1], dtype=np.float32),
                high=np.array([0.05, 0.05, 0.1], dtype=np.float32),
                shape=(3,),
                dtype=np.float32,
            )

            # Register the subspace environment for center goals only
            register(
                id="PandaMyReach{}{}SubspaceCenterOnly-v3".format(
                    control_suffix, reward_suffix
                ),
                entry_point=make_panda_wrapped,
                kwargs={
                    "reward_type": reward_type,
                    "control_type": control_type,
                    "goal": PandaGymDesiredGoalList(center_goal_space),
                    "action_space": gymnasium.spaces.Box(
                        low=-0.5, high=0.5, shape=(3,), dtype=np.float32
                    ),
                },
                max_episode_steps=101,
            )

            left_front_goal_space = gymnasium.spaces.Box(
                low=np.array([-0.5, -0.5, 0.1], dtype=np.float32),
                high=np.array([-0.1, -0.1, 0.1], dtype=np.float32),
                shape=(3,),
                dtype=np.float32,
            )

            register(
                id="PandaMyReach{}{}SubspaceLeftFront-v3".format(
                    control_suffix, reward_suffix
                ),
                entry_point=make_panda_wrapped,
                kwargs={
                    "reward_type": reward_type,
                    "control_type": control_type,
                    "goal": PandaGymDesiredGoalList(left_front_goal_space),
                    "action_space": gymnasium.spaces.Box(
                        low=-0.5, high=0.5, shape=(3,), dtype=np.float32
                    ),
                },
                max_episode_steps=101,
            )


panda_gym_register()
