import gymnasium
import numpy as np
from gymnasium.envs.registration import register
from gr_envs.wrappers import ParkingWrapper, ParkingGoalList


def make_parking_wrapped(**kwargs):
    assert "n_spots" in kwargs, f"n_spots must be in {kwargs=}"
    assert "goal" in kwargs, f"goal_index must be in {kwargs=}"
    assert "heading" in kwargs, f"heading must be in {kwargs=}"
    assert "parked_cars" in kwargs, f"parked_cars must be in {kwargs=}"

    n_spots = kwargs.pop("n_spots")
    goal = kwargs.pop("goal")
    heading = kwargs.pop("heading")
    parked_cars = kwargs.pop("parked_cars")

    assert (
        len(set(goal.goal_list) & set(parked_cars)) == 0
    ), f"cant be intersection between {goal.goals} & {parked_cars}"

    assert (
        type(goal) == ParkingGoalList
    ), f"type of goal should be ParkingGoalList not {type(goal)}"

    env = gymnasium.make("parking-v0", **kwargs)
    return ParkingWrapper(
        env=env, n_spots=n_spots, goal=goal, heading=heading, parked_cars=parked_cars
    )


def register_highway_envs():
    for spots_in_row in [10, 12, 14, 16]:
        for goal_index, parked_cars in [
            (1, ()),
            (2, ()),
            (3, ()),
            (4, ()),
            (5, ()),
            (6, ()),
            (7, ()),
            (8, ()),
            (9, ()),
            (10, ()),
            (11, ()),
            (12, ()),
            (13, ()),
            (14, ()),
            (15, ()),
            (16, ()),
            (17, ()),
            (18, ()),
            (19, ()),
            (20, ()),
            (21, ()),
            (22, ()),
            (23, ()),
            (24, ()),
            (2, (0, 1, 3, 4, 5, 7)),
            (6, (0, 1, 3, 4, 5, 7)),
            (8, (0, 1, 3, 4, 5, 7)),
        ]:
            env_id = f'Parking-S-{spots_in_row}-PC-{"Y".join([str(pc) for pc in parked_cars])}-GI-{goal_index}-v0'
            register(
                id=env_id,
                entry_point=make_parking_wrapped,
                kwargs={
                    "n_spots": spots_in_row,
                    "heading": np.pi,
                    "goal": ParkingGoalList([goal_index]),
                    "parked_cars": parked_cars,
                },
                order_enforce=False,
                disable_env_checker=True,
            )
            gc_env_id = f'Parking-S-{spots_in_row}-PC-{"Y".join([str(pc) for pc in parked_cars])}-v0'
            register(
                id=gc_env_id,
                entry_point=make_parking_wrapped,
                kwargs={
                    "heading": np.pi,
                    "n_spots": spots_in_row,
                    "parked_cars": parked_cars,
                    "goal": ParkingGoalList(
                        [x for x in range(spots_in_row * 2) if x not in parked_cars]
                    ),
                },
                order_enforce=False,
                disable_env_checker=True,
            )

    # Goal subspace registration for parking
    center_goal_spots = [
        8,
        10,
        13,
    ]  # Define the center goal spots for the subspace
    register(
        id="Parking-S-14-PC--GI-8Y10Y13-v0",
        entry_point=make_parking_wrapped,
        kwargs={
            "n_spots": 14,
            "goal": ParkingGoalList(center_goal_spots),
            "heading": np.pi,
            "parked_cars": (),
        },
        order_enforce=False,
        disable_env_checker=True,
    )


register_highway_envs()
