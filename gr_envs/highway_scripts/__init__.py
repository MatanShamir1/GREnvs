# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


from gymnasium.envs.registration import register


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
            (8, (0, 1, 3, 4, 5, 7))
        ]:
            env_id = f'Parking-S-{spots_in_row}-PC-{"Y".join([str(pc) for pc in parked_cars])}-GI-{goal_index}-v0'
            register(
                id=env_id,
                entry_point='gr_envs.highway_scripts.envs.parking_env:ParkingEnv',
                kwargs={
                    "n_spots": spots_in_row,
                    "goal_index": goal_index,
                    "parked_cars": parked_cars
                },
            )
            gc_env_id = f'Parking-S-{spots_in_row}-PC-{"Y".join([str(pc) for pc in parked_cars])}-v0'
            register(
                id=gc_env_id,
                entry_point='gr_envs.highway_scripts.envs.parking_env:ParkingEnv',
                kwargs={
                    "n_spots": spots_in_row,
                    "parked_cars": parked_cars
                },
            )

register_highway_envs()
