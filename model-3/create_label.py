from grid_adventure.levels.intro import (
    build_level_maze_turns, build_level_optional_coin, build_level_required_multiple, 
    build_level_key_door, build_level_hazard_detour, build_level_pushable_box, build_level_power_shield, 
    build_level_power_ghost, build_level_power_boots, build_level_combined_mechanics, build_level_boss
)

from grid_adventure.grid import GridState, from_state
from grid_adventure.env import ImageObservation
from grid_adventure.env import GridAdventureEnv
from grid_adventure.entities import (
    FloorEntity, WallEntity, AgentEntity, ExitEntity,
    CoinEntity, GemEntity, KeyEntity, LockedDoorEntity,
    LavaEntity, BoxEntity, SpeedPowerUpEntity, ShieldPowerUpEntity,
    PhasingPowerUpEntity, create_agent_entity
)
# State steppers
from grid_adventure.step import Action, step
from grid_adventure.grid import to_state

# Movements and Objectives are gridstate parameters. For Grid Adventure V1, we will be using the default ones.
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES

from utils import create_env
from PIL import Image

import numpy as np

def build_level_all_entities() -> GridState:
    gridstate = GridState(width=4, height=7, movement=MOVEMENTS['cardinal'], objective=OBJECTIVES['collect_gems_and_exit'], seed=0)

    for (x, y) in [(0, 2), (1, 5)]:
        gridstate.add((x, y), BoxEntity())
    for (x, y) in [(0, 1), (1, 4)]:
        gridstate.add((x, y), CoinEntity())
    for (x, y) in [(0, 4), (3, 0)]:
        gridstate.add((x, y), ExitEntity())
    for (x, y) in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]:
        gridstate.add((x, y), FloorEntity())
    for (x, y) in [(1, 1), (2, 4)]:
        gridstate.add((x, y), GemEntity())
    for (x, y) in [(2, 1), (3, 4)]:
        gridstate.add((x, y), KeyEntity())
    for (x, y) in [(1, 2), (2, 5)]:
        gridstate.add((x, y), LavaEntity())
    for (x, y) in [(0, 5), (3, 1)]:
        gridstate.add((x, y), LockedDoorEntity())
    for (x, y) in [(0, 3), (1, 6)]:
        gridstate.add((x, y), PhasingPowerUpEntity())
    for (x, y) in [(0, 6), (3, 2)]:
        gridstate.add((x, y), ShieldPowerUpEntity())
    for (x, y) in [(2, 2), (3, 5)]:
        gridstate.add((x, y), SpeedPowerUpEntity())
    for (x, y) in [(1, 0), (2, 3)]:
        gridstate.add((x, y), WallEntity())
    for (x, y) in [(2, 0), (3, 3), (3, 6)]:
        gridstate.add((x, y), create_agent_entity(health=5))

    return gridstate

def create_labels():
    tiles = []  # list of (tile_image, label)
    for seed in range(0, 500):
        env = create_env(build_level_all_entities, observation_type='image', seed=seed)
        image_obs, _ = env.reset()
        
        image = image_obs['image'] ## returns images in numpy array (H, W, 4)
        
        img_width = image.shape[1]
        img_height = image.shape[0]
        
        gridstate = from_state(env.state)
        grid_width = gridstate.width
        grid_height = gridstate.height
        
        tile_width = img_width // grid_width
        tile_height = img_height // grid_height
        
        for y in range(grid_height):
            for x in range(grid_width):
                tile_img = image[y*tile_height:(y+1)*tile_height,
                                x*tile_width:(x+1)*tile_width]
                
                if (x == 5 and y == 3) or (x == 6 and y == 3):
                    continue
                entities = gridstate.grid[x][y]
                label = "floor"
                
                for e in entities:
                    if isinstance(e, FloorEntity) == False:
                        label = e.appearance.name
                        break
                tiles.append((tile_img, label))
    return tiles
                    