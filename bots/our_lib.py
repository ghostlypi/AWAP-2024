from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import TowerType, Team, Tile, GameConstants, SnipePriority, get_debris_schedule
from src.debris import Debris
from src.tower import Tower
import numpy as np


def distance(l1, l2):
    return (l1[0]-l2[0])**2 + (l1[1]-l2[1])**2

def gen_distance_map(p, map):
    path = map.path
    p.dist_map = np.ones((p.map.width, p.map.height)) * 1000
    for loc in path:
        for x in range(p.dist_map.shape[0]):
            for y in range(p.dist_map.shape[1]):
                if map.is_space(x, y):
                    p.dist_map[x,y] = min(p.dist_map[x,y], distance(loc, (x,y)))

def sort_distance_map(p):
    p.dist_dict = {}
    for x in range(p.dist_map.shape[0]):
        for y in range(p.dist_map.shape[1]):
            d = p.dist_map[x,y]
            if d in p.dist_dict:
                p.dist_dict[d].append((x,y))
            else:
                p.dist_dict[d] = [(x,y)]
    del p.dist_dict[1000.0]
