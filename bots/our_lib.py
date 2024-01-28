from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import TowerType, Team, Tile, GameConstants, SnipePriority, get_debris_schedule
from src.debris import Debris
from src.tower import Tower
import numpy as np


def distance(l1, l2):
    return int((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)

def gen_distance_map(p, map):
    path = map.path
    p.dist_map = np.ones((p.map.width, p.map.height), dtype=int) * 1000
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
    
def init_distance_map(p,map):
    gen_distance_map(p,map)
    sort_distance_map(p)

def coverage(map,
             point : tuple[int, int],
             dist : int
            ):
    path = map.path
    acc = 0
    for loc in path:
        acc += 1 if distance(loc, point) <= (dist**2) else 0     
    return acc

def best_coverage(p, map,
                  dist : int
                ):
    top_coords = None
    top_c = 0
    top_i = None
    for i in p.dist_dict.keys():
        if i < dist:
            for x in p.dist_dict[i]:
                c = coverage(map, x, dist)
                if c > top_c:
                    top_coords = x
                    top_c = c
    
    return top_coords
