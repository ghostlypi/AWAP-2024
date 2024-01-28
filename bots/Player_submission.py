from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import TowerType, Team, Tile, GameConstants, SnipePriority, get_debris_schedule
from src.debris import Debris
from src.tower import Tower
import numpy as np
#from bots.our_lib import gen_distance_map, sort_distance_map
import random


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
    del p.dist_dict[1000]


class BotPlayer(Player):

    def getSurroundingPath(self, rc, path):
        surrounding = [(1,0),(0,1),(-1,0),(0,-1)]
        surrounding_path = []
        for p in path:
            x,y = p
            for s in surrounding:
                if rc.is_placeable(rc.get_ally_team(), x+s[0], y+s[1]):
                    surrounding_path.append((x+s[0], y+s[1]))
        return surrounding_path

    def __init__(self, map: Map):
        self.map = map
        self.sniper_offset = 0
        gen_distance_map(self, map)
        sort_distance_map(self)
        for k in list(self.dist_dict.keys()):
            random.shuffle(self.dist_dict[k])
        self.next_build = 0
        pass

    def iter_build(self):
        self.next_build += 1
        self.next_build = self.next_build % 3

    
    def play_turn(self, rc: RobotController):
        self.build_towers(rc)
        self.towers_attack(rc)
        '''d_cool = 1
        d_health = 24
        if rc.can_send_debris(d_cool,d_health):
            rc.send_debris(d_cool,d_health)'''

    def remove_dist_dict_item(self, k, loc):
        self.dist_dict[k].remove(loc)
        if len(self.dist_dict[k]) == 0:
            del self.dist_dict[k]


    '''
    if (rc.can_build_tower(TowerType.GUNSHIP, x, y) and 
            rc.can_build_tower(TowerType.BOMBER, x, y) and
            rc.can_build_tower(TowerType.SOLAR_FARM, x, y) and
            rc.can_build_tower(TowerType.REINFORCER, x, y)
        ):
    '''
    def build_towers(self, rc: RobotController):
        #available = self.getSurroundingPath(rc, self.map.path)
        keys = sorted(list(self.dist_dict.keys()))
        #print(self.next_build)
        if len(keys) > 0:
            if self.next_build == 0:
                if keys[0] > 1:
                    self.next_build = 1
                else: 
                    k = 1
                    for loc in self.dist_dict[k]:
                        x,y = loc
                        if rc.can_build_tower(TowerType.BOMBER, x, y):
                            rc.build_tower(TowerType.BOMBER, x, y)
                            self.remove_dist_dict_item(k, loc)
                            self.iter_build()
                            break
            if self.next_build == 1:
                k = None
                for i in range(2, int(max(keys))):
                    if i in keys:
                        k = i
                        break
                if k is None:
                    self.iter_build()
                for loc in self.dist_dict[k]:
                    x,y = loc
                    if rc.can_build_tower(TowerType.GUNSHIP, x, y):
                        rc.build_tower(TowerType.GUNSHIP, x, y)
                        self.remove_dist_dict_item(k, loc)
                        self.iter_build()
                        break
            if self.next_build == 2:
                k = None
                for i in range(int(max(keys)-1), 2, -1):
                    if i in keys:
                        k = i
                        break
                if k is None:
                    self.iter_build()
                for loc in self.dist_dict[k]:
                    x,y = loc
                    if rc.can_build_tower(TowerType.SOLAR_FARM, x, y):
                        rc.build_tower(TowerType.SOLAR_FARM, x, y)
                        self.remove_dist_dict_item(k, loc)
                        self.iter_build()
                        break

        
    def towers_attack(self, rc: RobotController):
        towers = rc.get_towers(rc.get_ally_team())
        for tower in towers:
            if tower.type == TowerType.GUNSHIP:
                rc.auto_snipe(tower.id, SnipePriority.FIRST)
            elif tower.type == TowerType.BOMBER:
                rc.auto_bomb(tower.id)

