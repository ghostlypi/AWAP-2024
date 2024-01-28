from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import TowerType, Team, Tile, GameConstants, SnipePriority, get_debris_schedule
from src.debris import Debris
from src.tower import Tower
import numpy as np


class BotPlayer(Player):

    def distance(l1, l2):
        return (l1[0]-l2[0])**2 + (l1[1]-l2[1])**2

    def gen_distance_map(self, path):
        self.dist_map = np.array(self.map.width, self.map.height)
        for loc in path:
            for x in range(self.dist_map.shape[0]):
                for y in range(self.dist_map.shape[1]):
                    self.dist_map[x,y] = min(self.dist_map[x,y], self.distance(loc, (x,y)))

    def sort_distance_map(self):
        self.dist_dict = {}
        for x in range(self.dist_map.shape[0]):
            for y in range(self.dist_map.shape[1]):
                d = self.dist_map[x,y]
                if d in self.dist_dict:
                    self.dist_dict[d].append((x,y))
                else:
                    self.dist_dict[d] = [(x,y)]

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
        self.gen_distance_map(map.path)
        self.sort_distance_map()
        pass

    def play_turn(self, rc: RobotController):
        self.build_towers(rc)
        self.towers_attack(rc)


    '''
    if (rc.can_build_tower(TowerType.GUNSHIP, x, y) and 
            rc.can_build_tower(TowerType.BOMBER, x, y) and
            rc.can_build_tower(TowerType.SOLAR_FARM, x, y) and
            rc.can_build_tower(TowerType.REINFORCER, x, y)
        ):
    '''
    def build_towers(self, rc: RobotController):
        available = self.getSurroundingPath(rc, self.map.path)
        if len(available) > 0:
            x,y = available[0]
            if rc.can_build_tower(TowerType.GUNSHIP, x, y):
                rc.build_tower(TowerType.GUNSHIP, x, y)
        '''if rc.can_build_tower(TowerType.BOMBER, x+1, y):
            rc.build_tower(TowerType.BOMBER, x, y)
        if rc.can_build_tower(TowerType.SOLAR_FARM, x+2, y):
            rc.build_tower(TowerType.SOLAR_FARM, x, y)
        if rc.can_build_tower(TowerType.REINFORCER, x+3, y):
            rc.build_tower(TowerType.REINFORCER, x, y)'''

    def towers_attack(self, rc: RobotController):
        towers = rc.get_towers(rc.get_ally_team())
        for tower in towers:
            if tower.type == TowerType.GUNSHIP:
                priority = None
                if self.sniper_offset == 0:
                    priority = SnipePriority.FIRST
                    self.sniper_offset = 1
                else:
                    priority = SnipePriority.STRONG
                rc.auto_snipe(tower.id, priority)
                
                