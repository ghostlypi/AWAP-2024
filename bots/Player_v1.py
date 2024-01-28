from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import TowerType, Team, Tile, GameConstants, SnipePriority, get_debris_schedule
from src.debris import Debris
from src.tower import Tower
import numpy as np
from bots.our_lib import gen_distance_map, sort_distance_map


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
        self.next_build = 0
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
        #available = self.getSurroundingPath(rc, self.map.path)
        keys = self.dist_dict.keys()

        if self.next_build == 0:
            k = min(keys)
            for loc in self.dist_dict[k]:
                x,y = loc
                if rc.can_build_tower(TowerType.BOMBER, x, y):
                    rc.build_tower(TowerType.BOMBER, x, y)
                    self.dist_dict[k].remove(loc)
                    self.next_build += 1
                    self.next_build = self.next_build % 2
                    break
        elif self.next_build == 1:
            k = max(keys)
            print(k)
            for loc in self.dist_dict[k]:
                x,y = loc
                if rc.can_build_tower(TowerType.GUNSHIP, x, y):
                    rc.build_tower(TowerType.GUNSHIP, x, y)
                    self.dist_dict[k].remove(loc)
                    self.next_build += 1
                    self.next_build = self.next_build % 2
                    break


        '''if len(available) > 0:
            x,y = available[0]
            if rc.can_build_tower(TowerType.GUNSHIP, x, y):
                rc.build_tower(TowerType.GUNSHIP, x, y)'''
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


