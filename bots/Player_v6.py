from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import TowerType, Team, Tile, GameConstants, SnipePriority, get_debris_schedule
from src.debris import Debris
from src.tower import Tower
import numpy as np
from bots.our_lib import gen_distance_map, sort_distance_map, best_coverage, distance
import random

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
        #self.spawn_dist = [1, 3, 7]
        self.spawn_dist = [0, 3, 4, 7]
        self.n_towers = []
        self.sold_solar = False
        self.debris_costs = dict()
    
    def precompute_debris(self, rc: RobotController):
        for i in range(45,940):
            self.debris_costs[rc.get_debris_cost(1,i)] = (1,i)
            
    def get_debris(self, c):
        keys = list(self.debris_costs.keys())
        for i in range(len(keys)):
            if c < keys[i]:
                return self.debris_costs[keys[i-1]]
        return 0
                
    def iter_build(self):
        self.next_build += 1
        self.next_build = self.next_build % (self.spawn_dist[-1] + 1)

    
    def play_turn(self, rc: RobotController):
        if (rc.get_turn() <= 1):
            self.precompute_debris(rc)
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

    def build_towers(self, rc: RobotController):
        #available = self.getSurroundingPath(rc, self.map.path)
        for i in range(min(1 + rc.get_turn()//1000, 4)):
            keys = sorted(list(self.dist_dict.keys()))
            added = False
            if len(keys) > 0:
                if self.next_build <= self.spawn_dist[0]:
                    k = None
                    for i in range(int(max(keys)-1), 0, -1):
                        if i in keys:
                            k = i
                            break
                    if k is not None:
                        for loc in self.dist_dict[k]:
                            x,y = loc
                            if rc.can_build_tower(TowerType.SOLAR_FARM, x, y):
                                rc.build_tower(TowerType.SOLAR_FARM, x, y)
                                self.remove_dist_dict_item(k, loc)
                                self.iter_build()
                                break
                    else:
                        self.iter_build()
                        
                elif self.spawn_dist[0] < self.next_build and self.next_build <= self.spawn_dist[1]:
                    if rc.get_balance(rc.get_ally_team()) >= 1000:
                        top_coord = best_coverage(self, self.map, 60)
                        if top_coord is not None:
                            (k,(x,y)) = top_coord
                            if (rc.can_build_tower(TowerType.GUNSHIP, x, y)):
                                rc.build_tower(TowerType.GUNSHIP, x, y)
                                self.remove_dist_dict_item(k,(x,y))
                                self.iter_build()
                        else:
                            k = None
                            for i in range(1, int(max(keys))):
                                if i in keys:
                                    k = i
                                    break
                            if k is not None:
                                for loc in self.dist_dict[k]:
                                    x,y = loc
                                    if rc.can_build_tower(TowerType.GUNSHIP, x, y):
                                        rc.build_tower(TowerType.GUNSHIP, x, y)
                                        self.remove_dist_dict_item(k, loc)
                                        self.iter_build()
                                        break
                            else:
                                self.iter_build()
                elif self.spawn_dist[1] < self.next_build and self.next_build <= self.spawn_dist[2]:
                    if rc.get_balance(rc.get_ally_team()) >= 1800:
                        top_coord = best_coverage(self, self.map, 2)
                        if top_coord is not None:
                            (k,(x,y)) = top_coord
                            if (rc.can_build_tower(TowerType.BOMBER, x, y)):
                                rc.build_tower(TowerType.BOMBER, x, y)
                                self.remove_dist_dict_item(k,(x,y))
                                self.iter_build()
                        else:
                            k = None
                            for i in range(2, int(max(keys))):
                                if i in keys:
                                    k = i
                                    break
                            if k is not None:
                                for loc in self.dist_dict[k]:
                                    x,y = loc
                                    if rc.can_build_tower(TowerType.BOMBER, x, y):
                                        rc.build_tower(TowerType.BOMBER, x, y)
                                        self.remove_dist_dict_item(k, loc)
                                        self.iter_build()
                                        break
                            else:
                                self.iter_build()
                elif self.spawn_dist[2] < self.next_build and self.next_build <= self.spawn_dist[3]:
                    k = None
                    for i in range(int(max(keys)-1), 0, -1):
                        if i in keys:
                            k = i
                            break
                    if k is not None:
                        for loc in self.dist_dict[k]:
                            x,y = loc
                            if rc.can_build_tower(TowerType.SOLAR_FARM, x, y):
                                rc.build_tower(TowerType.SOLAR_FARM, x, y)
                                self.remove_dist_dict_item(k, loc)
                                self.iter_build()
                                break
                    else:
                        self.iter_build()
            else:
                print(f'{len(keys)}')
                self.update_towers(rc, rc.get_towers(rc.get_ally_team()))
        
        keys = sorted(list(self.dist_dict.keys()))
        if len(keys) > 0:
            print(len(keys), end='\r')
            (cooldown, health) = self.get_debris(rc.get_balance(rc.get_ally_team())-4000)
        else:
            (cooldown, health) = self.get_debris(rc.get_balance(rc.get_ally_team()))
        if rc.can_send_debris(cooldown,health):
            rc.send_debris(1,health)
        
    def towers_attack(self, rc: RobotController):
        towers = rc.get_towers(rc.get_ally_team())
        for tower in towers:
            if tower.type == TowerType.GUNSHIP:
                rc.auto_snipe(tower.id, SnipePriority.STRONG)
            elif tower.type == TowerType.BOMBER:
                rc.auto_bomb(tower.id)
                
    def update_towers(self, rc, towers):
        if not self.sold_solar:
            for tower in towers:
                if tower.type == TowerType.SOLAR_FARM:
                    x = tower.x
                    y = tower.y
                    if self.dist_map[x,y] <= 60:
                        rc.sell_tower(tower.id)
                        rc.build_tower(TowerType.GUNSHIP, x, y)
            self.sold_solar = True

