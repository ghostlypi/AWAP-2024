import random
from src.game_constants import SnipePriority, TowerType
from src.robot_controller import RobotController
from src.player import Player
from src.map import Map
from bots.our_lib import init_distance_map, best_coverage

class BotPlayer(Player):
    def __init__(self, map: Map):
        self.map = map
        init_distance_map(self, map)
        self.build = TowerType.SOLAR_FARM

    def play_turn(self, rc: RobotController):
        self.build_towers(rc)
        self.towers_attack(rc)
    
    def remove_dist_dict_item(self, k, loc):
        self.dist_dict[k].remove(loc)
        if len(self.dist_dict[k]) == 0:
            del self.dist_dict[k]

    def build_towers(self, rc: RobotController):
        if self.build == TowerType.SOLAR_FARM:
            if rc.get_balance(rc.get_ally_team()) >= 2000:
                x = random.randint(0, self.map.height-1)
                y = random.randint(0, self.map.height-1)
                if (rc.can_build_tower(self.build, x, y)):
                    rc.build_tower(self.build, x, y)
                    self.build = TowerType.BOMBER
        
        elif self.build == TowerType.BOMBER:
            if rc.get_balance(rc.get_ally_team()) >= 1800:
                (k,(x,y)) = best_coverage(self, self.map, 10)
                print(x,y, end='\r')
                if (rc.can_build_tower(self.build, x, y)):
                    rc.build_tower(self.build, x, y)
                    self.remove_dist_dict_item(k,(x,y))
                    self.build = TowerType.SOLAR_FARM
    
    def towers_attack(self, rc: RobotController):
        towers = rc.get_towers(rc.get_ally_team())
        for tower in towers:
            if tower.type == TowerType.BOMBER:
                rc.auto_bomb(tower.id)
