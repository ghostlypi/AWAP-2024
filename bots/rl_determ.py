from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import TowerType, Team, Tile, GameConstants, SnipePriority, get_debris_schedule
from src.debris import Debris
from src.tower import Tower
import numpy as np
#from bots.our_lib import gen_distance_map, sort_distance_map, best_coverage, distance
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
        acc += 1 if distance(loc, point) <= dist else 0   
    return acc

def best_coverage(p, map,
                  dist : int,
                  min = 0
                ):
    top_coords = None
    top_c = 0
    for i in p.dist_dict.keys():
        if (i > min) and (i < dist):
            for x in p.dist_dict[i]:
                c = coverage(map, x, dist)
                if c > top_c:
                    top_coords = (i,x)
                    top_c = c
    return top_coords

class BotPlayer(Player):
    def __init__(self, map: Map):
        self.bot = None
        self.rl_bot = BotPlayerRL(map)
        self.determ = BotPlayerDeterm(map)
        self.rl = True
        if len(map.path) > 90:
            self.rl = False

    def play_turn(self, rc: RobotController):
        if rc.get_health(rc.get_ally_team()) < rc.get_health(rc.get_enemy_team()):
            self.determ.play_turn(rc)
        else:
            if self.rl:
                self.rl_bot.play_turn(rc)
            else:
                self.determ.play_turn(rc)

class BotPlayerDeterm(Player):
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
        return (1,0)
                
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
        noSolarAvail = False
        noGunshipAvail = False
        noBombAvail = False
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
                                added = True
                                break
                            elif not rc.is_placeable(rc.get_ally_team(), x,y):
                                noSolarAvail = True
                                self.iter_build()
                    else:
                        self.iter_build()
                        noSolarAvail = True
                        
                if not added and self.spawn_dist[0] < self.next_build and self.next_build <= self.spawn_dist[1]:
                    if rc.get_balance(rc.get_ally_team()) >= 1000:
                        top_coord = best_coverage(self, self.map, 60)
                        if top_coord is not None:
                            (k,(x,y)) = top_coord
                            if (rc.can_build_tower(TowerType.GUNSHIP, x, y)):
                                rc.build_tower(TowerType.GUNSHIP, x, y)
                                self.remove_dist_dict_item(k,(x,y))
                                self.iter_build()
                                added = True
                        if not added:
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
                                    elif not rc.is_placeable(rc.get_ally_team(), x,y):
                                        noGunshipAvail = True
                                        self.iter_build()
                            else:
                                self.iter_build()
                                noGunshipAvail = True
                if not added and self.spawn_dist[1] < self.next_build and self.next_build <= self.spawn_dist[2]:
                    if rc.get_balance(rc.get_ally_team()) >= 1750:
                        top_coord = best_coverage(self, self.map, 2)
                        if top_coord is not None:
                            (k,(x,y)) = top_coord
                            if (rc.can_build_tower(TowerType.BOMBER, x, y)):
                                rc.build_tower(TowerType.BOMBER, x, y)
                                self.remove_dist_dict_item(k,(x,y))
                                self.iter_build()
                                added = True
                        if not added:
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
                                    elif not rc.is_placeable(rc.get_ally_team(), x,y):
                                        noBombAvail = True
                                        self.iter_build()
                            else:
                                self.iter_build()
                                noBombAvail = True
                if not added and self.spawn_dist[2] < self.next_build and self.next_build <= self.spawn_dist[3]:
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
                                added = True
                                break
                            elif not rc.is_placeable(rc.get_ally_team(), x,y):
                                noSolarAvail = True
                                self.iter_build()
                    else:
                        self.iter_build()
                        noSolarAvail = True
            else:
                self.update_towers(rc, rc.get_towers(rc.get_ally_team()))

            if noBombAvail and noGunshipAvail and noSolarAvail:
                self.update_towers(rc, rc.get_towers(rc.get_ally_team()))
        
        keys = sorted(list(self.dist_dict.keys()))
        if len(keys) > 0:   
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

weights_string = """2.827065932481925836e-01
3.060831679631703661e-02
-1.417075230301014788e-01
-2.538226975430479815e-02
-1.287652603514207739e-01
4.621934523289698005e-01
1.496876573298402957e-01
-1.768122137938110960e-01
-1.462567212773128178e-01
-1.659829242825937856e-01
3.540820623423724944e-01
-2.767908857123254629e-01
4.047581363712786651e-01
-4.269358685749220506e-01
1.872045824713540474e-01
6.232132067992840474e-02
3.945271871724986834e-02
2.851071727385744436e-01
-1.954937217898792268e-01
-4.321593499905085345e-02
4.304637782908647958e-01
-3.377173665320607476e-01
-1.738322583380363939e-01
2.277921210909042982e-01
7.258401835837341443e-02
-2.750101662072967357e-01
-2.780102807727829473e-01
1.838304928946133676e-01
1.604994506765283768e-01
-3.099728278972776474e-01
5.598637497493652582e-02
7.284138853007759984e-02
-5.376823767134503163e-01
-4.965393604502021940e-01
1.044074818496135704e-01
-1.114624853302386143e-01
-3.570267292896661671e-01
-1.167374420528375367e-01
-4.303172987556800333e-01
-1.113544445234154745e-01
-1.079343068662083072e-01
2.622201557046557330e-01
4.493305880811889752e-01
-3.683202810379553327e-01
-1.071677364088839957e-01
-4.000294363812874066e-01
-2.842209709508769500e-01
3.514704534436030592e-02
1.864757761862795293e-01
3.864183511339102095e-01
-5.745616997688329786e-02
-6.908340779681532151e-02
-3.508155108521958665e-01
4.933264126591777599e-01
4.731370555095338881e-01
-2.774650621230211334e-01
2.823068585802376296e-01
-4.094094977573332628e-01
3.470572007593830666e-01
7.293804435562101229e-02
1.122893884554657662e-01
-4.889429541424498149e-01
-5.931232688147725085e-02
5.107772996600845605e-02
-3.310044511641241360e-01
2.543896845016290209e-01
4.919909248958387327e-01
4.087533332568510369e-01
2.648562116445043468e-01
8.585467468997376450e-02
-3.545880479713787725e-01
-2.577759566186053064e-01
-2.076808599084256635e-01
4.018021653467821475e-01
4.275624603962907555e-01
2.053325547396478434e-01
2.036403579998630198e-01
4.010077862170814411e-01
7.793721589711016806e-02
6.044664262675064625e-02
-1.469906890538141830e-01
-4.211271041633021794e-02
-4.649897466479091523e-01
1.697696418402313556e-01
3.424576312817315671e-01
3.298572228472125278e-01
-4.426467011314798583e-01
-3.091074040424662828e-01
1.181770855649759699e-01
-4.301489289405862060e-01
1.349079428949644388e-01
-2.910019783309144570e-01
-2.683640025326983203e-02
3.259962175949308039e-01
-5.691354690650340942e-02
-4.492205063559574496e-01
-4.619704938714094267e-01
-1.034180125189133603e-01
8.501843894688931869e-02
3.535182556920009045e-01
3.853337078591386744e-01
2.882105258595516473e-01
2.820266288466829474e-01
-1.426379738753551507e-01
-1.920662581421715975e-01
9.387532794192277041e-02
-2.585984649857989703e-01
4.605815386711269444e-01
3.665370097107436553e-01
3.860730083422736492e-01
-7.795191470165740988e-02
-4.007888188482400249e-01
-3.394557480928961368e-01
2.519696105751813020e-01
-1.204207973530360132e-01
-1.743490153476222604e-01
6.918506923978218204e-02
1.077598067819534045e-01
4.019712511430927337e-01
3.720809735566843468e-02
-3.938505719774436642e-01
-3.096760137671755730e-01
-1.099856980020142982e-01
-3.110724669844256995e-01
-4.858580465329588316e-01
-4.961314275334869350e-01
-1.856284563154957379e-01
1.152500289232336605e-01
8.075112294653497003e-02
2.320022790300158011e-01
-2.412144888633451778e-01
1.775369471586436543e-01
3.212702067675318185e-01
-5.900444443846097897e-01
-1.426117563999793059e-02
1.220425638048800443e-01
9.538534665491671394e-02
1.640814260922872636e-01
7.200580068025153047e-03
-3.783662054699207244e-01
-3.852110013343936767e-01
1.259887944588525266e-02
-1.647852934666816926e-01
3.867632908842836281e-01
-4.424841982726781531e-01
-1.373200715273771577e-01
-2.782674616045459626e-01
-1.713818222643507250e-01
9.937503620370821689e-02
4.819338209586035982e-01
-3.013470781456804026e-01
-4.646124784348832915e-01
-2.193301552594218162e-01
3.086830706761070631e-02
5.831213239669458259e-02
2.641343899956672958e-01
3.479361102780140769e-01
-1.394796278090411901e-01
1.223313860884767124e-02
-4.640939707815179283e-01
3.928999665418234088e-02
-6.276910514262312724e-02
-4.833985895973945679e-01
3.631392094269690896e-01
-1.126908318234209894e-01
3.156048102320402338e-01
-2.163759610157227620e-01
-4.789353093060295663e-01
2.118995987955171190e-01
-2.050619032050506052e-01
-1.042075941635606240e-03
2.517327278505058219e-01
-8.316307260541233592e-02
4.947911391845633000e-01
2.055319033028162323e-01
-4.555456836278582999e-01
-4.327158447401645569e-01
-3.322254706030806082e-01
-2.668412065501100106e-01
-2.543829555364808259e-01
6.262883022220863682e-02
-5.006566308802389287e-01
-2.155285916007741998e-01
-2.929871805892316772e-01
6.239342349989626668e-02
-4.910397653884164271e-01
2.708930721914425410e-01
-4.118178001052211767e-01
-4.910303687452985955e-01
-4.909734108964354249e-01
-1.924345939933777005e-01
-4.510427579272926479e-01
5.308671174676204974e-02
-1.401957673710060304e-01
-3.323935054158586588e-01
-2.526353507764708128e-01
-3.903801971293327511e-02
-5.102874054735865306e-01
3.823771721497569720e-01
-2.978463121427087357e-01
-2.904136081213222198e-01
-1.105981126115636037e-01
9.638008792483276910e-02
2.811676544382373244e-01
-4.400663323212824496e-01
-4.262042222135790093e-01
-4.606698307193696218e-01
1.272800041934227244e-02
6.653137129345204626e-03
-2.693318896554695030e-01
1.806637158373909369e-01
3.576491942242087108e-01
2.686311303124626626e-01
-1.077120538171328912e-01
-1.170802848745113034e-01
4.311235668097786089e+01
1.813357105961320714e-01
1.984489833044872364e-01
-2.023038785030950582e-02
2.507985884709118540e-01
-3.151787930720575392e-01
-4.414250905450134432e-01
-2.189975035002696968e-01
-3.761356769544802470e-01
1.155326064932517965e-02
-4.923318163709203832e-01
-7.715134756428726481e-02
1.892189893838767956e-01
3.633775369750420481e-01
3.732621258127606367e-01
4.534016239599317766e-01
2.752308032610869537e-01
1.459584597358581481e+00
2.187170895548683713e+00
2.306209976203301704e-01
1.102640713189205202e-01
1.731347335126024178e-01
-2.991241915841155175e-01
3.956857962825188757e-03
3.128153642362068165e-01
-4.495758914967947772e-01
4.152803634447616332e+01
4.155255752694386473e+01
-3.490956121423468561e-01
4.552230948533897426e-01
-1.484864341382049480e-01
-2.223434234797094788e-04
-4.170715747116384708e-01
-1.672502940565926410e-01
-1.799688846157900812e-01
1.797986563262479742e-01
6.205184820845544014e-01
7.821134238469279465e-03
4.154340561800105025e-01
-2.743896621605268482e-01
1.097901406238855904e-01
-1.481669821288414401e-02
-1.346891511037005829e-01
-4.426472195947805943e-02
2.425126371721769980e-01
-3.254430474824044905e-01
1.048930938457474094e-01
1.008037648009263432e-02
-1.982901180578486855e-01
-1.942513424524190402e-01
-3.988194754239046125e-01
3.850006498719948933e-01
1.375720540658309510e-01
-3.490292070907360933e-01
-6.310328743279104424e-01
2.548057707074647960e-01
3.786226257084951419e-01
2.899844297120788283e-01
3.744416276368580654e-01
4.401401469100298947e-01
1.669606976043913615e-01
-4.506283896456220184e-02
-4.391985597541959163e-01
-4.582144814405190170e-01"""

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, weights=None):
        if weights is not None:
            self.layers = weights
        else:
            self.layers = [np.random.randn(next_size, input_size + 1) 
                           for input_size, next_size in zip([input_size] + hidden_layers, hidden_layers + [output_size])]

    def load_weights_from_file(self, filename, input_size, hidden_layers, output_size):
        all_weights = np.loadtxt(filename)
        layers = []
        idx = 0
        for input_size, next_size in zip([input_size] + hidden_layers, hidden_layers + [output_size]):
            layer_size = (next_size, input_size + 1)
            num_weights = np.prod(layer_size)
            layer_weights = all_weights[idx:idx + num_weights].reshape(layer_size)
            layers.append(layer_weights)
            idx += num_weights
        return layers
    
    def load_weights_from_string(self, weights_string, input_size, hidden_layers, output_size):
        # Convert string to a list of numbers
        all_weights = np.fromstring(weights_string, sep='\n')
        layers = []
        idx = 0
        for input_size, next_size in zip([input_size] + hidden_layers, hidden_layers + [output_size]):
            layer_size = (next_size, input_size + 1)
            num_weights = np.prod(layer_size)
            layer_weights = all_weights[idx:idx + num_weights].reshape(layer_size)
            layers.append(layer_weights)
            idx += num_weights
        return layers

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def backpropagate(self, features, action, td_error):
        # Forward pass
        activations = [np.append(features, 1)]  # Input layer activation with bias
        for layer in self.layers[:-1]:
            x = np.dot(layer, activations[-1])
            x = self.relu(x)
            activations.append(np.append(x, 1))  # Add bias for next layer
        # Output layer without ReLU and bias
        activations.append(np.dot(self.layers[-1], activations[-1]))

        # Compute output layer error
        output_error = np.zeros_like(activations[-1])
        output_error[action] = td_error

        # Initialize gradients
        gradients = [np.zeros_like(layer) for layer in self.layers]

        # Backward pass
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            current_activation = activations[i]
            next_activation = activations[i + 1]

            if i != len(self.layers) - 1:
                # Exclude bias from next activation and apply ReLU derivative
                next_activation_no_bias = next_activation[:-1]
                output_error = output_error * self.relu_derivative(next_activation_no_bias)

            gradients[i] = np.outer(output_error, current_activation)

            # Update error for next layer, skip if it's the input layer
            if i != 0:
                output_error = np.dot(layer.T[:-1], output_error)

        return gradients



    def forward(self, x):
        x = np.append(x, 1)  # Add bias
        for layer in self.layers[:-1]:
            x = self.relu(np.dot(layer, x))  # ReLU activation for hidden layers
            x = np.append(x, 1)  # Add bias
        return np.dot(self.layers[-1], x)  # No activation for output layer

    def update(self, gradients, alpha):
        # Gradient clipping
        clip_value = 1  # You can adjust this value
        for i in range(len(gradients)):
            gradients[i] = np.clip(gradients[i], -clip_value, clip_value)

        for layer, grad in zip(self.layers, gradients):
            layer += alpha * grad


    def save_weights(self, filename='weights.txt'):
        # Flatten and save all layer weights to a file
        flattened_weights = np.concatenate([layer.flatten() for layer in self.layers])
        np.savetxt(filename, flattened_weights)

class BotPlayerRL(Player):
    def __init__(self, map: Map):
        self.map = map
        self.alpha = 0.025  # Learning rate
        self.gamma = 1  # Discount factor
        self.epsilon = 0.0  # Exploration rate
        self.ally_previous_health = 2500  # Starting health
        self.enemy_previous_health = 2500
        self.turn_counter = 0  # Counter to track the number of turns
        self.num_features = 14  # Number of features in your RL agent
        self.next_action = None
        self.current_features = None
        #self.bomber_locations = self.find_best_bomber_locations(self)
        self.bomber_locations = None
        input_size = 17
        hidden_layers = [8, 8]
        output_size = 7
        nn_weights = NeuralNetwork(input_size, hidden_layers, output_size).load_weights_from_string(weights_string, input_size, hidden_layers, output_size)
        self.nn = NeuralNetwork(input_size, hidden_layers, output_size, weights=nn_weights)
        self.action_history = []
        self.feature_history = []
        self.reward_history = []
        self.next_build = 0
        self.spawn_dist = [0, 3, 7]
        self.path_length = len(map.path)
        self.n_turns = self.path_length * 10  # The interval for giving out rewards
        self.path_coverage = (map.width * map.height)

        init_distance_map(self, map)

    def play_turn(self, rc: RobotController):
        if self.bomber_locations is None:
            self.bomber_locations = self.find_best_bomber_locations(rc)
        action = 0
        if self.next_action is not None:
            action = self.next_action
        else:
            self.current_features = self.extract_features(rc)
            action = self.choose_action(self.current_features)
        
        # Execute the chosen action
        played = False
        while True:
            balance = rc.get_balance(rc.get_ally_team())
            if action == 0:
                if balance >= 2000:
                    played = self.place_solar_panel(rc)
            elif action == 1:
                if balance >= 3000:
                    played = self.place_bomber(rc)
            elif action == 2:
                if balance >= 200:
                    played = self.send_big_debris(rc)
            elif action == 3:
                if balance >= 200:
                    played = self.send_debris(rc)
            elif action == 4:
                if balance >= 1000:
                    played = self.place_gunship(rc)
            elif action == 5:
                played = self.sell_tower(rc)
            elif action == 6:
                played = self.send_massive_debris(rc)
            if played:
                self.next_action = None
                played = False
            else:
                self.next_action = action
                break
        # Increment turn counter
        self.turn_counter += 1
        self.feature_history.append(self.current_features)
        self.action_history.append(action)
        # Check if it's time to calculate and apply the reward
        if self.turn_counter % self.n_turns == 0:
            # Calculate delayed reward based on health difference
            self.update_weights_with_history(rc)
        self.towers_attack(rc)

    def update_weights_with_history(self, rc):
        return
        current_ally_health = rc.get_health(rc.get_ally_team())
        current_enemy_health = rc.get_health(rc.get_enemy_team())
        if current_enemy_health == 0:
            current_enemy_health = 1
        #reward = (current_ally_health - current_enemy_health) / current_enemy_health
        reward = current_ally_health - current_enemy_health
        self.reward_history.append(reward)
        cumulative_reward = 0
        for reward in reversed(self.reward_history):
            cumulative_reward = reward + self.gamma * cumulative_reward

        # Initialize accumulated gradients
        accumulated_gradients = [np.zeros_like(layer) for layer in self.nn.layers]

        # Accumulate gradients over history
        for features, action in zip(self.feature_history, self.action_history):
            future_q = max(self.nn.forward(features))
            td_target = cumulative_reward + self.gamma * future_q
            td_error = np.clip(td_target - self.nn.forward(features)[action], -1, 1)
            gradients = self.nn.backpropagate(features, action, td_error)

            # Accumulate gradients
            for i, grad in enumerate(gradients):
                accumulated_gradients[i] += grad

        # Update weights with accumulated gradients
        self.nn.update(accumulated_gradients, self.alpha)

        self.nn.save_weights()

        # Clear history after updating
        self.action_history.clear()
        self.feature_history.clear()
        self.reward_history.clear()

    def distance(self, loc1, loc2):
        return (loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2

    def extract_features(self, rc: RobotController):
        features = np.zeros(17) 

        # Feature 1-3: Number of solar panels, bombers, reinforcers for ally team
        ally_towers = rc.get_towers(rc.get_ally_team())
        features[0] = sum(1 for tower in ally_towers if tower.type == TowerType.SOLAR_FARM)
        features[1] = sum(1 for tower in ally_towers if tower.type == TowerType.BOMBER)
        features[2] = sum(0 for tower in ally_towers if tower.type == TowerType.REINFORCER)

        # Feature 4-6: Number of solar panels, bombers, reinforcers for enemy team
        enemy_towers = rc.get_towers(rc.get_enemy_team())
        features[3] = sum(1 for tower in enemy_towers if tower.type == TowerType.SOLAR_FARM)
        features[4] = sum(1 for tower in enemy_towers if tower.type == TowerType.BOMBER)
        features[5] = sum(0 for tower in enemy_towers if tower.type == TowerType.REINFORCER)

        # Feature 7-8: Health for ally and enemy team
        features[6] = rc.get_health(rc.get_ally_team())
        features[7] = rc.get_health(rc.get_enemy_team())

        # Feature 9-10: Balance for ally and enemy team
        features[8] = rc.get_balance(rc.get_ally_team())
        features[9] = rc.get_balance(rc.get_enemy_team())

        # Feature 11-12: Number of debris targeting ally and enemy team
        features[10] = len(rc.get_debris(rc.get_ally_team()))
        features[11] = len(rc.get_debris(rc.get_enemy_team()))

        features[12] =  sum(1 for tower in ally_towers if tower.type == TowerType.GUNSHIP)
        features[13] =  sum(1 for tower in enemy_towers if tower.type == TowerType.GUNSHIP)

        # features[14] = total path length
        # features[15] = total path coverage
        features[14] = self.path_length
        features[15] = self.path_coverage
        features[16] = self.turn_counter
        return features

    def choose_action(self, features):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3, 4, 5])  # Random action
        else:
            q_values = self.nn.forward(features)
            return np.argmax(q_values)

    def sell_tower(self, rc: RobotController):
        towers = rc.get_towers(rc.get_ally_team())
        for tower in towers:
            if tower.type == TowerType.SOLAR_FARM:
                rc.sell_tower(tower.id)
                return True
        return False
        
    def place_solar_panel(self, rc: RobotController):
        for x in range(self.map.width):
            for y in range(self.map.height):
                if rc.can_build_tower(TowerType.SOLAR_FARM, x, y):
                    rc.build_tower(TowerType.SOLAR_FARM, x, y)
                    return True # Place one solar panel and return
        return False

    def place_bomber(self, rc: RobotController):
        location = self.next_bomber_location(rc)
        if location:
            rc.build_tower(TowerType.BOMBER, location[0], location[1])
            return True
        return False

    #def place_reinforcer(self, rc: RobotController):
    #    location = self.find_best_reinforcer_location(rc)
    #    if location:
    #        rc.build_tower(TowerType.REINFORCER, location[0], location[1])

    def place_gunship(self, rc: RobotController):
        location = self.find_best_gunship_location(rc)
        if location:
            rc.build_tower(TowerType.GUNSHIP, location[0], location[1])
            return True
        return False

    def send_debris(self, rc: RobotController):
        if rc.can_send_debris(1, 45):
            rc.send_debris(1, 45) # or 4, 95
            return True
        return False

    def send_big_debris(self, rc: RobotController):
        if rc.can_send_debris(4, 95):
            rc.send_debris(4, 95) # or 4, 95
            return True
        return False
    
    def send_massive_debris(self, rc: RobotController):
        #get largest debris that can be sent
        for i in range(2500, 0, -100):
            if rc.can_send_debris(6, i):
                rc.send_debris(6, i)
                return True
        return False

    def find_best_bomber_locations(self, rc: RobotController):
        all_locations = []
        for x in range(self.map.width):
            for y in range(self.map.height):
                if rc.is_placeable(rc.get_ally_team(),x, y):
                    coverage = self.calculate_path_coverage(self.map, (x, y), TowerType.BOMBER.range)
                    all_locations.append(((x, y), coverage))
        
        return sorted(all_locations, key=lambda x: x[1], reverse=True)

    def next_bomber_location(self, rc: RobotController):
        for i in range(len(self.bomber_locations)):
            location, coverage = self.bomber_locations[i]
            if rc.can_build_tower(TowerType.BOMBER, location[0], location[1]):
                #del self.bomber_locations[i]
                return location

    def find_best_bomber_location(self, rc: RobotController):
        best_location = None
        max_path_coverage = 0

        for x in range(self.map.width):
            for y in range(self.map.height):
                if rc.can_build_tower(TowerType.BOMBER, x, y):
                    coverage = self.calculate_path_coverage(self.map, (x, y), TowerType.BOMBER.range)
                    if coverage > max_path_coverage:
                        max_path_coverage = coverage
                        best_location = (x, y)

        return best_location
    
    def iter_build(self):
        self.next_build += 1
        self.next_build = self.next_build % (self.spawn_dist[-1] + 1)
    
    def remove_dist_dict_item(self, k, loc):
        self.dist_dict[k].remove(loc)
        if len(self.dist_dict[k]) == 0:
            del self.dist_dict[k]

    def find_best_gunship_location(self, rc: RobotController):
        k = None
        keys = sorted(list(self.dist_dict.keys()))
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
    
    def find_best_reinforcer_location(self, rc: RobotController):
        best_location = None
        max_tower_adjacency = 0

        for x in range(self.map.width):
            for y in range(self.map.height):
                if rc.can_build_tower(TowerType.REINFORCER, x, y):
                    adjacency = self.calculate_tower_adjacency(rc, (x, y))
                    if adjacency > max_tower_adjacency:
                        max_tower_adjacency = adjacency
                        best_location = (x, y)

        return best_location
    

    def calculate_path_coverage(self, map, location, range):
        coverage = 0
        for path_location in map.path:
            if self.distance(location, path_location) <= range ** 2:
                coverage += 1
        return coverage

    def calculate_tower_adjacency(self, rc, location):
        adjacency_count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the current tower location
                adj_x, adj_y = location[0] + dx, location[1] + dy
                if rc.get_map().is_in_bounds(adj_x, adj_y) and self.is_tower_at(rc, adj_x, adj_y):
                    adjacency_count += 1
        return adjacency_count

    def is_tower_at(self, rc, x, y):
        for tower in rc.get_towers(rc.get_ally_team()):
            if tower.x == x and tower.y == y:
                return True
        return False

    def towers_attack(self, rc: RobotController):
        towers = rc.get_towers(rc.get_ally_team())
        for tower in towers:
            if tower.type == TowerType.GUNSHIP:
                rc.auto_snipe(tower.id, SnipePriority.STRONG)
            elif tower.type == TowerType.BOMBER:
                rc.auto_bomb(tower.id)