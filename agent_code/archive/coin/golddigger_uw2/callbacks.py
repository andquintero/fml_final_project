import os
import pickle
import random

import numpy as np

#------------------------------------------------------------------------------#
# Imported by us
from random import shuffle
import copy
#from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

#from sklearn.ensemble import GradientBoostingRegressor
# HistGradientBoostingRegressor is still experimental requieres:
# explicitly require this experimental feature
#from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
#from sklearn.ensemble import HistGradientBoostingRegressor
#------------------------------------------------------------------------------#
## WARNING!!!!
# if set to True, reset the whole training
#reset = False
random_prob = 0.3
trackNcoins = 3
trackNcrates = 3
#------------------------------------------------------------------------------#

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            self.random_prob = random_prob


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train and random.random() < self.random_prob :
        if self.reset:
            a = self.reseter(self, game_state)
            #print('action train;', a)
            return a
        # 100%: walk in any direction
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    current_features = state_to_features(game_state)
    #print('state_to_features:', current_features)
    model_pred = self.model.predict(current_features)
    #print('model predict:', model_pred)
    return ACTIONS[np.random.choice(np.flatnonzero(model_pred == model_pred.max())) ]
    #return ACTIONS[np.argmax(model_pred)]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    #--------------------------------------------------------------------------#
    #                           Read game state                                #
    #--------------------------------------------------------------------------#

    # extract the field, coin positions and agent information
    field = game_state['field']
    coins = game_state['coins']
    name, score, bomb, location = game_state['self']
    bombs = game_state['bombs']

    # Copy of field to change
    field_ = field.copy()
    # Find if bombs are blocking tiles
    if len(bombs)>0:
        bombs_location = [bomb[0] for bomb in bombs]
        bombs_ticker = np.array([bomb[1] for bomb in bombs])
        # An old bomb blocks the tile
        
        bombs_location_not_self = [bombs_location[i] for i in np.where(location not in bombs_location)[0]]
        for i,j in bombs_location_not_self:
            field_[i,j]=-1

    

    # agent movements (top - right - down - left)
    area = [(0,-1), (1,0), (0,1), (-1,0)]
    # get info for the surroundings of the agent (N-E-S-W)
    sur = [tuple(map(sum, zip(location, n))) for n in area]

    bombs_location = [bomb[0] for bomb in bombs]
    idx = np.where([s in bombs_location for s in sur])[0]

    sur_val = np.array([field[c[0], c[1]] for c in sur])

    if len(idx) > 0:
        sur_val[idx] = -1
    
    #--------------------------------------------------------------------------#
    #                              Field graphs                                #
    #--------------------------------------------------------------------------#
    # Create graph with paths to all crates
  
    graph_empty_field = make_field_graph(np.invert(field_ >= 0)*1)
    graph_walkable    = make_field_graph(field_)
    #print('graph_walkable: ', graph_walkable)
    #--------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------#
    #                        Coin related features                             #
    #--------------------------------------------------------------------------#
    # Distance to all coins
    if len(coins)>0:
        # calculate distance to each coin
        coin_dist = np.array([BFS_SP(graph_walkable, location, coin, return_path=False) for coin in coins])
        coin_reldis = np.array(coins) - np.array(location)[None,]
        idx = np.argsort(coin_dist)[0:trackNcoins]
        coinf = np.hstack((coin_dist[idx, None], coin_reldis[idx, :])).flatten()
    else:
        coinf = []

    to_fill = trackNcoins*3 - len(coinf)
    #coinf = np.hstack((coinf, np.repeat(0, to_fill)))
    coinf = np.hstack((coinf, np.repeat(np.nan, to_fill)))

    #--------------------------------------------------------------------------#
    #                        Crate related features                            #
    #--------------------------------------------------------------------------#
    # Distance to 3 closest crates and number of crates that can be blown away at current position
    xcrate, ycrate = np.where(field==1)
    crates = [(xcrate[i], ycrate[i]) for i in range(len(xcrate))]

    if len(crates) > 0:      
        # Look for closest crate
        crate_reldis = np.array(crates) - np.array(location)[None,]
        crate_dist = np.array([BFS_SP(graph_empty_field, location, crate, return_path=False) for crate in crates])

        idx = np.argsort(crate_dist)[0:trackNcrates]
        cratef = np.hstack((crate_dist[idx, None], crate_reldis[idx, :])).flatten()
    else:
        cratef = []

    to_fill = trackNcrates*3 - len(cratef)
    #cratef = np.hstack((cratef, np.repeat(0, to_fill)))
    cratef = np.hstack((cratef, np.repeat(np.nan, to_fill)))

    # Number of crates that will explode
    crates_to_explode = []
    for direction in area:
        loc = location
        for i in range(1,4):
            neighbor = tuple(map(sum, zip(loc, direction)))
            if field[neighbor[0], neighbor[1]] == -1:
                break
            if neighbor in crates:
                crates_to_explode.append(neighbor)
            loc = neighbor
    cratef = np.hstack((cratef, np.array(len(crates_to_explode))))
    

    #--------------------------------------------------------------------------#
    #                           Bomb related features                          #
    #--------------------------------------------------------------------------#
    #trackNbombs = len(game_state['others']) + 1
    trackNbombs =  1
    # if it can place a bomb
    # if the bomb will harm you
    # ticker
    # relative distance to bomb
    if len(bombs)>0:
        #Relative distance to bomb in X and Y axis
        bomb_reldis = np.array(bombs_location) - np.array(location)[None,]
        # Find if the bomb can harm the player
        bomb_harm = explosion_zone(field, bomb_reldis, bombs_location, location)
        # Features
        bombsf = np.hstack((np.array(bomb_harm)[:, None], bombs_ticker[:, None], bomb_reldis)).flatten()        
    else:
        bombsf = []

    to_fill = trackNbombs*4 - len(bombsf)
    if to_fill > 0:
        #print('bombsf to_fill:', to_fill)
        #print('bombsf nofill:', bombsf)
        # we have to choose if nan or a high number
        bombsf = np.hstack((bombsf, np.repeat(np.nan, to_fill)))

    bombav = np.array(game_state['self'][2]*1) # if the BOMB action is available
    bombsf = np.hstack((bombav, bombsf))
    #print('bombsf:', bombsf)

    #--------------------------------------------------------------------------#
    #                         Bomb placement features                          #
    #--------------------------------------------------------------------------#
    # Find if this is a good location for dropping a bomb 
    # Also returns the direction of scape

    # filter out free fields in agent radius of 4
    free_tiles = list(graph_walkable.keys())
    tile_dis = np.abs(np.array(free_tiles) - np.array(location)[None,])
    idx = np.where(np.sum(tile_dis <= 4, axis=1) == 2)[0]
    free_tiles = [free_tiles[i] for i in idx]

    # calculate if they are accessible and if yes, the shortest path
    # Returns a list of tupples, each tupple has:
    # distance to target, second node in path
    free_tile_dist_and_escape = [BFS_SP(graph_walkable, location, tile) for tile in free_tiles]
    #idx = np.where(np.array(free_tile_dist_and_escape) != None)[0]
    idx = np.where([elem != None for elem in free_tile_dist_and_escape])[0]
    free_tile_dist_and_escape = [free_tile_dist_and_escape[i] for i in idx]
    #print('free_tile_dist_and_escape', free_tile_dist_and_escape)
    free_tile_dist = [freetile[0] for freetile in free_tile_dist_and_escape] # Distance
    free_tile_escape = [freetile[1] for freetile in free_tile_dist_and_escape] # Paths
    #print('free_tile_dist', free_tile_dist)

    free_tiles = [free_tiles[i] for i in idx]

    # long_escapes = np.where(np.array(free_tile_dist) >= 40)[0]
    # short_scapes = np.where(np.sum(np.array(free_tiles) == np.array(location), axis=1) == 0)[0]
    long_escapes = np.array(free_tile_dist) >= 40 # This is a good spot 4 tiles
    short_scapes = np.sum(np.array(free_tiles) == np.array(location), axis=1) == 0 # This is a good spot for escape route
    good_spot = 1 if any(long_escapes) or any(short_scapes) else 0  # good spot =1, bad spot = 0
    #print('free_tile_escape', free_tile_escape)

    #--------------------------------------------------------------------------#
    #                           Bomb escape features                           #
    #--------------------------------------------------------------------------#
    # Safe path is a path that you can reach before bomb explodes
    # len of path should be less or equal than ticker
    # If the end point of the path is the explotion range, then it is not a good path

    # Find which routes to a free tile are a trap!
    if len(bombs)>0:
        # Filter out paths that are not reachable before bomb goes off

        
        # returns a list for each path, 1 Harm, 0 No harm
        danger_last_tiles = []
        for tile_path in free_tile_escape:
            x = []
            for i in range(len(bombs_location)):
                if len(tile_path) < bombs_ticker[i]:
                    # "Dead end"
                    x.append(1)
                else:
                    #print('bombs_location', bombs_location, 'aaaaaa ',  bombs_location[i])
                    x.extend(explosion_zone(field, bomb_reldis[i], [bombs_location[i]], tile_path[-1]))
            danger_last_tiles.append(x)


        #danger_last_tiles = [explosion_zone(field, bomb_reldis, bombs_location, tile_path[-1]) for tile_path in free_tile_escape]
        # if there are no harmful bombs in the last tile
        no_danger_last_tiles = np.where([1 not in danger_last_tile for danger_last_tile in danger_last_tiles])[0]
        good_escape_routes = [free_tile_escape[i] for i in no_danger_last_tiles]
        # get the next step in the good escape routes
        # If the path is len 1, then the best option for this route is to staty still
        good_next_tiles = [route[1] if len(route)>1 else route[0] for route in good_escape_routes ]
        good_next_tiles = list(set(good_next_tiles)) # the the unique good tiles


        #print('good_next_tiles:', good_next_tiles)
        # List of tupples with relative coordinates of next good step
        if len(good_next_tiles) > 0:
            good_step = list(map(tuple, np.array(good_next_tiles) - np.array(location)[None,]))
            good_step = np.array([0 if n in good_step else -1 for n in [(0,-1), (1,0), (0,1), (-1,0), (0,0)]])
        else:
            good_step = np.repeat(-1, 5)
        #print('good_next_ step features to pos:',  good_step)
    else:
        good_step = np.hstack((np.abs(sur_val)*-1, 0))
        #good_step = np.repeat(np.nan, 5)

    #--------------------------------------------------------------------------#
    #                         Return state to features                         #
    #--------------------------------------------------------------------------#
    # print('Feature sur_val n: ', sur_val.shape)
    # print('Feature coinf n: ', coinf.shape)
    # print('Feature cratef n: ', cratef.shape)
    # print('Feature bombsf n: ', bombsf.shape)
    #features = np.hstack((sur_val, coinf, cratef, bombsf, np.array(good_spot), good_step))
    features = np.hstack((good_step[0:4], coinf, cratef, bombsf, np.array(good_spot), good_step[4]))
    #print('features: ', features)
    return features.reshape(1, -1)

def explosion_zone(field, bomb_reldis, bombs_location, location):
    """
    Given a list of bomb locations and one query position, this function tells you if the
    query position is in the blasting area of any bomb
    DANGER  ZONE!

    Args:
        field: The current fielf of the game
        bombs_location: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    #print('location query: ', location)
    # calculate relative distance to each bomb
    bombs_location = np.array(bombs_location)
    #bombs_ticker = np.array([bomb[1] for bomb in bombs])
    #bomb_dist = np.array([BFS_SP(graph, location, coin) for coin in coins])
    bomb_reldis = bombs_location - np.array(location)[None,]
    # shortest distance in x or y axis
    bomb_mindist = np.amin(np.abs(bomb_reldis), axis=1)
    #idx_bombs = np.argsort(bomb_mindist)[0:len(bombs)]

    # Find if the bomb can harm the player
    bomb_harm = []
    loc = np.array(location)
    for i in range(len(bombs_location)):
        # Location of bomb and player
        bombl = bombs_location[i]
        
        if bomb_mindist[i] > 4 or not any(bombl == loc):
            # 'NO HARM'
            bomb_harm.append(0)
        else:
            # select index of the axis in which bomb can harm you
            idx = np.argmax(bombl == loc) # 0 x axis, 1 y axis
            if idx == 0:
                f = min((bombl[1], loc[1]))
                t = max((bombl[1], loc[1]))
                bomb_range = field[loc[0], f:t]
            else:
                f = min((bombl[0], loc[0]))
                t = max((bombl[0], loc[0]))
                bomb_range = field[f:t, loc[1]]
            # check if there are walls
            if len(bomb_range) > 4 or sum(bomb_range == -1) > 0:
                # 'NO HARM'
                bomb_harm.append(0)
            else:
                # 'HARM'
                bomb_harm.append(1)
    return bomb_harm

def look_for_targets_dist(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    #if len(targets) == 0: return None
    if len(targets) == 0: return np.zeros((3)) 

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    #print('coin len:', len(targets))
    
    while len(frontier) > 0:
        current = frontier.pop(0)
        #print('coin current:', current)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            #print('coin dist:', d + dist_so_far[current])
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    #while True:
    #    if parent_dict[current] == start: return current
    #    current = parent_dict[current]
    #print('current: ', current)
    #print('start: ', start)
    #dis = np.sqrt(np.sum((np.array(start) - np.array(current))**2))
    dis = np.array(current) - np.array(start)
    #print('dis:', np.hstack((dis, best_dist)))
    #return (current, dis)
    dis = np.hstack((dis, best_dist))
    #return dis if dis is not None else np.zeros((1,3)) 
    return dis


def make_field_graph(field):
    """
    Takes as input the field and returns a graph representation

    :param field:  np.array
    :return: dict
    """
    field

    # agent movements (top - right - down - left)
    area = [(0,-1), (1,0), (0,1), (-1,0)]

    # create graph for possible movements through the field
    x_0, y_0 = np.where(field == 0)
    zero_vals = [(x_0[i], y_0[i]) for i in range(len(x_0))]
    targets = []
    for coord in zero_vals:
        pb = [tuple(map(sum, zip(coord, n))) for n in area]
        targets.append([x for x in pb if x in zero_vals])
    
    return dict(zip(zero_vals, targets))    



def make_empty_field_graph(game_state):
    """
    Takes as input the field and returns a graph representation

    :param field:  np.array
    :return: dict
    """

    field = game_state['field'].copy()
    bombs = game_state['bombs']
    _, _, _, location = game_state['self']

    bombs_location = [bomb[0] for bomb in bombs]
    bombs_location = [b for b in bombs_location if b != location]
    for i,j in bombs_location:
        field[i,j] = -1 

    # agent movements (top - right - down - left)
    area = [(0,-1), (1,0), (0,1), (-1,0)]

    # create graph for possible movements through the field
    x_0, y_0 = np.where(field == 0)
    zero_vals = [(x_0[i], y_0[i]) for i in range(len(x_0))]
    #zero_vals2 = [z for z in zero_vals if z not in bombs_location]

    targets = []
    for coord in zero_vals:
        pb = [tuple(map(sum, zip(coord, n))) for n in area]
        targets.append([x for x in pb if x in zero_vals])
    
    return dict(zip(zero_vals, targets))  

# define function for BFS
# https://www.geeksforgeeks.org/building-an-undirected-graph-and-finding-shortest-path-using-dictionaries-in-python/
def BFS_SP(graph, start, goal, return_path=True): 
    explored = [] 
    queue = [[start]] 
    if start == goal: 
        return (0, [start]) if return_path else 0
    while queue: 
        path = queue.pop(0) 
        node = path[-1] 
        if node not in explored: 
            neighbours = graph[node] 
            for neighbour in neighbours: 
                new_path = list(path) 
                new_path.append(neighbour) 
                queue.append(new_path) 
                if neighbour == goal: 
                    return (len(new_path)-1, new_path) if return_path else len(new_path)-1
            explored.append(node)
    #return (None, None)
    
