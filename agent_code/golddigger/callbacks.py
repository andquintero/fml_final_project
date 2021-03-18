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
random_prob = 0.1
trackNcoins = 3
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
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25])

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

    # extract the field, coin positions and agent information
    field = game_state['field']
    coins = game_state['coins']
    name, score, bomb, location = game_state['self']

    # agent movements (top - right - down - left)
    area = [(0,-1), (1,0), (0,1), (-1,0)]
    # get info for the surroundings of the agent (N-E-S-W)
    sur = [tuple(map(sum, zip(location, n))) for n in area]
    sur_val = np.array([field[c[0], c[1]] for c in sur])

    #print('coins: ', coins)
    #free_space = field == 0
    #--------------------------------------------------------------------------#
    # Field graph
    graph = make_field_graph(field)
    #--------------------------------------------------------------------------#
    # Distance to all coins
    if len(coins)>0:
        # calculate distance to each coin
        #coin_dist = np.array([BFS_SP(graph, location, coin) for coin in coins])
        coin_dist = np.array([calculate_weighted_distance(graph, location, coin) for coin in coins])
        coin_reldis = np.array(coins) - np.array(location)[None,]
        idx = np.argsort(coin_dist)[0:trackNcoins]
        coinf = np.hstack((coin_dist[idx, None], coin_reldis[idx, :])).flatten()
    else:
        coinf = []

    to_fill = trackNcoins*3 - len(coinf)
    coinf = np.hstack((coinf, np.repeat(0, to_fill)))
    #--------------------------------------------------------------------------#
    # Relative distance to all bombs
    #print('game_state', game_state)
    bombs = game_state['bombs']
    trackNbombs = len(game_state['others']) + 1
    #print('game_state[others]', game_state['others'], len(game_state['others']))
    #trackNbombs =  1
    

    # if it can place a bomb
    # if the bomb will harm you
    # ticker
    # relative distance to bomb

    #xx
    if len(bombs)>0:
        # calculate relative distance to each bomb
        bombs_location = [bomb[0] for bomb in bombs]
        bombs_ticker = np.array([bomb[1] for bomb in bombs])
        #bomb_dist = np.array([BFS_SP(graph, location, coin) for coin in coins])
        bomb_reldis = np.array(bombs_location) - np.array(location)[None,]
        # shortest distance in x or y axis
        bomb_mindist = np.amin(np.abs(bomb_reldis), axis=1)
        idx = np.argsort(bomb_mindist)[0:len(bombs)]
        #print('bombsf idx:', idx, type(idx))
        #print('bomb_reldis :', bomb_reldis)
        #print('bombs_location :', bombs_location)

        for i in range(len(bombs_location)):
            print('bomb_mindist[i]', bomb_mindist[i])
            bombl = bombs_location[i]
            if bomb_mindist[i] > 4:
                "NO HARM"
            else:
                # check if there are walls
                field[bombl[0]:location[0]]
                print('bombl', bombl)
                print('location', location)
                print('range bomb x:', field[bombl[0]:location[0], location[1]])
                print('range bomb y:', field[location[0], bombl[1]:location[1]])

            

            field

            bomb_reldis
            # look only at bombs that are closer than 4 tiles

            #print('field :', field)
            #print('bombs_location :', bombl)
            #print('location :', location)
            #print('field location :', field[location])

        #location



        bombsf = np.hstack((bombs_ticker[idx, None], bomb_reldis[idx, :])).flatten()        

    else:
        bombsf = []

    to_fill = 4*3 - len(bombsf)
    # we have to choose if nan or a high number
    bombsf = np.hstack((bombsf, np.repeat(np.nan, to_fill)))
    print('bombsf:', bombsf)
    #--------------------------------------------------------------------------#
    
    features = np.hstack((sur_val, coinf))
    return features.reshape(1, -1)

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


# Dijkstra algorithm (weighted shortest path)
# adapted from https://likegeeks.com/python-dijkstras-algorithm/
def calculate_weighted_distance(graph, start, end):
    
    graph = copy.deepcopy(graph)
    
    def initialize_costs(source, graph):
        nodes = list(graph.keys())
        values = np.repeat(np.inf, len(nodes))
        values[nodes.index(source)] = 0
        return dict(zip(nodes, values))

    def search(source, target, graph, costs, parents={}):
        nextNode = source
        while nextNode != target:
            for neighbor in graph[nextNode]:
                if graph[nextNode][neighbor] + costs[nextNode] < costs[neighbor]:
                    costs[neighbor] = graph[nextNode][neighbor] + costs[nextNode]
                    parents[neighbor] = nextNode
                del graph[neighbor][nextNode]
            del costs[nextNode]
            nextNode = min(costs, key=costs.get)
        return parents

    def backpedal(source, target, searchResult):
        node = target
        backpath = [target]
        path = []
        while node != source:
            backpath.append(searchResult[node])
            node = searchResult[node]
        for i in range(len(backpath)):
            path.append(backpath[-i - 1])
        return path

    def return_distance(path, graph):
        steps = [(path[i], path[i+1]) for i in range(len(path)-1)]
        return np.sum(np.array([graph[x[0]][x[1]] for x in steps]))

    costs = initialize_costs(start, graph)
    parents = search(start, end, graph, costs, parents={})
    path = backpedal(start, end, parents)
    distance = return_distance(path, graph)
    return distance
    

def make_field_graph(field):
    """
    Takes as input the field and returns a graph representation

    :param field:  np.array
    :return: dict
    """

    # agent movements (top - right - down - left)
    area = [(0,-1), (1,0), (0,1), (-1,0)]

    # create graph for possible movements through the field (free tiles and crates)
    x0, y0 = np.where(np.logical_or(field==0, field==1))
    nodes = [(x0[i], y0[i]) for i in range(len(x0))]
    targets = []

    # this time we are creating a graph with weighted edges (1 if next field is a free tile, 0 if)
    for coord in nodes:
        pb = [tuple(map(sum, zip(coord, n))) for n in area]
        targets.append({x: 1 if field[x[0], x[1]] == 0 else 3 for x in pb if x in nodes})
    
    return dict(zip(nodes, targets))   

