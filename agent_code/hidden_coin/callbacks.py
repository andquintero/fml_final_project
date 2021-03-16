import os
import pickle
import random

import numpy as np

#------------------------------------------------------------------------------#
# Imported by us
from random import shuffle
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
#from sklearn.ensemble import GradientBoostingRegressor
# HistGradientBoostingRegressor is still experimental requieres:
# explicitly require this experimental feature
#from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
#from sklearn.ensemble import HistGradientBoostingRegressor
#------------------------------------------------------------------------------#


#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

class MultiRegression():
    '''
    This class fit one regressor for each column of a 2d input response 
    '''
    def __init__(self, regressor):
        # saves the regressor
        self.regressor = regressor
    
    def fit(self, trainingX, trainingY):
        '''
        fit one regressor for every column of trainingY
        '''
        
        #self.fitted = [self.regressor.fit(trainingX, trainingY[:,i]) for i in range(trainingY.shape[1])]
        self.fitted = []
        for i in range(trainingY.shape[1]):
            idx =  ~np.isnan(trainingY[:,i])
            self.fitted.append(self.regressor.fit(trainingX[idx,], trainingY[idx,i]))

        print(len(self.fitted), " regressors fitted")

    def predict(self, testX):
        '''
        predict from a new set of features
        '''
        y = [self.fitted[i].predict(testX) for i in range(len(self.fitted))]
        #print('y', len(y), type(y))
        #print('y', y)
        #print('y', np.stack(y, axis=1).shape)
        #print(np.hstack(y))
        return np.stack(y, axis=1)




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
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()

        # Start GradientBoostingRegressor for every action
        #reg = HistGradientBoostingRegressor()
        #reg = LGBMRegressor(use_missing=False, zero_as_missing=True)
        reg = LGBMRegressor(use_missing=False, zero_as_missing=False)
        self.model = MultiRegression(reg)

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # print('game_state[field]:', game_state['field'])
    # print('game_state[coins]:', game_state['coins'])
    # print('game_state[self]:', game_state['self'])
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25])

    self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)
    current_features = state_to_features(game_state)
    #print('state_to_features:', current_features)
    model_pred = self.model.predict(current_features)
    #print('model predict:', model_pred)
    #return np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
    
    
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

    # Find next coin
    #print('coins: ', coins)
    free_space = field == 0

    # Distance to all coins
    graph = make_field_graph(field)
    if len(coins)>0:
        # calculate distance to each coin
        coin_dist = np.array([BFS_SP(graph, location, coin) for coin in coins])
        coin_reldis = np.array(coins) - np.array(location)[None,]
        idx = np.argsort(coin_dist)
        coinf = np.hstack((coin_dist[idx, None], coin_reldis[idx, :])).flatten()
    else:
        coinf = []

    to_fill = 9*3 - len(coinf)
    coinf = np.hstack((coinf, np.repeat(np.nan, to_fill)))


    #print('coin f', coinf)
    




    #print('next coin dir: ', look_for_targets_dist(free_space, location, coins))
    #best_coin_dist = look_for_targets_dist(free_space, location, coins)
    
    
    #d = look_for_targets(free_space, (x, y), targets, self.logger)
    #look_for_targets(free_space, start, targets, logger=None)

    # define features 
    # we have 13 features in total (4 fields next to the agent) and distances to all coins starting with the closest

    #print("h1: ", sur_val)
    #print("h1: ", best_coin_dist)
    
    #features = np.hstack((sur_val, best_coin_dist))
    features = np.hstack((sur_val, coinf))
    
    #features = sur_val
    #to_fill = 13 - len(features)
    
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
    x0, y0 = np.where(np.logical_or(nums==0, nums==1))
    nodes = [(x0[i], y0[i]) for i in range(len(x0))]
    targets = []

    # this time we are creating a graph with weighted edges (1 if next field is a free tile, 0 if)
    for coord in nodes:
        pb = [tuple(map(sum, zip(coord, n))) for n in neighboring]
        targets.append({x: 1 if nums[x[0], x[1]] == 0 else 3 for x in pb if x in nodes})
    
    return dict(zip(nodes, targets))   

    
