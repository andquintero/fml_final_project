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
        reg = LGBMRegressor(use_missing=False, zero_as_missing=True)
        self.model = MultiOutputRegressor(reg)

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
    random_prob = .7
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

    #print('next coin dir: ', look_for_targets_dist(free_space, location, coins))
    best_coin_dist = look_for_targets_dist(free_space, location, coins)
    #d = look_for_targets(free_space, (x, y), targets, self.logger)
    #look_for_targets(free_space, start, targets, logger=None)

    # define features 
    # we have 13 features in total (4 fields next to the agent) and distances to all coins starting with the closest

    #print("h1: ", sur_val)
    #print("h1: ", best_coin_dist)
    
    features = np.hstack((sur_val, best_coin_dist))
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

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
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
    #dis = np.array(start) - np.array(current)
    dis = np.array(current) - np.array(start)
    #print('dis:', np.hstack((dis, best_dist)))
    #return (current, dis)
    dis = np.hstack((dis, best_dist))
    #return dis if dis is not None else np.zeros((1,3)) 
    return dis