import os
import pickle
import random

import numpy as np

#------------------------------------------------------------------------------#
# Imported by us
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
    print('model predict:', model_pred)
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

    # agent movements (north - east - south - west)
    #area = [(0,1), (1,0), (0,-1), (-1,0)]
    # agent movements (down - right - top - left)
    #area = [(0,1), (1,0), (0,-1), (-1,0)]
    # agent movements (top - left - down - right)
    area = [(0,-1), (1,0), (0,1), (-1,0)]

    # create graph for possible movements through the field
    x_0, y_0 = np.where(field == 0)
    zero_vals = [(x_0[i], y_0[i]) for i in range(len(x_0))]
    targets = []
    for coord in zero_vals:
        pb = [tuple(map(sum, zip(coord, n))) for n in area]
        targets.append([x for x in pb if x in zero_vals])
    graph = dict(zip(zero_vals, targets))

    # define function for BFS
    # https://www.geeksforgeeks.org/building-an-undirected-graph-and-finding-shortest-path-using-dictionaries-in-python/
    def BFS_SP(graph, start, goal): 
        explored = [] 
        queue = [[start]] 
        if start == goal: 
            return 0
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
                        return len(new_path)-1
                explored.append(node) 

    # calculate distance to each coin
    coin_dist = np.sort(np.array([BFS_SP(graph, location, coin) for coin in coins]))

    # get info for the surroundings of the agent (N-E-S-W)
    sur = [tuple(map(sum, zip(location, n))) for n in area]
    sur_val = np.array([field[c[0], c[1]] for c in sur])

    # define features 
    # we have 13 features in total (4 fields next to the agent) and distances to all coins starting with the closest
    # (if a coin is already collected, its distance is set to 1000)
    features = np.hstack((sur_val, coin_dist))
    to_fill = 13 - len(features)
    #features = np.hstack((features, np.repeat(1000, to_fill)))
    features = np.hstack((features, np.repeat(-1, to_fill)))
    return features.reshape(1, -1)

    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)
