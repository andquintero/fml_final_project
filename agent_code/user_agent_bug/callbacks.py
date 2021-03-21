import os
import pickle
import random

import numpy as np

#------------------------------------------------------------------------------#
# Imported by us
from random import shuffle
import copy
#from sklearn.multioutput import MultiOutputRegressor
#from lightgbm import LGBMRegressor


from .getstate import state_to_features
from .getstate import ACTIONS

from .resetRuleBased import reseter
from .getstate import random_prob
from .getstate import trackNcoins
from .getstate import trackNcrates
from .getstate import trackNbombs
from .getstate import trackBombLoca



def setup(self):
    pass
    # if self.train or not os.path.isfile("my-saved-model.pt"):
    #     self.logger.info("Setting up model from scratch.")

    # else:
    #     self.logger.info("Loading model from saved state.")
    #     with open("my-saved-model.pt", "rb") as file:
    #         self.model = pickle.load(file)
    #         self.random_prob = random_prob


def act(self, game_state: dict):
    # todo Exploration vs exploitation
    if self.train and random.random() < self.random_prob :
        if self.reset:
            a = self.reseter(self, game_state)
            #print('action train;', a)
            return a
        # 100%: walk in any direction
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']
