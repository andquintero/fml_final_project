import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS

#------------------------------------------------------------------------------#
# Imported by us
import os
import numpy as np
#from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

from .resetRuleBased import reseter
from .callbacks import random_prob
from .callbacks import trackNcoins
from .callbacks import trackNcrates
#from sklearn.ensemble import GradientBoostingRegressor
# HistGradientBoostingRegressor is still experimental requieres:
# explicitly require this experimental feature
#from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
#from sklearn.ensemble import HistGradientBoostingRegressor
#------------------------------------------------------------------------------#

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

MOVED_TOWARDS_COIN = ['MOVED_TOWARDS_COIN' + str(n) for n in range(1,10)]
MOVED_AWAY_FROM_COIN = ['MOVED_AWAY_FROM_COIN' + str(n) for n in range(1,10)]
MOVED_TOWARDS_CRATE = ['MOVED_TOWARDS_CRATE' + str(n) for n in range(1,4)]
MOVED_AWAY_FROM_CRATE = ['MOVED_AWAY_FROM_CRATE' + str(n) for n in range(1,4)]
MOVED_BACK_AND_FORTH = 'MOVED_BACK_AND_FORTH'
ITS_A_TRAP = 'ITS_A_TRAP'


#------------------------------------------------------------------------------#
#                         Class to run multiple regressor                      #
#------------------------------------------------------------------------------#
class MultiRegression():
    '''
    This class fit one regressor for each column of a 2d input response 
    '''
    def __init__(self, regressors):
        # saves the regressor
        self.regressors = regressors
    
    def fit(self, trainingX, trainingY):
        '''
        fit one regressor for every column of trainingY
        '''
        for i in range(trainingY.shape[1]):
            idx =  ~np.isnan(trainingY[:,i])
            #print("Regressor", i, 'features n=', sum(idx))
            #print('trainingY:', trainingY[idx,i])
            #print('trainingY:', trainingY[0:5])
            self.regressors[i].fit(trainingX[idx,], trainingY[idx,i])
        print(len(self.regressors), " regressors fitted")

    def predict(self, testX):
        '''
        predict from a new set of features
        '''
        y = [regfitted.predict(testX) for regfitted in self.regressors]
        return np.stack(y, axis=1)
#------------------------------------------------------------------------------#


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    # if not os.path.isfile("data/trainingXold.npy"):
    #     # If starting training from scratch, load dta searned by the rule based agent
    #     self.trainingXold = np.load('initial_guess/trainingXold.npy')
    #     self.trainingXnew= np.load('initial_guess/trainingXnew.npy')
    #     self.rewards = np.load('initial_guess/rewards.npy')
    #     self.trainingQ = np.load('initial_guess/trainingQ.npy')
    #     self.action = np.load('initial_guess/action.npy')

    print('self.train:', self.train)
    print('self.reset:', self.reset)
    # Start GradientBoostingRegressor for every action
    reg = [LGBMRegressor(use_missing=False, zero_as_missing=False) for i in range(len(ACTIONS))]
    self.model = MultiRegression(reg)
    self.nFeatures = 4 + (3 * trackNcoins) + (3 * trackNcrates) + 7 + 5
    if self.reset is True:

        self.random_prob = 1
        #self.resetTraining = True
        self.reseter = reseter
        self.current_round = 0

        # If starting training from scratch, there is no data
        self.trainingXold = np.empty((0,self.nFeatures))
        self.trainingXnew = np.empty((0,self.nFeatures))
        self.trainingQ    = np.empty((0,len(ACTIONS)))
        self.rewards      = np.empty((0,1))
        self.action       = np.empty((0,1))
        self.terminal     = np.empty((0,1))


    else:
        self.random_prob = random_prob
        #self.resetTraining = False

        self.trainingXold = np.load('data/trainingXold.npy')
        self.trainingXnew= np.load('data/trainingXnew.npy')
        self.rewards = np.load('data/rewards.npy')
        self.trainingQ = np.load('data/trainingQ.npy')
        self.action = np.load('data/action.npy')
        self.terminal = np.load('data/terminal.npy')

        #print('self.trainingQ:', self.trainingQ)
        self.model.fit(self.trainingXold, self.trainingQ)

location_history = []

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    new_features = state_to_features(new_game_state)
    # if new_game_state['step'] == 1:
    #     print('Features:', new_features)
    #     print('Bombs action available: ',  new_features[0,23]==1)
    #     print('Good spot for bomb: ',  new_features[0,28]==1)
    #     print('Escape routes: ',  new_features[0,29:34])

    if new_game_state['step'] > 1:
        _, _, _, location = old_game_state['self']
        location_history.append(location)

        old_features = state_to_features(old_game_state)

        # print('self location', location)
        # print('Features:', old_features)
        # print('Bombs action available: ',  old_features[0,23]==1)
        # print('Good spot for bomb: ',  old_features[0,28]==1)
        # print('Escape routes: ',  old_features[0,29:34])
        
        #print('self_action', self_action)
        #print('events: ', events)
        # if self_action is None:
        #     print('events: ', events)
        #     print('self_action', self_action)

        # add old and new state
        # self.trainingXold = np.vstack((self.trainingXold, state_to_features(old_game_state)))
        # self.trainingXnew = np.vstack((self.trainingXnew, state_to_features(new_game_state)))
        self.trainingXold = np.vstack((self.trainingXold, old_features))
        self.trainingXnew = np.vstack((self.trainingXnew, new_features))

        self.terminal = np.vstack((self.terminal, False))

        # Idea: Add your own events to hand out rewards
        # Penalize moving back and forth
        reward_moving_back(self, events, new_game_state)
        # If the agent is trapped penalize
        reward_its_a_trap(self, events, new_game_state)
        # Rewards according to coin position
        reward_moving_to_coin(self, events, new_game_state)
        reward_moving_to_crate(self, events, new_game_state)
        reward = reward_from_events(self, events)
        #print('events: ', events)
        # Index: find if state was already present in dataset
        idx_action = ACTIONS.index(self_action) if self_action is not None else 0
        # Update tables with Q valuesm rewards and action
        update_stepTables(self, idx_action, reward)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Add final state
    # Update old and new state tables
    self.trainingXold = np.vstack((self.trainingXold, self.trainingXnew[-1]))
    self.trainingXnew = np.vstack((self.trainingXnew, state_to_features(last_game_state)))
    self.terminal = np.vstack((self.terminal, True))
    
    # Idea: Add your own events to hand out rewards
    # Penalize moving back and forth
    reward_moving_back(self, events, last_game_state)
    # If the agent is trapped penalize
    reward_its_a_trap(self, events, last_game_state)
    # Rewards according to coin position
    reward_moving_to_coin(self, events, last_game_state)
    reward = reward_from_events(self, events)
    
    
    # Index: find if state was already present in dataset
    #print('last_action', last_action)
    #print('events: ', events)

    idx_action = ACTIONS.index(last_action)
    # Update tables with Q valuesm rewards and action
    update_stepTables(self, idx_action, reward)

    # if last_action is not None:
    #     idx_action = ACTIONS.index(last_action)
    #     # Update tables with Q valuesm rewards and action
    #     update_stepTables(self, idx_action, reward)
    # else:
    #     print('last action none:')
    #     update_stepTables(self, 0, reward)
    #     for i in range(1, len(ACTIONS)):
    #         #print('actions left:', i)
    #         i
    
    # Remove duplicated states and actions pairs
    _, unique_pairs = np.unique(np.hstack((self.trainingXold, self.action)), axis=0, return_index=True)
    unique_pairs = np.sort(unique_pairs)
    print('unique_pairs:', unique_pairs.shape)

    self.trainingXold = self.trainingXold[unique_pairs,]
    self.trainingXnew = self.trainingXnew[unique_pairs,]
    self.trainingQ = self.trainingQ[unique_pairs,]
    self.rewards = self.rewards[unique_pairs,]
    self.action = self.action[unique_pairs,]
    self.terminal = self.terminal[unique_pairs]

    # update Q values
    #print('sumsum', np.sum(~np.isnan(self.trainingQ), axis=0))
    #print('sumsum', self.trainingQ)
    
    # if self.reset and all(np.sum(~np.isnan(self.trainingQ), axis=0) > 10):
    #     update_Q_values(self)
    # else:
    #     c
    
    if self.reset is not True:
        #update_Q_values(self)
        self.model.fit(self.trainingXold, self.trainingQ)

        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)

    # print('self.trainingXold:', self.trainingXold.shape)
    # print('self.trainingXnew:', self.trainingXnew.shape)
    # print('self.rewards:', self.rewards.shape)
    # print('self.action:', self.action.shape)
    
    
    #Save
    np.save('data/trainingXold.npy', self.trainingXold)
    np.save('data/trainingXnew.npy', self.trainingXnew)
    np.save('data/rewards.npy', self.rewards)
    np.save('data/trainingQ.npy', self.trainingQ)
    np.save('data/action.npy', self.action)
    np.save('data/terminal.npy', self.terminal )

    #print('self.trainingXold:', self.trainingXold.shape)
    print("End of round, step:", last_game_state['step'])



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT  : -1,
        e.MOVED_RIGHT : -1,
        e.MOVED_UP    : -1,
        e.MOVED_DOWN  : -1,
        e.WAITED      : -10,
        e.INVALID_ACTION : -100,

        MOVED_BACK_AND_FORTH: -300,

        #e.MOVED_TOWARDS_COIN1   : 20,
        #e.MOVED_AWAY_FROM_COIN1 : -40,

        e.BOMB_DROPPED  : 5,
        ITS_A_TRAP      : -500,
        #BOMB_EXPLODED : 

        #e.CRATE_DESTROYED : 50,
        #e.COIN_FOUND      : 50,
        e.COIN_COLLECTED  : 400,

        #KILLED_OPPONENT = 'KILLED_OPPONENT'
        e.KILLED_SELF : -1000,

        e.GOT_KILLED  : -2000,
        #OPPONENT_ELIMINATED = 'OPPONENT_ELIMINATED'
        e.SURVIVED_ROUND : 100



        #e.TIME_TO_COIN : 100
        #e.KILLED_OPPONENT: 5,
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }

    coin_keys = MOVED_TOWARDS_COIN[:trackNcoins] + MOVED_AWAY_FROM_COIN[:trackNcoins]
    coin_vals = np.hstack((np.linspace(60,20,trackNcoins), np.linspace(-120,-40,trackNcoins)))
    
    crate_keys = MOVED_TOWARDS_CRATE[:trackNcrates] + MOVED_AWAY_FROM_CRATE[:trackNcrates]
    crate_vals = np.hstack((np.linspace(20,5,trackNcrates), np.linspace(-20,-5,trackNcrates)))

    game_rewards.update(dict(zip(coin_keys, coin_vals)))
    game_rewards.update(dict(zip(crate_keys, crate_vals)))

    bomb_drop_weight = get_bomb_drop_weight(self)
    game_rewards[e.BOMB_DROPPED] = game_rewards[e.BOMB_DROPPED] * bomb_drop_weight

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def get_bomb_drop_weight(self):
    crate_num = 0
    if self.trainingXold[-1, 23] == 1 and self.trainingXnew[-1,23] == 0:
        crate_num = self.trainingXold[-1, 22]
    return crate_num


def update_Q_values(self):
    """
    Update Q values depedending on new state and auxiliary rewards

    """
    # Weight
    GAMMA = 0.95
    # Learning rate
    ALPHA = 1

    if self.reset is True:
        self.model.fit(self.trainingXold, self.trainingQ)

    q_values = np.amax(self.model.predict(self.trainingXnew), axis=1)
    #q_update = np.zeros(self.trainingQ.shape)
    for i in range(len(self.trainingQ)):
        idx_action = self.action[i].astype(int) 
        q_old = self.trainingQ[i, idx_action]
        reward = self.rewards[i]

        # Update Q
        if self.terminal[i] is not True:
            new_v = q_values[i]
            q_new = reward + (GAMMA * new_v)
            q_update = q_old + (ALPHA * (q_new - q_old))
            #q_update [idx[0], idx[1]] = q_toupdate
        else:
            q_update = reward
        self.trainingQ[i, idx_action] = q_update

def reward_moving_back(self, events, new_game_state):
    if new_game_state['step'] > 3:
        # print(location_history[-4:-1])
        # print('comparison: ', location_history[-3] == location_history[-1])
        # print('comparison elem 1: ', location_history[-3] )
        # print('comparison elem 3: ', location_history[-1] )
        if location_history[-3] == location_history[-1]:
            #print("WARNING!!!! moved back and forth")
            events.append(MOVED_BACK_AND_FORTH)

            #print("events adter moving back", events)

        #print( "unique: ", np.unique(self.trainingXold[(-3,-1), :], axis=0).shape[0] )
        #print( "unique: ", np.unique(self.trainingXold[(-3,-1), :], axis=0))
        #if np.unique(self.trainingXold[(-3,-1), :], axis=0).shape[0] ==1 or np.unique(self.trainingXold[-3:-1], axis=0).shape[0] == 1:
        #if np.unique(self.trainingXold[-3:-1], axis=0).shape[0] == 1:

def reward_its_a_trap(self, events, new_game_state):
    if sum(self.trainingXnew[-1, 0:4] == 0) == 0:
        events.append(ITS_A_TRAP)

    if self.trainingXold[-1, 28] == 0 and self.trainingXold[-1, 23] == 1 and self.trainingXnew[-1,23] == 0:
        events.append(ITS_A_TRAP)
    
    # Check if the seleced path was a dead end
    if all(self.trainingXnew[-1, 29:34] == -1)  :
        events.append(ITS_A_TRAP)
    

def reward_moving_to_coin(self, events, new_game_state):

    coin_coords = np.arange(0,3)*3+4
    remaining_coins = len(new_game_state['coins'])
    tracking = trackNcoins
    if remaining_coins < tracking:
        tracking = remaining_coins

    if 'COIN_COLLECTED' not in events:
        for i in range(tracking):
            if self.trainingXnew[-1, coin_coords[i]] < self.trainingXold[-1, coin_coords[i]]:
                events.append(MOVED_TOWARDS_COIN[i])
            else:
                events.append(MOVED_AWAY_FROM_COIN[i])

def reward_moving_to_crate(self, events, new_game_state):

    crate_coords = np.arange(0,3)*3+13
    remaining_crates = len(np.where(new_game_state['field'] == 1))
    tracking = trackNcrates
    if remaining_crates < tracking:
        tracking = remaining_crates
    
    if 'CRATE_DESTROYED' not in events and self.trainingXnew[-1, 24] == 0:
        for i in range(tracking):
            if self.trainingXnew[-1, crate_coords[i]] < self.trainingXold[-1, crate_coords[i]]:
                events.append(MOVED_TOWARDS_CRATE[i])
            else:
                events.append(MOVED_AWAY_FROM_CRATE[i])


    #if 'COIN_COLLECTED' not in events and len(new_game_state['coins']) > 0:
    #    if self.trainingXnew[-1, 4] < self.trainingXold[-1, 4]:
    #        events.append(e.MOVED_TOWARDS_COIN1)
    #    else:
    #        events.append(e.MOVED_AWAY_FROM_COIN1)

    #    if len(new_game_state['coins']) > 1:
    #        if self.trainingXnew[-1, 7] < self.trainingXold[-1, 7]:
    #            events.append(e.MOVED_TOWARDS_COIN2)
    #        else:
    #            events.append(e.MOVED_AWAY_FROM_COIN2)
        
    #    if len(new_game_state['coins']) > 2:
    #        if self.trainingXnew[-1, 10] < self.trainingXold[-1, 10]:
    #            events.append(e.MOVED_TOWARDS_COIN3)
    #        else:
    #            events.append(e.MOVED_AWAY_FROM_COIN3)

        
def update_stepTables(self, idx_action, reward):
    # add reward as Q value
    empty_Q = np.full((1,len(ACTIONS)), np.nan)
    self.trainingQ = np.vstack((self.trainingQ, empty_Q))
    self.trainingQ[-1, idx_action] = reward
    # add action and reward
    self.rewards = np.vstack((self.rewards, reward))
    self.action = np.vstack((self.action, idx_action))
