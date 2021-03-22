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
from .callbacks import trackNbombs
from .callbacks import trackBombLoca
#from sklearn.ensemble import GradientBoostingRegressor
# HistGradientBoostingRegressor is still experimental requieres:
# explicitly require this experimental feature
#from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
#from sklearn.ensemble import HistGradientBoostingRegressor
#------------------------------------------------------------------------------#
#                               Feature indexes                                #
#------------------------------------------------------------------------------#
i_coin_dis     = 4
i_crate_dis    = 7
i_ncrates_exp  = 10
i_bomb_avail   = 11
i_bomb_harms   = 12
i_bomb_badpos  = 13 #25 if trackBombLoca True
i_wait         = 14
i_enemy_dis    = 15
i_nenemies_exp = 18

print_events = True


#------------------------------------------------------------------------------#
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
ITS_A_TRAP             = 'ITS_A_TRAP'
MOVED_INTO_DEAD_END    = 'MOVED_INTO_DEAD_END'
MOVED_TO_FREE_WAY      = 'MOVED_TO_FREE_WAY'
MOVED_TOWARDS_COIN1    = 'MOVED_TOWARDS_COIN1'
MOVED_AWAY_FROM_COIN1  = 'MOVED_AWAY_FROM_COIN1'
MOVED_TOWARDS_CRATE1   = 'MOVED_TOWARDS_CRATE1'
MOVED_AWAY_FROM_CRATE1 = 'MOVED_AWAY_FROM_CRATE1'

BOMB_WITH_NO_TARGET    = 'BOMB_WITH_NO_TARGET'
HOLD_BOMB_NO_TARGET    = 'HOLD_BOMB_NO_TARGET'
MOVED_AWAY_FROM_DANGER = 'MOVED_AWAY_FROM_DANGER'

TARGETED_ENEMY         = 'TARGETED_ENEMY'
MOVED_TOWARDS_ENEMY1   = 'MOVED_TOWARDS_ENEMY1'
MOVED_AWAY_FROM_ENEMY1 = 'MOVED_AWAY_FROM_ENEMY1'

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
    self.nFeatures = 4 + (3 * trackNcoins) + (3 * trackNcrates) + 3 + (trackBombLoca * 3 * trackNbombs) + 2 + 4

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
    
    # if new_game_state['step'] == 1:
    #     print('Features:', new_features)
    #     print('Bombs action available: ',  new_features[0,23]==1)
    #     print('Good spot for bomb: ',  new_features[0,28]==1)
    #     print('Escape routes: ',  new_features[0,[0,1,2,3,29]])

    if new_game_state['step'] > 1:
        _, _, _, location = old_game_state['self']
        location_history.append(location)

        old_features = state_to_features(old_game_state)
        new_features = state_to_features(new_game_state)

        # print('self location', location)
        # print('Features:', old_features)
        # print('Bombs action available: ',  old_features[0,23]==1)
        # print('Good spot for bomb: ',  old_features[0,28]==1)
        # print('Escape routes: ',  old_features[0,[0,1,2,3,29]])
        
        #print('self_action', self_action)
        #print('events: ', events)
        # if self_action is None:
        #     print('events: ', events)

        # add old and new state
        # self.trainingXold = np.vstack((self.trainingXold, state_to_features(old_game_state)))
        # self.trainingXnew = np.vstack((self.trainingXnew, state_to_features(new_game_state)))
        self.trainingXold = np.vstack((self.trainingXold, old_features))
        self.trainingXnew = np.vstack((self.trainingXnew, new_features))

        self.terminal = np.vstack((self.terminal, False))

        # Idea: Add your own events to hand out rewards
        # Penalize moving back and forth
        #reward_moving_back(self, events, new_game_state)
        # If the agent is trapped penalize
        reward_its_a_trap(self, self_action, events, new_game_state)
        # Rewards according to coin and crate position
        reward_moving_to_coin(self, events, new_game_state)
        # Rewards according to enemy position
        reward_enemy_targeting(self, events, new_game_state)
        reward = reward_from_events(self, events)
        # Index: find if state was already present in dataset
        idx_action = ACTIONS.index(self_action) if self_action is not None else 0
        # Update tables with Q valuesm rewards and action
        update_stepTables(self, idx_action, reward)
        if print_events:
            print('self_action', self_action, ' reward: ', reward)
            print('events: ', events)



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
    #reward_moving_back(self, events, last_game_state)
    # If the agent is trapped penalize
    reward_its_a_trap(self, last_action, events, last_game_state)
    # Rewards according to coin position
    reward_moving_to_coin(self, events, last_game_state)
    # Rewards according to enemy position
    reward_enemy_targeting(self, events, last_game_state)
    reward = reward_from_events(self, events)
    
    
    # Index: find if state was already present in dataset
    if print_events:
        print('last_action', last_action, ' reward: ', reward)
        print('events: ', events)

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

    bomb_drop_weight = get_bomb_drop_weight(self, events)

    game_rewards = {
        # Movement
        e.MOVED_LEFT     : -1,
        e.MOVED_RIGHT    : -1,
        e.MOVED_UP       : -1,
        e.MOVED_DOWN     : -1,
        e.WAITED         : -50,
        e.INVALID_ACTION : -100,

        # Distance to targets
        MOVED_TOWARDS_COIN1   : 20,
        MOVED_AWAY_FROM_COIN1 : -40,
        MOVED_TOWARDS_CRATE1   : 20,
        MOVED_AWAY_FROM_CRATE1 : -10,

        # Bomb related
        #e.BOMB_DROPPED         : -25,
        #BOMB_WITH_NO_TARGET    : -500,
        e.BOMB_DROPPED         : 25 * bomb_drop_weight,
        HOLD_BOMB_NO_TARGET    : 5,
        ITS_A_TRAP             : -1000,
        MOVED_AWAY_FROM_DANGER : 500,
        MOVED_INTO_DEAD_END    : -100,
        MOVED_TO_FREE_WAY      : 50,

        TARGETED_ENEMY         : 200,
        MOVED_TOWARDS_ENEMY1   : 30,
        MOVED_AWAY_FROM_ENEMY1 : -5,
        
        #BOMB_EXPLODED : 

        #e.CRATE_DESTROYED : 50,
        #e.COIN_FOUND      : 50,
        e.COIN_COLLECTED  : 400,


        e.KILLED_SELF : -1000,
        e.GOT_KILLED  : -2000
        #KILLED_OPPONENT = 'KILLED_OPPONENT'
        #OPPONENT_ELIMINATED = 'OPPONENT_ELIMINATED'
        #e.SURVIVED_ROUND : 100

    }    

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def get_bomb_drop_weight(self, events):
    crate_num = 0
    # Bomb action available (i_bomb_avail) and crates that will explode (i_ncrates_exp)
    if self.trainingXold[-1, i_bomb_avail] == 1 and self.trainingXnew[-1,i_bomb_avail] == 0:
        crate_num = self.trainingXold[-1, i_ncrates_exp]
        #print('crates to explode: ', crate_num)
    #if 'INVALID_ACTION' in events or self.trainingXold[-1, i_bomb_badpos] == 0:
    #    # Check if bomb dropped was ivalid or in a bad spot
    #    crate_num = 0
    if 'INVALID_ACTION' in events:
       # Check if bomb dropped was ivalid or in a bad spot
       crate_num = 0

    # Reward holding a bomb when no target on sight and bomb option is available
    #if 'BOMB_DROPPED' not in events and self.trainingXold[-1, i_ncrates_exp] ==0:
    if 'BOMB_DROPPED' not in events and self.trainingXold[-1, i_bomb_avail] == 1 and self.trainingXold[-1, i_ncrates_exp] == 0:
        events.append(HOLD_BOMB_NO_TARGET)

    return crate_num if crate_num > 0 else -20





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


def reward_its_a_trap(self, action, events, new_game_state):
    # i_coin_dis  = 4
    # i_crate_dis = 7
    # i_ncrates_exp = 10
    # i_bomb_avail  = 11
    # i_bomb_harms  = 12
    # i_bomb_ticker = 13
    # i_bomb_badpos = 16
    # i_wait        = 17
    
    #--------------------------------------------------------------------------#
    #                     Penalize if moving into a trap                       #
    #--------------------------------------------------------------------------#
    printhelp = True
    print("movement tiles old: ", self.trainingXold[-1, [0,1,2,3,i_wait]]) if printhelp else None
    print("movement tiles new: ", self.trainingXnew[-1, [0,1,2,3,i_wait]]) if printhelp else None

    # Check if Bomb action was done (i_bomb_avail), and it was a bad spot (i_bomb_badpos)
    # This check is reduncdant with the -500 penalization by dropping a bomb 
    # in a bad spot in function get_bomb_drop_weight
    #if self.trainingXold[-1, i_bomb_badpos] == 0 and action == 'BOMB' and self.trainingXold[-1, i_nenemies_exp] == 0:
    if self.trainingXold[-1, i_bomb_badpos] == 0 and action == 'BOMB':
       events.append(ITS_A_TRAP)
       print("ITS_A_TRAP: Bad spot") if printhelp else None
    
    # Check if the seleced path was a dead end and trap
    if all(self.trainingXnew[-1, [0,1,2,3,i_wait]] == -1):
        events.append(ITS_A_TRAP)
        print("ITS_A_TRAP: dead end") if printhelp else None
    # Moving into danger
    #print("Bomb can harm old: ", self.trainingXold[-1, i_bomb_harms]==1,  self.trainingXold[-1, i_bomb_harms])
    #print("Bomb can harm new: ", self.trainingXnew[-1, i_bomb_harms]==1,  self.trainingXnew[-1, i_bomb_harms])
    
    #!WARNING need to add a new feature to find if it's been targeted by an anemy
    # and avoid penalizing
    # Check if the last movement was a good movement (did not move into a harmful position)
    idx = np.where(self.trainingXold[-1, [0,1,2,3,i_wait,i_bomb_avail]] >= [0,0,0,0,0,1])[0]
    #print('action in ACTIONS', action in [ACTIONS[i] for i in idx])
    if action not in [ACTIONS[i] for i in idx] and self.trainingXnew[-1, i_bomb_harms] == 1:
        events.append(ITS_A_TRAP)
        print("ITS_A_TRAP: move in wrong dir") if printhelp else None
    elif self.trainingXold[-1, i_bomb_harms] == 0 and self.trainingXnew[-1, i_bomb_harms] == 1  and action != 'BOMB' :
        events.append(ITS_A_TRAP)
        print("ITS_A_TRAP: moving into danger zone") if printhelp else None
    
    #--------------------------------------------------------------------------#
    #                  Reward if moving out or darger zone                     #
    #--------------------------------------------------------------------------#

    if 'INVALID_ACTION' not in events and 'ITS_A_TRAP' not in events:
        if self.trainingXold[-1, i_bomb_harms] == 1 and self.trainingXnew[-1, i_bomb_harms] == 0:
            # Check if moving away from danger, bomb is harmfull (12) ==1
            print("MOVED_AWAY_FROM_DANGER: moved from danger to safety") if printhelp else None
            events.append(MOVED_AWAY_FROM_DANGER)
        elif any(self.trainingXnew[-1, [0,1,2,3]] >= 0) and self.trainingXnew[-1, i_bomb_harms] == 1 and 'BOMB_DROPPED' not in events and action != 'WAIT':
            # Check if in danger zone and there's scape route
            print("MOVED_AWAY_FROM_DANGER: in danger zone but still a way to scape") if printhelp else None
            events.append(MOVED_AWAY_FROM_DANGER)
        elif self.trainingXold[-1, i_wait] == 0 and all(self.trainingXold[-1, [0,1,2,3]] == -1) and action == 'WAIT':
            # Check if staying waiting is the only action
            print("MOVED_AWAY_FROM_DANGER: Waiting is the best option to escape") if printhelp else None
            events.append(MOVED_AWAY_FROM_DANGER)
        # elif any(self.trainingXnew[-1, [0,1,2,3,i_wait]] == 0) and 'BOMB_DROPPED' in events:
        #     # Check if bomb was dropped and there's scape route
        #     #print("placed a bomb but there is a escape route")
        #     events.append(MOVED_AWAY_FROM_DANGER)

    #--------------------------------------------------------------------------#
    #         Penalize or reward if moving into a dead end free way            #
    #--------------------------------------------------------------------------#
    # This reward only works of the option find dead ends (line 28 in callbacks is True)
    # It has to be fine tune, because in some cases produces and infinite reward loop 
    # by moving away from a dead end and then comming back to the old spot 
    # because the target was closer to it:
    # e.g.:
    # self_action LEFT  reward:  39
    # events:  ['MOVED_LEFT', 'MOVED_TO_FREE_WAY', 'MOVED_AWAY_FROM_CRATE1']
    # old_moves:  [-1.  1. -1.  1.]
    # self_action RIGHT  reward:  19
    # events:  ['MOVED_RIGHT', 'MOVED_TOWARDS_CRATE1']
    # old_moves:  [1. 0. 1. 1.]
    # self_action LEFT  reward:  39
    # events:  ['MOVED_LEFT', 'MOVED_TO_FREE_WAY', 'MOVED_AWAY_FROM_CRATE1']
    # old_moves:  [-1.  1. -1.  1.]
    # self_action RIGHT  reward:  19
    # events:  ['MOVED_RIGHT', 'MOVED_TOWARDS_CRATE1']

    #old_moves = self.trainingXold[-1, [0,1,2,3,i_wait]]
    old_moves = self.trainingXold[-1, [0,1,2,3]]

    if 'INVALID_ACTION' not in events and 'ITS_A_TRAP' not in events and action != 'BOMB' and action != 'WAIT':
        if np.any(old_moves == 1) and old_moves[ACTIONS.index(action)] == 0:
            # if the selected action was dead end == 0, if it was a better way == 1 was available
            events.append(MOVED_INTO_DEAD_END)
        
        elif np.any(old_moves == 0) and old_moves[ACTIONS.index(action)] == 1 :
            # if the selected action was free way == 1, and a dead end was also avaialbe
            events.append(MOVED_TO_FREE_WAY)        


def reward_moving_to_coin(self, events, new_game_state):
    #print('coin features old: ', self.trainingXold[-1, 4:10])
    #print('coin features new: ', self.trainingXnew[-1, 4:10])
    if 'COIN_COLLECTED' not in events and len(new_game_state['coins']) > 0 and 'COIN_FOUND' not in events and self.trainingXold[-1, i_coin_dis] > 0:
        # Better to check if coin dis not 0 in old 
        if self.trainingXnew[-1, i_coin_dis] < self.trainingXold[-1, i_coin_dis]:
            events.append(MOVED_TOWARDS_COIN1)
        else:
            events.append(MOVED_AWAY_FROM_COIN1)
    
    
    #if 'CRATE_DESTROYED' not in events and 'ITS_A_TRAP' not in events and 'BOMB_DROPPED' not in events:     
    # Checks also if there is at leat one crate
    if 'CRATE_DESTROYED' not in events and 'BOMB_DROPPED' not in events and np.any(new_game_state['field'] == 1) :     
        if self.trainingXnew[-1, i_crate_dis] < self.trainingXold[-1, i_crate_dis]:
            events.append(MOVED_TOWARDS_CRATE1)
        else:
            events.append(MOVED_AWAY_FROM_CRATE1)


def reward_enemy_targeting(self, events, new_game_state):
    #i_enemy_dis    = 27
    #i_nenemies_exp = 30

    if 'BOMB_DROPPED' in events and self.trainingXold[-1, i_nenemies_exp] > 0 :     
        events.append(TARGETED_ENEMY)

    # Keep at least 1 tile of distance
    if 'KILLED_OPPONENT' not in events and self.trainingXold[-1, i_enemy_dis] > 1:
        if self.trainingXnew[-1, i_enemy_dis] < self.trainingXold[-1, i_enemy_dis]:
            events.append(MOVED_TOWARDS_ENEMY1)
        else:
            events.append(MOVED_AWAY_FROM_ENEMY1)
            


        
def update_stepTables(self, idx_action, reward):
    # add reward as Q value
    empty_Q = np.full((1,len(ACTIONS)), np.nan)
    self.trainingQ = np.vstack((self.trainingQ, empty_Q))
    self.trainingQ[-1, idx_action] = reward
    # add action and reward
    self.rewards = np.vstack((self.rewards, reward))
    self.action = np.vstack((self.action, idx_action))

# TO DO:
# relative distance to bomb