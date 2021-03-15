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
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

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

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    if not os.path.isfile("trainingX.npy"):
        # If starting training from scratch, load dta searned by the rule based agent
        self.trainingX = np.load('initial_guess/trainingX.npy')
        self.trainingQ = np.load('initial_guess/trainingQ.npy')
        self.actionSequence = np.load('initial_guess/actionSequence.npy')
        self.rewards = np.load('initial_guess/rewards.npy')

        # # If starting training from scratch, there is no data
        # # Initial guess, agent is in the bottom right corner and moved up or left
        # #ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        # guess_init_reward = np.array([-1, -10, -10, -1]).reshape(1, -1)
        # guess_init_state  = np.array([0, -1,  -1, 0,  5,  7, 11, 14, 16, 16, 18, 19, 25]).reshape(1, -1)
        # #guess_init_state  = np.array([0, -1,  -1, 0,  -1,  -1, -1,  -1, -1,  -1, -1,  -1, -1]).reshape(1, -1)
        # aug_state, aug_rewards = augment_features(guess_init_state, guess_init_reward)

        # self.trainingX = aug_state
        # self.trainingQ = aug_rewards
        # # Create table indicating the index of the state and action
        # self.actionSequence = np.empty((0,2))
        # self.rewards = np.empty((0,1))
    else:
        self.trainingX = np.load('trainingX.npy')
        self.trainingQ = np.load('trainingQ.npy')
        self.actionSequence = np.load('actionSequence.npy')
        self.rewards = np.load('rewards.npy')



    #self.model.fit(self.trainingX, self.trainingQ)
    self.model.fit(self.trainingX, np.nan_to_num(self.trainingQ))

    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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
    #print("Events, train: ", events)
    #print("self.actionRegressor:", self.model)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    #### WARNING
    # There is a bug in the main code, and the new_game_state is actually the old_game_state
    reward = reward_from_events(self, events)
    #old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    
    #print('old_game_state', new_game_state)
    #print('old_features :', old_features)
    # Index: find if state was already present in dataset
    idx_s = ((self.trainingX == new_features).all(axis=1).nonzero())[0]
    #print('comp states:', self.trainingX == old_features)
    #idx_action = np.where(ACTIONS == self_action)[0]
    idx_action = ACTIONS.index(self_action) if self_action is not None else np.nan

    
    #print('reward :', reward)
    #print('ACTIONS :', ACTIONS)
    #print('index same:', idx_s, idx_s.shape)
    #print('idx_action :', idx_action)
    
    if new_game_state['step'] > 1:
        if len(idx_s) == 0:
            empty_Q = np.full((1,len(ACTIONS)), np.nan)
            self.trainingQ = np.vstack((self.trainingQ, empty_Q))
            self.trainingX = np.vstack((self.trainingX, new_features))
            idx_s = len(self.trainingQ)-1
            self.trainingQ[idx_s, idx_action] = reward
        else:
            idx_s = idx_s[0]
            if np.isnan(self.trainingQ[idx_s, idx_action]):
                self.trainingQ[idx_s, idx_action] = reward
    else:
        idx_s = np.nan
        
    # Create table indicating the index of the state and action
    self.actionSequence = np.vstack((self.actionSequence, np.array([[idx_s, idx_action]])))
    self.rewards = np.vstack((self.rewards, reward))
    print('rewards :', reward)
    #print('self_action :', self.actionSequence)

    #if new_game_state['step'] > 2:
    #    update_Q_values(self)
    #    self.model.fit(self.trainingX, np.nan_to_num(self.trainingQ))

    
    

    #print("train self.trainingQ:", self.trainingQ)
    #print("train self.trainingQ:", self.trainingQ.shape)
    #print("train self:", self)
    #print("train self transitions:", self.transitions)
    #self.model.fit(self.trainingX, self.trainingQ)
    #self.model.fit(self.trainingX, np.nan_to_num(self.trainingQ))


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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))


    reward = reward_from_events(self, events)
    new_features = state_to_features(last_game_state)
    idx_s = ((self.trainingX == new_features).all(axis=1).nonzero())[0]
    idx_action = ACTIONS.index(last_action) 
    
    if len(idx_s) == 0:
        empty_Q = np.full((1,len(ACTIONS)), np.nan)
        self.trainingQ = np.vstack((self.trainingQ, empty_Q))
        self.trainingX = np.vstack((self.trainingX, new_features))
        idx_s = len(self.trainingQ)-1
        self.trainingQ[idx_s, idx_action] = reward
    else:
        idx_s = idx_s[0]
        
    # Create table indicating the index of the state and action
    self.actionSequence = np.vstack((self.actionSequence, np.array([[idx_s, idx_action]])))
    self.rewards = np.vstack((self.rewards, reward))

    update_Q_values(self)
    self.model.fit(self.trainingX, np.nan_to_num(self.trainingQ))

    
    # Save
    np.save('trainingX.npy', self.trainingX)
    np.save('rewards.npy', self.rewards)
    np.save('trainingQ.npy', self.trainingQ)
    np.save('actionSequence.npy', self.actionSequence)
    
    print("train self.trainingQ:", self.trainingQ.shape)
    print("End of round")

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


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
        e.INVALID_ACTION : -100,

        e.COIN_COLLECTED : 100
        #e.TIME_TO_COIN : 100
        #e.KILLED_OPPONENT: 5,
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# def update_Q_values(self, q_old, reward, new_features):
#     """
#     Update Q values depedending on new state and auxiliary rewards

#     """
#     # Weight
#     GAMMA = 0.95
#     # Learning rate
#     ALPHA = 1

#     q_new = reward + (GAMMA * self.model.predict(new_features).max())
#     q_update = q_old + (ALPHA * (q_new - q_old))
#     return q_update  

def update_Q_values(self):
    """
    Update Q values depedending on new state and auxiliary rewards

    """
    # Weight
    GAMMA = 0.95
    # Learning rate
    ALPHA = 1

    
    for i in range(len(self.actionSequence)-1):
        #if not np.isnan(self.actionSequence[i, 0]):
        if not np.any(np.isnan(self.actionSequence[i:i+2, 0])):
            #print('iddddd', self.actionSequence[i, :])
            #print('iddddd+1', self.actionSequence[i+1, :])
            idx = self.actionSequence[i, :].astype(int)
            q_old = np.nan_to_num(self.trainingQ[idx[0], idx[1]])
            #reward = self.rewards[i+1]
            reward = self.rewards[i]
            
            #new_features = self.trainingX[idx[0]+1].reshape(1, -1)
            new_features = self.trainingX[self.actionSequence[i+1, 0].astype(int)].reshape(1, -1)
            #print('new_features', new_features, new_features.shape)

            # Update Q
            q_new = reward + (GAMMA * self.model.predict(new_features).max())
            q_update = q_old + (ALPHA * (q_new - q_old))
            
            self.trainingQ[idx[0], idx[1]] = q_update
        
