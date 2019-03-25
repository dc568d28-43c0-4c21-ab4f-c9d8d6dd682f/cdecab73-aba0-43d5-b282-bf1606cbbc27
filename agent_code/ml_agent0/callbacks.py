import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from settings import settings
import random
import json

# Must be adjusted
epsilon = 0.8
REWARD_WIN = 20
REWARD_LOSE = -REWARD_WIN
REWARD_DEFAULT = -.5
REWARD_CRATE= 1
REWARD_KILL = REWARD_WIN // settings['max_agents']
REWARD_COIN = 2
REWARD_FORBIDDEN_MOVE = -50

REWARDS =  [
    REWARD_DEFAULT, # 'MOVED_LEFT'
    REWARD_DEFAULT, # 'MOVED_RIGHT',
    REWARD_DEFAULT, # 'MOVED_UP',
    REWARD_DEFAULT, # 'MOVED_DOWN',
    REWARD_DEFAULT, # 'WAITED',
    REWARD_DEFAULT, # 'INTERRUPTED',
    REWARD_FORBIDDEN_MOVE, # 'INVALID_ACTION',

    REWARD_DEFAULT, # 'BOMB_DROPPED',
    0, # 'BOMB_EXPLODED',

    REWARD_CRATE, # 'CRATE_DESTROYED',
    0, # 'COIN_FOUND',
    REWARD_COIN, # 'COIN_COLLECTED',

    REWARD_KILL, # 'KILLED_OPPONENT',
    REWARD_LOSE, # 'KILLED_SELF',

    REWARD_LOSE, # 'GOT_KILLED',
    0, # 'OPPONENT_ELIMINATED',
    REWARD_WIN, # 'SURVIVED_ROUND',
]

# Do not change something here
ENV_SELF = 1
ENV_STONE = 2
ENV_CRATE = 4
ENV_BOMB_1 = 8
ENV_BOMB_2 = 16
ENV_BOMB_3 = 32
ENV_BOMB_4 = 64
ENV_COIN = 128
ENV_EXPLOSION = 256
ENV_PLAYER = 512
ENV_BOMBSPOT = 1024

def build_env(self):
    """
    Flatten the game state to include other players and bombs in a single 2d array
    """
    game_state = self.game_state
    player = game_state['self']
    arena = game_state['arena']
    width = arena.shape[0]
    height = arena.shape[1]
    arena = arena.reshape(arena.size)
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosions']
    explosions = explosions.reshape(explosions.size)
    others = game_state['others']
    env = np.zeros(arena.size, dtype=np.uint16)
    # add self
    env[player[1] * width + player[0]] += ENV_SELF
    env[player[1] * width + player[0]] += ENV_BOMBSPOT if player[3] == 1 else 0
    # add crates
    for i in range(arena.size):
        env[i] += ENV_STONE if arena[i] == -1 else 0
        env[i] += ENV_CRATE if arena[i] == 1 else 0
    # add bombs
    for bomb in bombs:
        env[bomb[1] * width + bomb[0]] += ENV_BOMB_1 if bomb[2] == 1 else 0
        env[bomb[1] * width + bomb[0]] += ENV_BOMB_2 if bomb[2] == 2 else 0
        env[bomb[1] * width + bomb[0]] += ENV_BOMB_3 if bomb[2] == 3 else 0
        env[bomb[1] * width + bomb[0]] += ENV_BOMB_4 if bomb[2] == 4 else 0
    # add coins
    for coin in coins:
        env[coin[1] * width + coin[0]] += ENV_COIN
    # add explosions
    for i in range(arena.size):
        env[i] += ENV_EXPLOSION if explosions[i] > 0 else 0
    # add other players
    for player in others:
        env[player[1] * width + player[0]] += ENV_PLAYER
        env[player[1] * width + player[0]] += ENV_BOMBSPOT if player[3] == 1 else 0
    return env.reshape((1, -1))

def get_valid_actions(game_state):
    actions = [5] # Wait
    self = game_state['self']
    arena = game_state['arena']
    if self[1] > 0 and arena[self[0],self[1] - 1] is 0:
        actions.append(0) # Up
    if self[1] < arena.shape[1] and arena[self[0],self[1] + 1] is 0:
        actions.append(1) # Down
    if self[0] > 0 and arena[self[0] - 1,self[1]] is 0:
        actions.append(2) # Left
    if self[0] < arena.shape[0] and arena[self[0] + 1,self[1]] is 0:
        actions.append(3) # Right
    if self[3] is 0:
        actions.append(4) # Bomb
    # Get the four directly connected tiles to the player

def train_model(self, data_size=10, discount=.95):
    env_size = self.states[0][0].shape[1]
    states_count = len(self.states)
    data_size = min(states_count, data_size)
    inputs = np.zeros((data_size, env_size))
    targets = np.zeros((data_size, len(settings['actions'])))
    for i, j in enumerate(np.random.choice(range(states_count), data_size, replace=False)):
        envstate, action, reward, envstate_next, game_over = self.states[j]
        inputs[i] = envstate
        # There should be no target values for actions not taken.
        targets[i] = self.model.predict(envstate)[0]
        # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
        Q_sa = np.max(self.model.predict(envstate_next)[0])
        if game_over:
            targets[i, action] = reward
        else:
            # reward + gamma * max_a' Q(s', a')
            targets[i, action] = reward + discount * Q_sa
    return inputs, targets


def setup(self):
    size = settings['cols'] * settings['rows']
    self.model = Sequential()
    self.model.add(Dense(size, input_shape=(size, )))
    self.model.add(PReLU())
    self.model.add(Dense(size))
    self.model.add(PReLU())
    # # Must result in a valid action (0-5)
    self.model.add(Dense(len(settings['actions'])))
    self.model.compile(optimizer='adam', loss='mse')
    try:
        self.model.load_weights(f'agent_code/ml_agent0/weights/{self.name}.h5')
    except Exception:
        pass
    self.logger.debug('Finished setup')
    self.states = list()
    self.loss = .0
    self.env = None

def act(self):
    # First set a random action
    self.next_action = random.choice(settings['actions'])
    if np.random.rand() < epsilon:
        return # Have some randomized behaviour to train against each other
    # Then try to predict the result
    self.env = build_env(self)
    prediction = self.model.predict(self.env)[0]
    self.action = np.argmax(prediction)
    self.next_action = settings['actions'][self.action]


def reward_update(self, end=False):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    if len(self.events) is 0 or self.env is None:
        return
    reward = sum(REWARDS[i] for i in self.events)
    self.logger.debug(str(self.events))
    self.logger.debug(str(reward))
    if 13 in self.events or 14 in self.events:
        end = True
    self.states.append((self.env, self.action, reward, build_env(self), end))
    inputs, targets = train_model(self)
    self.model.fit(
        inputs,
        targets,
        epochs=8,
        batch_size=16,
        verbose=0,
    )
    self.loss = self.model.evaluate(inputs, targets, verbose=0)
    # Save the experience into our memory


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    # reward_update(self, True)
    h5file = f'agent_code/ml_agent0/weights/{self.name}.h5'
    json_file = f'agent_code/ml_agent0/weights/{self.name}.json'
    self.model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(self.model.to_json(), outfile)
    print('loss: {}'.format(self.loss))
    self.loss = .0
    self.states = list()
    self.env = None
