import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from settings import settings
import random
import json
import datetime
import pickle

# Must be adjusted
epsilon = .6
REWARD_WIN = 500
REWARD_LOSE = -300
REWARD_DEFAULT = -1
REWARD_WAIT = REWARD_DEFAULT
REWARD_CRATE= 30
REWARD_KILL = 100
REWARD_COIN = 50
REWARD_FORBIDDEN_MOVE = -2

REWARDS =  [
    REWARD_DEFAULT, # 'MOVED_LEFT'
    REWARD_DEFAULT, # 'MOVED_RIGHT',
    REWARD_DEFAULT, # 'MOVED_UP',
    REWARD_DEFAULT, # 'MOVED_DOWN',
    REWARD_WAIT, # 'WAITED',
    REWARD_DEFAULT, # 'INTERRUPTED',
    REWARD_FORBIDDEN_MOVE, # 'INVALID_ACTION',

    REWARD_DEFAULT, # 'BOMB_DROPPED',
    -REWARD_DEFAULT, # 'BOMB_EXPLODED',

    REWARD_CRATE, # 'CRATE_DESTROYED',
    REWARD_COIN / 5, # 'COIN_FOUND',
    REWARD_COIN, # 'COIN_COLLECTED',

    REWARD_KILL, # 'KILLED_OPPONENT',
    REWARD_LOSE, # 'KILLED_SELF',

    REWARD_LOSE, # 'GOT_KILLED',
    0, # 'OPPONENT_ELIMINATED',
    REWARD_WIN, # 'SURVIVED_ROUND',
]

# Do not change something here
NAME = 'ml_agent1'
ENV_CELL_STATE_SIZE = 5
EPSILON_DECREASE = 0
MAX_STATES = 100

def build_env(self):
    """
    Flatten the game state to include other players and bombs in a single array
    """
    game_state = self.game_state
    player = game_state['self']
    arena = game_state['arena']
    width = arena.shape[0]
    arena = arena.reshape(arena.size)
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosions']
    explosions = explosions.reshape(explosions.size)
    others = game_state['others']
    env = np.zeros(arena.size * ENV_CELL_STATE_SIZE, dtype=np.float32)
    # create state for each cell
    for i in range(arena.size):
        offset = i * ENV_CELL_STATE_SIZE
        x = i % width
        y = i // width
        pos = (x, y)
        # -1 stone, 0 free, 1 crate
        env[offset] = arena[i]
        # 1 is player, 0 is not player
        env[offset + 1] = 1 if player[0] == x and player[1] == y else 0 # TODO: optimize by pretransforming
        # 1 is enemy, 0 is not enemy
        env[offset + 2] = 1 if pos in [(player[0], player[1]) for player in others] else 0 # TODO: optimize by pretransforming
        # 1 is coin, 0 is not coin
        env[offset + 3] = 1 if pos in [(coin[0], coin[1]) for coin in coins] else 0 # TODO: optimize by pretransforming
        # danger level between -1 and 1, 0 is no danger, 1 is complete danger, -1 is complate danger caused by our agent
        for bomb in bombs:
            if pos == (bomb[0], bomb[1]):
                env[offset + 4] = (1 if not self.bomb or self.bomb != pos else -1) * (settings['bomb_timer'] - bomb[2]) / settings['bomb_timer']
        # 1 is explosion, 0 is not explosion
        # env[offset + 5] = explosions[i]
        if explosions[i] > 0:
            env[offset + 4] = 1
    return env.reshape((1, -1))

def get_valid_actions(self):
    actions = [] # [5] # Wait
    player = self.game_state['self']
    arena = self.game_state['arena']
    if arena[player[0],player[1] - 1] == 0:
        actions.append(0) # Up
    if arena[player[0],player[1] + 1] == 0:
        actions.append(1) # Down
    if arena[player[0] - 1,player[1]] == 0:
        actions.append(2) # Left
    if arena[player[0] + 1,player[1]] == 0:
        actions.append(3) # Right
    if player[3] == 1:
        actions.append(4) # Bomb
    return actions

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
    size = settings['cols'] * settings['rows'] * ENV_CELL_STATE_SIZE
    self.model = Sequential()
    self.model.add(Dense(size, input_shape=(size, )))
    self.model.add(PReLU())
    self.model.add(Dense(size))
    self.model.add(PReLU())
    # # Must result in a valid action (0-5)
    self.model.add(Dense(len(settings['actions'])))
    self.model.compile(optimizer='adam', loss='mse')
    try:
        self.model.load_weights(f'agent_code/{NAME}/weights/{self.name}.h5')
    except Exception:
        pass
    self.logger.debug('Finished setup')
    self.states = list()
    self.loss = .0
    self.env = None
    self.bomb = None
    self.bomb_timer = 0
    self.win_loss = np.array([], dtype=np.uint8)
    try:
        self.win_loss = np.append(self.win_loss, np.fromfile(f'agent_code/{NAME}/weights/{self.name}.wins', dtype=np.uint8))
    except Exception:
        pass
    self.episode = 0
    self.start = datetime.datetime.now()

def act(self):
    global epsilon
    valid_actions = get_valid_actions(self)
    # First set a random action
    self.action = random.choice(valid_actions)
    if np.random.rand() < epsilon:
        # decrease epsilon
        epsilon *= 1 - EPSILON_DECREASE
        epsilon = max(.05, epsilon)
    else:
        # Then try to predict the result
        self.env = build_env(self)
        predictions = self.model.predict(self.env)[0]
        for i in range(len(settings['actions'])):
            if i not in valid_actions:
                predictions[i] = -9999
        self.action = np.argmax(predictions)
    # decrease our bomb timer
    self.bomb_timer -= 1
    # delete our bomb position if exploded
    if self.bomb_timer == 0:
        self.bomb = None
    # if we dropped a bomb, save its position
    if self.action == 4:
        self.bomb = (self.game_state['self'][0], self.game_state['self'][1])
        self.bomb_timer = settings['bomb_timer']
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
    if 13 in self.events:
        reward = REWARDS[13]
        self.win_loss = np.append(self.win_loss, [0])
    elif 14 in self.events:
        reward = REWARDS[14]
        self.win_loss = np.append(self.win_loss, [0])
    elif 16 in self.events:
        reward = REWARDS[16]
        self.win_loss = np.append(self.win_loss, [1])
    else:
        reward = sum(REWARDS[i] for i in self.events)
    self.logger.debug(str(self.events))
    self.logger.debug(str(reward))
    if 13 in self.events or 14 in self.events:
        end = True
    self.states.append((self.env, self.action, reward, build_env(self), end))
    if len(self.states) > MAX_STATES:
        del self.states[0]
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
    reward_update(self, True)
    total_reward = sum([state[2] for state in self.states])
    h5file = f'agent_code/{NAME}/weights/{self.name}.h5'
    json_file = f'agent_code/{NAME}/weights/{self.name}.json'
    self.model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(self.model.to_json(), outfile)
    try:
        self.win_loss.tofile(f'agent_code/{NAME}/weights/{self.name}.wins')
    except Exception:
        pass
    end = datetime.datetime.now()
    self.logger.info('Episode {} loss: {}, reward: {}, rounds: {}, wins: {}, win ratio: {}, time: {}'.format(self.episode, self.loss, total_reward, len(self.states), np.count_nonzero(self.win_loss), np.count_nonzero(self.win_loss) / self.win_loss.size, (end - self.start).total_seconds()))
    with open(f'agent_code/{NAME}/weights/{self.name}_history.json', 'a+') as f:
        f.write(json.dumps([self.episode, self.loss, total_reward, int(self.win_loss[-1]), self.events, (end - self.start).total_seconds()]))
        f.write('\n')
    self.start = datetime.datetime.now()
    self.loss = .0
    self.states = list()
    self.env = None
    self.episode += 1
