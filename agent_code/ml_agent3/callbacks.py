import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from settings import settings
from random import shuffle
from time import time, sleep
from collections import deque
import random
import json
import datetime
import pickle

# Must be adjusted
epsilon = 0.6
REWARD_WIN = 500
REWARD_LOSE = -300
REWARD_DEFAULT = -1
REWARD_WAIT = REWARD_DEFAULT
REWARD_CRATE= 30
REWARD_KILL = 100
REWARD_COIN = 50
REWARD_FORBIDDEN_MOVE = -2

REWARDS =  [
    REWARD_DEFAULT, # 0 'MOVED_LEFT'
    REWARD_DEFAULT, # 1 'MOVED_RIGHT',
    REWARD_DEFAULT, # 2 'MOVED_UP',
    REWARD_DEFAULT, # 3 'MOVED_DOWN',
    REWARD_WAIT, # 4 'WAITED',
    REWARD_DEFAULT, # 5 'INTERRUPTED',
    REWARD_FORBIDDEN_MOVE, # 6 'INVALID_ACTION',

    REWARD_DEFAULT, # 7 'BOMB_DROPPED',
    -REWARD_DEFAULT, # 8 'BOMB_EXPLODED',

    REWARD_CRATE, # 9 'CRATE_DESTROYED',
    REWARD_COIN / 5, # 10 'COIN_FOUND',
    REWARD_COIN, # 11 'COIN_COLLECTED',

    REWARD_KILL, # 12 'KILLED_OPPONENT',
    REWARD_LOSE, # 13 'KILLED_SELF',

    REWARD_LOSE, # 14 'GOT_KILLED',
    0, # 15 'OPPONENT_ELIMINATED',
    REWARD_WIN, # 16 'SURVIVED_ROUND',
]

# Do not change something here
NAME = 'ml_agent3'
ENV_CELL_STATE_SIZE = 5
EPSILON_DECREASE = 0
MAX_STATES = 100

def look_for_targets(free_space, start, targets, logger=None):
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
    if len(targets) == 0: return None

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
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

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
    actions = [5] # Wait
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

def train_model(self, data_size=40, discount=.8):
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
    # Must result in a valid action (0-5)
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
    self.win_loss = list()
    try:
        self.win_loss = json.load(open(f'agent_code/{NAME}/weights/{self.name}_wins.json', 'r'))
    except Exception:
        pass
    self.episode = 0
    self.start = datetime.datetime.now()
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def act(self):
    global epsilon
    valid_actions = get_valid_actions(self)
    # First set a random action
    self.action = random.choice(valid_actions)
    if np.random.rand() < epsilon:
        # decrease epsilon
        epsilon *= 1 - EPSILON_DECREASE
        epsilon = max(.05, epsilon)
        # Gather information about the game state
        arena = self.game_state['arena']
        x, y, _, bombs_left = self.game_state['self']
        bombs = self.game_state['bombs']
        bomb_xys = [(x,y) for (x,y,t) in bombs]
        others = [(x,y) for (x,y,n,b) in self.game_state['others']]
        coins = self.game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for xb,yb,t in bombs:
            for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i,j] = min(bomb_map[i,j], t)

        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x,y)) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append((x,y))

        # Check which moves make sense at all
        directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                (self.game_state['explosions'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x-1,y) in valid_tiles: valid_actions.append(2)
        if (x+1,y) in valid_tiles: valid_actions.append(3)
        if (x,y-1) in valid_tiles: valid_actions.append(0)
        if (x,y+1) in valid_tiles: valid_actions.append(1)
        if (x,y)   in valid_tiles: valid_actions.append(5)
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if (bombs_left > 0) and (x,y) not in self.bomb_history: valid_actions.append(4)
        self.logger.debug(f'Valid actions: {valid_actions}')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = [0, 1, 2, 3]
        shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                        and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
        crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
        targets = coins + dead_ends + crates
        # Add other agents as targets if in hunting mode or no crates/coins left
        if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
            targets.extend(others)

        # Exclude targets that are currently occupied by a bomb
        targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        if self.ignore_others_timer > 0:
            for o in others:
                free_space[o] = False
        d = look_for_targets(free_space, (x,y), targets, self.logger)
        if d == (x,y-1): action_ideas.append(0)
        if d == (x,y+1): action_ideas.append(1)
        if d == (x-1,y): action_ideas.append(2)
        if d == (x+1,y): action_ideas.append(3)
        if d is None:
            self.logger.debug('All targets gone, nothing to do anymore')
            action_ideas.append(5)

        # Add proposal to drop a bomb if at dead end
        if (x,y) in dead_ends:
            action_ideas.append(4)
        # Add proposal to drop a bomb if touching an opponent
        if len(others) > 0:
            if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
                action_ideas.append(4)
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x,y) and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(1) > 0):
            action_ideas.append(4)

        # Add proposal to run away from any nearby bomb about to blow
        for xb,yb,t in bombs:
            if (xb == x) and (abs(yb-y) < 4):
                # Run away
                if (yb > y): action_ideas.append(0)
                if (yb < y): action_ideas.append(1)
                # If possible, turn a corner
                action_ideas.append(2)
                action_ideas.append(3)
            if (yb == y) and (abs(xb-x) < 4):
                # Run away
                if (xb > x): action_ideas.append(2)
                if (xb < x): action_ideas.append(3)
                # If possible, turn a corner
                action_ideas.append(0)
                action_ideas.append(1)
        # Try random direction if directly on top of a bomb
        for xb,yb,t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])

        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                self.action = a
                break

        # Keep track of chosen action for cycle detection
        if self.action == 4:
            self.bomb_history.append((x,y))
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
        self.win_loss.append(0)
    elif 14 in self.events:
        reward = REWARDS[14]
        self.win_loss.append(0)
    elif 16 in self.events:
        reward = REWARDS[16]
        self.win_loss.append(1)
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
    with open(f'agent_code/{NAME}/weights/{self.name}_wins.json', 'w') as f:
        json.dump(self.win_loss, f)
    end = datetime.datetime.now()
    self.logger.info('Episode {} {}; loss: {}, reward: {}, rounds: {}, wins: {}, win ratio: {}, time: {}'.format(self.episode, 'won' if self.win_loss[-1] == 1 else 'lost', self.loss, total_reward, len(self.states), np.count_nonzero(self.win_loss), np.count_nonzero(self.win_loss) / len(self.win_loss), (end - self.start).total_seconds()))
    with open(f'agent_code/{NAME}/weights/{self.name}_history.json', 'a+') as f:
        f.write(json.dumps([self.episode, self.loss, total_reward, self.win_loss[-1], self.events, (end - self.start).total_seconds()]))
        f.write('\n')
    self.start = datetime.datetime.now()
    self.loss = .0
    self.states = list()
    self.env = None
    self.episode += 1
