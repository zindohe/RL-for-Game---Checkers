import arcade
import numpy as np
from sklearn.neural_network import MLPRegressor
import random

BOARD = """
############
# . . . . .#
#. . . . . #
# . . . . .#
#. . . . . #
# . . . . .#
#. . . . . #
# . . . . .#
#. . . . . #
############
"""

MOVE_lEFT, MOVE_RIGHT = 'ML', 'MR'
ACTIONS = [MOVE_lEFT, MOVE_RIGHT, EAT_LEFT, EAT_RIGHT]
PLAYER_0, PLAYER_1 = 0, 1

REWARD_STUCK = -60
REWARD_LOSS = -6
REWARD_DEFAULT = -1
REWARD_GAIN = 6
REWARD_WIN = 60

DEFAULT_LEARNING_RATE = 1
DEFAULT_DISCOUNT_FACTOR = 0.5

class Environment:
    def __init__(self, text):
        self.states = {}
        lines = text.strip().split('\n')
        self.height = len(lines)
        self.width = len(lines[0])
        for row in range(self.height):
            for col in range(len(lines[row])):
                self.states[(row, col)] = lines[row][col]

    def apply(self, state, action):
        if action == MOVE_lEFT:
            new_state = (state[0]-1, state[1]-1)
        elif action == MOVE_RIGHT:
            new_state = (state[0]-1, state[1]+1)
        elif action == EAT_LEFT:
            new_state = (state[0] - 2, state[1] + 2)
        elif action == EAT_RIGHT:
            new_state = (state[0] - 2, state[1] - 2)

        if new_state in self.states:
            if self.states[new_state] in ['#', 'O']:
                reward = REWARD_STUCK
            elif self.states[new_state] in ['.']:
                reward = REWARD_DEFAULT
        else:
            new_state = state
            reward = REWARD_LOSS
        return new_state, reward

class Policy:
    def __init__(self, actions, width, height,
                 learning_rate = DEFAULT_LEARNING_RATE,
                 discount_factor = DEFAULT_DISCOUNT_FACTOR):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions
        self.maxX = width
        self.maxY = height

        self.mlp = MLPRegressor(hidden_layer_sizes = (8,),
                                activation = 'tanh',
                                solver = 'sgd',
                                learning_rate_init = self.learning_rate,
                                max_iter = 1,
                                warm_start = True)
        self.mlp.fit([[0, 0]], [[0, 0]])
        self.q_vector = None

        def __repr__(self):
            return self.q_vector

        def state_to_dataset(self, state_pawn):
            return np.array([[state_pawn[0] / self.maxX, state_pawn[1] / self.maxY]])

        def best_action(self, state_player):
            pawn = state_player[random.randint(0, len(state_player))]
            self.q_vector = self.mlp.predict(self.state_to_dataset(state_player))[0]
            action = self.actions[np.argmax(self.q_vector)]
            return action

class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.policy = Policy(environment.states.keys(), ACTIONS)
        self.reset()

    def reset(self):
        positions0 = []
        positions1 = []
        for i in range(self.environment.height):
            for j in range(self.environment.width):
                if self.environment.states[(i, j)] == ' ':
                    if i <= 3:
                        positions0.append((i, j))
                    elif i >= 6:
                        positions1.append((i, j))
        self.state = (positions0, positions1)
        self.previous = self.state
        self.score = 0

    def best_action(self, player):
        return self.policy.best_action(self.state)

    def do(self, action, player):
        self.previous_state = self.state
        self.state, self.reward = self.environment.apply(self.state[player], action)
        self.score += self.reward
        self.last_action = action



if __name__ == '__main__':
    environment = Environment(BOARD)
    agent = Agent(environment)