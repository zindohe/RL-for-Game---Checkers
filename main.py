import arcade

BOARD = """
############
#X.X.X.X.X.#
#.X.X.X.X.X#
#X.X.X.X.X.#
#..........#
#..........#
#O.O.O.O.O.#
#.O.O.O.O.O#
#O.O.O.O.O.#
############
"""

MOVE_lEFT, MOVE_RIGHT, EAT_LEFT, EAT_RIGHT = 'ML', 'MR', 'EL', 'ER'
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

    def apply(self, state, action, player):
        if player == PLAYER_0:
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
        else:
            if action == MOVE_lEFT:
                new_state = (state[0]+1, state[1]-1)
            elif action == MOVE_RIGHT:
                new_state = (state[0]+1, state[1]+1)
            elif action == EAT_LEFT:
                new_state = (state[0] + 2, state[1] + 2)
            elif action == EAT_RIGHT:
                new_state = (state[0] + 2, state[1] - 2)

            if new_state in self.states:
                if self.states[new_state] in ['#', 'X']:
                    reward = REWARD_STUCK
                elif self.states[new_state] in ['.']:
                    reward = REWARD_DEFAULT
            else:
                new_state = state
                reward = REWARD_LOSS

        return new_state, reward


class Policy:



class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.policy = Policy(environment.states.keys(), ACTIONS)
        self.reset()

    def reset(self):
        self.state = None #TODO


if __name__ == '__main__':


