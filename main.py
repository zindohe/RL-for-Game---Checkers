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
ACTIONS = [MOVE_lEFT, MOVE_RIGHT]
PLAYER_0, PLAYER_1 = 0, 1

REWARD_STUCK = -60
REWARD_NOT_AVAILABLE = -40
REWARD_CANT_EAT = -20
REWARD_LOSS = -6
REWARD_DEFAULT = -1
REWARD_GAIN = 6
REWARD_WIN = 60

DEFAULT_LEARNING_RATE = 1
DEFAULT_DISCOUNT_FACTOR = 0.5

SPRITE_SIZE = 64

class Environment:
    def __init__(self, text):
        self.states = {}
        lines = text.strip().split('\n')
        self.height = len(lines)
        self.width = len(lines[0])
        for row in range(self.height):
            for col in range(len(lines[row])):
                self.states[(row, col)] = lines[row][col]

    def apply(self, state, pawn, action, player):
        reward = REWARD_DEFAULT
        new_state = None

        # Correspondance de la case
        find = False
        for temp_pawn in state[player]:

            if temp_pawn == pawn:

                find = True
                break

        if not find:

            reward = REWARD_STUCK

        else:

            # Le bouger
            # Calcul coordonéee
            # Column x:
            #   Si gauche x + 1
            #   Si droite x - 1
            # Ligne y:
            #  Joeur 1 y + 1
            #  Joeur 2 y - 1
            # 
            new_player_state = (
                pawn[0] + 1 if player == 0 else pawn[0] - 1, 
                pawn[1] + 1 if action == MOVE_RIGHT else pawn[1] - 1
            )


            for temp_pawn in state[player]:

                if temp_pawn == new_player_state:

                    reward = REWARD_NOT_AVAILABLE
                    break

            for temp_pawn in state[not player]:

                if temp_pawn == new_player_state:

                    new_player_state, reward = self.check_position(new_player_state, state[not player], player)


            if new_player_state[0] == 0 or new_player_state[0] == 9 or new_player_state[1] == 0 or new_player_state[1] == 11:

                reward = REWARD_NOT_AVAILABLE



            print(pawn)
            print(new_player_state)
        

        return state, reward


    def check_position(self, pawn, ennemy_state, player):

        # Créer un state droite
        # Vérifier que ce nouveau n'est pas déja pris
        #   si oui => X on check à gauche
        #   si non => on garde ce state et donne un gros reward

        # Créer un state gauche
        # Vérifier que ce nouveau n'est pas déja pris
        #   si oui => On annule et on donne un reward -20 
        #   si non => on garde ce state et donne un gros reward
        state_right = (
            pawn[0] + 1 if player == 0 else pawn[0] - 1, 
            pawn[1] + 1
        )

        found  = False

        for ennemy_pawn in ennemy_state:

            if ennemy_pawn == state_right:

                found = True
                break

        if not found:

            return state_right, REWARD_GAIN

        state_left = (
            pawn[0] + 1 if player == 0 else pawn[0] - 1, 
            pawn[1] - 1
        )

        for ennemy_pawn in ennemy_state:

            if ennemy_pawn == state_left:

                return pawn, REWARD_CANT_EAT

        return state_left, REWARD_GAIN

class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.policy = Policy(ACTIONS, environment.width, environment.height)
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
        return self.policy.best_action(self.state, player)

    def do(self, pawn, action, player):
        self.previous_state = self.state
        self.previous_pawn = pawn
        self.state, self.reward = self.environment.apply(self.state, pawn, action, player)

        print("reward:", self.reward)
        self.score += self.reward
        self.last_action = action

    def update_policy(self):
        self.policy.update(agent.previous_pawn, agent.state, self.last_action, self.reward)

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
                                activation = 'relu',
                                solver = 'sgd',
                                learning_rate_init = self.learning_rate,
                                max_iter = 1,
                                warm_start = True)

        output_fit_array = []
        output_fit_array.append(np.zeros(width * height).tolist())
        output_fit_array.append([0,0])
        # output_fit_array.append([0,0])

        self.mlp.fit(
            [[0 for x in range(width * height)]], 
            [[item for sublist in output_fit_array for item in sublist]]
        )
        self.q_vector = None

    def __repr__(self):
        return self.q_vector

    def state_to_dataset(self, state, player):
        output = []

        for y in range(self.maxY):

            temp = np.zeros(self.maxX).tolist()

            for x in range(self.maxX):

                if x == 0 or x == self.maxX - 1 or y == 0 or y == self.maxY - 1:

                    temp[x] = -1


            output.append(temp)

        for temp_player_index, temp_player in enumerate(state):

            value = 1 if temp_player_index == player else 2

            for paw_position in temp_player:

                output[paw_position[0]][paw_position[1]] = value

        return [[item for sublist in output for item in sublist]]

    def best_action(self, state, player):
        format_dataset = self.state_to_dataset(state, player)
        self.q_vector = self.mlp.predict(format_dataset)[0]

        test = self.q_vector[:len(self.q_vector) - 2]

        # print(len(format_dataset[0]))
        max = np.argmax(test)
        print(format_dataset[0][max])

        col = (max % 12)
        row = (max // 11)

        return (row, col), self.actions[np.argmax(self.q_vector[-2:])]
        # for test in self.q_vector:

        #     print(test)
        # # print(self.q_vector[0])
        # print(np.argmax(self.q_vector))
        # print(np.argmin(self.q_vector))
        # action = self.actions[np.argmax(self.q_vector)]
        # return pawn, action

    def update(self, previous_state, state, last_action, reward):
        maxQ = np.amax(self.q_vector)
        last_action = ACTIONS.index(last_action)
        print(self.q_vector, maxQ, self.q_vector[last_action])
        self.q_vector[last_action] += reward + self.discount_factor * maxQ

        inputs = self.state_to_dataset(previous_state)
        outputs = np.array([self.q_vector])
        self.mlp.fit(inputs, outputs)


class BoardWindow(arcade.Window):
    def __init__(self, agent):
        super().__init__(agent.environment.width * SPRITE_SIZE,
                         agent.environment.height * SPRITE_SIZE,
                         "Checkers")
        self.agent = agent

    def setup(self):
        self.walls = arcade.SpriteList()
        self.tiles = arcade.SpriteList()
        self.pawns = arcade.SpriteList()

        for state in agent.environment.states:
            if agent.environment.states[state] == '#':
                sprite = arcade.Sprite(":resources:images/tiles/boxCrate.png", 0.5)
                sprite.center_x = sprite.width * (state[1] + 0.5)
                sprite.center_y = sprite.height * (agent.environment.height - state[0] - 0.5)
                self.walls.append(sprite)
            elif agent.environment.states[state] == '.':
                sprite = arcade.Sprite(":resources:images/tiles/sandCenter.png", 0.5)
                sprite.center_x = sprite.width * (state[1] + 0.5)
                sprite.center_y = sprite.height * (agent.environment.height - state[0] - 0.5)
                self.tiles.append(sprite)
            for pawn_0 in agent.state[0]:
                if pawn_0 == state:
                    sprite = arcade.Sprite(":resources:images/items/coinSilver.png", 0.5)
                    sprite.center_x = sprite.width * (state[1] + 0.5)
                    sprite.center_y = sprite.height * (agent.environment.height - state[0] - 0.5)
                    self.pawns.append(sprite)
            for pawn_1 in agent.state[1]:
                if pawn_1 == state:
                    sprite = arcade.Sprite(":resources:images/items/coinGold.png", 0.5)
                    sprite.center_x = sprite.width * (state[1] + 0.5)
                    sprite.center_y = sprite.height * (agent.environment.height - state[0] - 0.5)
                    self.pawns.append(sprite)

    def on_update(self, delta_time):
        pawn, action = self.agent.best_action(0)
        self.agent.do(pawn, action, 0)
        self.agent.update_policy()
        self.setup()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.agent.reset()

    def on_draw(self):
        arcade.start_render()

        self.walls.draw()
        self.tiles.draw()
        self.pawns.draw()
        arcade.draw_text(f"Score: {self.agent.score}", 10, 10, arcade.csscolor.WHITE, 20)

if __name__ == '__main__':
    environment = Environment(BOARD)
    agent = Agent(environment)

    pawn, action = agent.best_action(0)

    print("pawn", pawn)
    print("action", action)

    agent.do(pawn, action, 0)

    

    # window = BoardWindow(agent)
    # window.setup()
    # arcade.run()
