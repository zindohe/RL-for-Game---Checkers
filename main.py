import time

import arcade
import numpy as np
from sklearn.neural_network import MLPRegressor
import random
from pawn import Player, Pawn
from logs import GameLogs
import time

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

MOVE_lEFT, MOVE_RIGHT, MOVE_BACKWARD_LEFT, MOVE_BACKWARD_RIGHT = 'ML', 'MR', 'MBL', 'MBR'
ACTIONS = [MOVE_lEFT, MOVE_RIGHT, MOVE_BACKWARD_LEFT, MOVE_BACKWARD_RIGHT]
PLAYER_0, PLAYER_1 = 0, 1

REWARD_STUCK = -60
REWARD_NOT_AVAILABLE = -40
REWARD_CANT_EAT = -20
REWARD_LOSS = -6
REWARD_DEFAULT = -1
REWARD_GAIN = 6
REWARD_WIN = 60

DEFAULT_LEARNING_RATE = 1
DEFAULT_DISCOUNT_FACTOR = 0.000000005

SPRITE_SIZE = 64


class Type:
    WALL = 0
    PLAYER_PLAYING = 1
    PLAYER_ENEMY = 2
    EMPTY = 3

class Environment:
    def __init__(self, text, logs):
        self.states = {}
        lines = text.strip().split('\n')
        self.height = len(lines)
        self.width = len(lines[0])
        self.logs = logs
        for row in range(self.height):
            for col in range(len(lines[row])):
                self.states[(row, col)] = lines[row][col]

    def environment_to_string(self, state, to_move, display = False, to_eat = None):

        new_environment = []

        index = []
        for j in range(self.width):

            index.append(str(j))

        new_environment.append(index)

        for i in range(self.height):
            temp = []
            temp.append(str(i))
            for j in range(self.width):
                temp.append(self.states[(i, j)])
            new_environment.append(temp)

        for temp_player in state:

            value = '*' if temp_player.type else 'x'

            for paw_position in temp_player.pawn:

                p_value = 'm' if paw_position.is_pawn_equal(to_move.x, to_move.y) else value

                if display:
                    p_value = 'e' if paw_position.is_pawn_equal(to_eat.x, to_eat.y) else p_value

                new_environment[paw_position.x + 1][paw_position.y + 1] = p_value

        return "\n".join(list(map(lambda v: "".join(v), new_environment)))

    def check_cooordinates_type(self, x,y, state, player):

        # mur
        if x == 0 or x == 9 or y == 0 or y == 11:

            return Type.WALL

        # is_box_filled 1
        if state[player].is_box_filled(x, y):

            return Type.PLAYER_PLAYING

        # is_box_filled 2
        if state[not player].is_box_filled(x, y):

            return Type.PLAYER_ENEMY

        # vide
        return Type.EMPTY

    def available_action(self, pawn, state, player, current = 0, displayDebug = False, indent = ""):

        availableAction = []
        availableType = []

        for action in ACTIONS:

            if not pawn.king_piece and (action == MOVE_BACKWARD_LEFT or action == MOVE_BACKWARD_RIGHT):

                continue

            new_pawn = state[player].predict_move(action, pawn)
            
            new_position_type = self.check_cooordinates_type(new_pawn.x, new_pawn.y, state, player)
            availableType.append(new_position_type)

            if displayDebug:
                print(indent + "available_action")
                print(indent + "current", current)
                print(indent + "pawn", pawn)
                print(indent + "new_pawn", new_pawn)
                print(indent + "new_position_type", new_position_type)

            if new_position_type == Type.EMPTY:
                availableAction.append(action)                

            if new_position_type == Type.PLAYER_ENEMY and current == 0:

                available_action_under_pawn = self.available_action(new_pawn, state, player, current + 1, displayDebug, indent + "      ")

                if len(available_action_under_pawn[0]) > 0:
                    availableAction.append(action)

        # print("#######################available_action#######################")

        return (availableAction, availableType)

    def apply(self, state, pawn, action, player):

        pawn = state[player].get_pawn(pawn[0], pawn[1])

        if pawn == None:

            return state, REWARD_STUCK

        available_actions, available_types = self.available_action(pawn, state, player, 0)

        if not action in available_actions:

            return state, REWARD_NOT_AVAILABLE

        new_pawn_position = state[player].predict_move(action, pawn)
        reward = REWARD_DEFAULT

        if available_types[available_actions.index(action)] == Type.PLAYER_ENEMY:

            self.logs.add_log(f"************ Debut Operation mangeage ***************")
            self.logs.add_log(f"  Le pion {pawn} du jouer {int(player)} bouge a {action}")
            self.logs.add_log(f"  Le tableau avant")
            self.logs.add_log(self.environment_to_string(state, pawn, True, new_pawn_position))
            self.logs.add_log(f"  Il mange le {new_pawn_position}")

            state[not player].delete_pawn(new_pawn_position.x, new_pawn_position.y)
            under_available_position = self.available_action(new_pawn_position, state, player, 0)

            if len(under_available_position[0]) == 0:

                return state, reward

            new_pawn_position = state[player].predict_move(under_available_position[0][0], new_pawn_position)
            reward = REWARD_GAIN


        #Bouger le pion
        # print(state[player].pawn)
        pawn.move_pawn(new_pawn_position.x, new_pawn_position.y)
        # print(state[player].pawn)

        if new_pawn_position.x == 1 or new_pawn_position.x == 8:

            pawn.king_piece = True

        if available_types[available_actions.index(action)] == Type.PLAYER_ENEMY:
            
            self.logs.add_log(f"  Le tableau après")
            self.logs.add_log(self.environment_to_string(state, new_pawn_position))

            self.logs.add_log(f"Sa position final {new_pawn_position}")
            self.logs.add_log(f"************ Fin Operation mangeage ***************\n\n")

        new_state = None

        return state, reward


class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.policy = Policy(ACTIONS, environment.width, environment.height)
        self.reset()

    def reset(self):
        player0 = Player(0)
        player1 = Player(1)

        for i in range(self.environment.height):
            for j in range(self.environment.width):
                if self.environment.states[(i, j)] == ' ':
                    if i <= 3:
                        player0.add_pawn(i,j)
                    elif i >= 6:
                        player1.add_pawn(i,j)
        self.state = (player0, player1)
        self.previous = self.state
        self.score = 0

    def best_action(self, player):
        return self.policy.best_action(self.state, player)

    def do(self, pawn, action, player):
        self.previous_state = self.state
        self.previous_pawn = pawn
        self.state, self.reward = self.environment.apply(self.state, pawn, action, player)

        # print("reward:", self.reward)
        self.score += self.reward
        self.last_pawn_action = (pawn, action)

        return self.reward

    def update_policy(self, player):
        self.policy.update(agent.state, self.last_pawn_action, self.reward, player)


class Policy:
    def __init__(self, actions, width, height,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 discount_factor=DEFAULT_DISCOUNT_FACTOR):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions
        self.maxX = width
        self.maxY = height

        self.mlp = MLPRegressor(hidden_layer_sizes=(8,),
                                activation='relu',
                                solver='sgd',
                                learning_rate_init=self.learning_rate,
                                max_iter=1,
                                warm_start=True)

        output_fit_array = []
        output_fit_array.append(np.zeros(width * height).tolist())
        output_fit_array.append([0, 0, 0, 0])
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

            for paw_position in temp_player.pawn:
                output[paw_position.x][paw_position.y] = value

        return [[item for sublist in output for item in sublist]]

    def best_action(self, state, player):
        format_dataset = self.state_to_dataset(state, player)
        prediction = self.mlp.predict(format_dataset)[0].tolist()

        self.q_vector = (prediction[:len(prediction) - 4], prediction[-4:])

        # print(len(format_dataset[0]))
        max = np.argmax(self.q_vector[0])
        print(format_dataset[0][max])

        col = (max % 12)
        row = (max // 11)

        return (row, col), self.actions[np.argmax(self.q_vector[1])]

    def update(self, state, last_pawn_action, reward, player):
        max_pawn = np.amax(self.q_vector[0])
        max_action = np.amax(self.q_vector[1])

        last_action = ACTIONS.index(last_pawn_action[1])
        last_pawn = np.argmax(self.q_vector[0])

        # print('self.q_vector : ', self.q_vector[0], '\nmax_pawn : ', max_pawn, '\nmax_action : ', max_action)
        self.q_vector[1][last_action] += reward + self.discount_factor * max_action
        self.q_vector[0][last_pawn] += reward + self.discount_factor * max_pawn

        inputs = self.state_to_dataset(state, player)

        outputs = np.array([self.q_vector[0] + self.q_vector[1]])


        self.mlp.fit(inputs, outputs)


class BoardWindow(arcade.Window):
    def __init__(self, agent):
        super().__init__(agent.environment.width * SPRITE_SIZE,
                         agent.environment.height * SPRITE_SIZE,
                         "Checkers")
        self.agent = agent
        self.current_player = 0

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
            for pawn_0 in agent.state[0].pawn:

                # print(pawn_0.x == state[0] and pawn_0.y == state[1])
                if pawn_0.x == state[0] and pawn_0.y == state[1]:
                    sprite = arcade.Sprite(":resources:images/items/coinSilver.png", 0.5)
                    sprite.center_x = sprite.width * (state[1] + 0.5)
                    sprite.center_y = sprite.height * (agent.environment.height - state[0] - 0.5)
                    self.pawns.append(sprite)
            for pawn_1 in agent.state[1].pawn:
                if pawn_1.x == state[0] and pawn_1.y == state[1]:
                    sprite = arcade.Sprite(":resources:images/items/coinGold.png", 0.5)
                    sprite.center_x = sprite.width * (state[1] + 0.5)
                    sprite.center_y = sprite.height * (agent.environment.height - state[0] - 0.5)
                    self.pawns.append(sprite)

    def on_update(self, delta_time):
        if self.agent.state[0].pawn and self.agent.state[1].pawn:
            pawn, action = self.agent.best_action(self.current_player)
            # print("******************* Start Turn **************************")
            # print("current_player", self.current_player)
            # print("pawn", pawn)
            # print("action", action)
            reward = self.agent.do(pawn, action, self.current_player)
            self.agent.update_policy(self.current_player)
            self.setup()

            if reward == REWARD_DEFAULT or reward == REWARD_GAIN:
                self.current_player = not self.current_player

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
    with GameLogs("./game.logs") as logs:

        logs.add_log("###### New game ##########")
        environment = Environment(BOARD, logs)
        agent = Agent(environment)

        # pawn, action = agent.best_action(0)
        
        # pawn = (3, 1)
        # action = MOVE_RIGHT
        # print("pawn", pawn)
        # print("action", action)
        
        # agent.do(pawn, action, 0)

        # print("agent.state[0].pawn", agent.state[0].pawn)
        
        # agent.update_policy(0)

        
        window = BoardWindow(agent)
        window.setup()
        arcade.run()

        # BOARD2 = """
        # ############
        # # . . . . .#
        # #. . . . . #
        # # . .o.o. .#
        # #. . .o. . #
        # # . .x. . .#
        # #. . . . . #
        # # . . . . .#
        # #. . . . . #
        # ############
        # """

        # player0 = Player(0)
        # player0.add_pawn(3,6)
        # player0.add_pawn(3,8)
        # player0.add_pawn(4,7)
        # player0.add_pawn(4,5)
        # player1 = Player(1)
        # player1.add_pawn(5,6)

        # pawn = Pawn(5, 6)

        # print(environment.available_action(pawn, (player0, player1), 1, 0, True))
