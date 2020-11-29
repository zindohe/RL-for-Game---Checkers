MOVE_lEFT, MOVE_RIGHT, MOVE_BACKWARD_LEFT, MOVE_BACKWARD_RIGHT = 'ML', 'MR', 'MBL', 'MBR'

class Player:

    def __init__(self, type):
        self.pawn = []
        self.type = type

    def add_pawn(self, x, y):
        self.pawn.append(Pawn(x,y))

    def is_box_filled(self, x, y):

        for pawn in self.pawn:

            if pawn.x == x and pawn.y == y:

                return True

        return False
    
    def get_pawn(self, x, y):

        for pawn in self.pawn:

            if pawn.x == x and pawn.y == y:

                return pawn

        return None

    def delete_pawn(self, x, y):

        self.pawn = list(filter(lambda v: False if v.x == x and v.y == y else True, self.pawn))

    def predict_move(self, action, pawn):

        if action == MOVE_BACKWARD_LEFT or action == MOVE_BACKWARD_RIGHT:

            return Pawn(
                pawn.x + 1 if self.type == 1 else pawn.x - 1,
                pawn.y + 1 if action == MOVE_BACKWARD_RIGHT else pawn.y - 1
            )

        return Pawn(
            pawn.x + 1 if self.type == 0 else pawn.x - 1,
            pawn.y + 1 if action == MOVE_RIGHT else pawn.y - 1
        )

class Pawn:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.king_piece = False

    def move_pawn(self, x, y):
        self.x = x
        self.y = y

    def is_pawn_equal(self, x, y):

        if self.x != x or self.y != y:

            return False

        return True

    def __repr__(self):
        return f"(x: {self.x}, y:{self.y})"