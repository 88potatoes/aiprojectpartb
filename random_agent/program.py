# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Action, PlaceAction, Coord, BOARD_N
from referee.game.coord import Direction
import random


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Tetress game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self.current_state = {}
        self.first_turn = True

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        if self.first_turn:
            self.first_turn = False
            match self._color:
                case PlayerColor.RED:
                    # print("Testing: RED is playing a PLACE action")
                    return PlaceAction(
                        Coord(3, 3), 
                        Coord(3, 4), 
                        Coord(4, 3), 
                        Coord(4, 4)
                    )
                case PlayerColor.BLUE:
                    # print("Testing: BLUE is playing a PLACE action")
                    return PlaceAction(
                        Coord(2, 3), 
                        Coord(2, 4), 
                        Coord(2, 5), 
                        Coord(2, 6)
                    )

        possible_moves = get_possible_moves(self.current_state, self._color)
        size = len(possible_moves)
        assert size > 0
        return possible_moves[random.randrange(0, size)]

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        self.current_state = get_next_state(self.current_state, action, color)


def get_possible_moves_from_coord(board: dict, coord: Coord):
    """
    Given a state and a coordinate, returns all possible moves from that coordinate.
    Returns a set of frozensets of Coords
    """

    path = [coord]
    possible_moves = set() # set of frozensets

    # generates all the possible moves
    # accounts for traversing across borders
    # returns an array of array of coords
    def dfs(coord, piece_number):
        if coord in path or coord in board:
            return
        
        # add coord to path
        path.append(coord)

        # sprouting from piece 1
        if piece_number == 1:

            # find all open squares from the current square
            open_coords = []
            for direction in Direction:
                new_coord = coord + direction
                
                # don't add square if it's already occupied
                if new_coord in path or new_coord in board:
                    continue
                
                open_coords.append(new_coord)

            # add a branch and continue
            for i in range(len(open_coords)):
                path.append(open_coords[i])
                for j in range(len(open_coords)):
                    if i == j:
                        continue
                    assert len(path) == 3
                    dfs(open_coords[j], piece_number + 2)
                path.pop()
                

            # if 3 squares are open then add the T-piece
            if len(open_coords) == 3:
                assert len(path) == 2
                possible_moves.add(frozenset([path[-1]] + open_coords))



        # sprouting from piece 2 to find T-piece
        if piece_number == 2:

            # find all open squares from the current square
            open_coords = []
            for direction in Direction:
                new_coord = coord + direction
                
                # don't add square if it's already occupied
                if new_coord in path or new_coord in board:
                    continue
                
                open_coords.append(new_coord)

            # if there are more than two, then add combinations to find T-pieces
            if len(open_coords) >= 2:
                for i in range(len(open_coords)):
                    for j in range(i+1, len(open_coords)):
                        assert len(path) == 3
                        possible_moves.add(frozenset(path[1:] + [open_coords[i], open_coords[j]]))


        # placed the final block
        if piece_number == 4:
            assert len(path) == 5
            possible_moves.add(frozenset(path[1:])) # take the last 4 to not include the original square
            path.pop()
            return
        
        # place the next block
        for direction in Direction:
            dfs(coord + direction, piece_number + 1)

        path.pop()

    for direction in Direction:
        dfs(coord + direction, 1)
    
    return possible_moves

def get_possible_moves(state: dict, player: PlayerColor):
    """
    Get all possible moves for a particular player.
    Return a list of PlaceAction
    """
    moveset = set()

    for coord, colour in state.items():
        if colour == player:
            moveset = moveset.union(get_possible_moves_from_coord(state, coord))

    return [set_to_place_action(piece) for piece in moveset]

def get_next_state(current_state: dict, piece: PlaceAction, color: PlayerColor):
    """
    Gets in a state and a piece (PlaceAction), and returns the next state.
    Accounts for lines being cleared.
    Assumes the pieces(PlaceAction) being placed is valid.
    """
    
    # cleared rows / cols
    cleared_rows = set()
    cleared_cols = set()

    current_state_copy = current_state.copy()
    # print("COPY", current_state_copy)

    # places new red squares
    for square in piece.coords:
        # print(square)
        current_state_copy[square] = color

    # getting a grid of filled squares
    grid = [[True if Coord(row, col) in current_state_copy else False for col in range(BOARD_N)] for row in range(BOARD_N)]
    # pprint(grid)
    
    # gets all full rows
    for row in range(BOARD_N):
        is_full_row = True
        for col in range(BOARD_N):
            if not grid[row][col]:
                is_full_row = False
                break
        
        if is_full_row:
            cleared_rows.add(row)

    # gets all full columns
    for col in range(BOARD_N):
        is_full_col = True
        for row in range(BOARD_N):
            if not grid[row][col]:
                is_full_col = False
                break
        
        if is_full_col:
            cleared_cols.add(col)
        
    # add in all squares except for ones that are cleared
    next_state = {coord: current_state_copy[coord] for coord in current_state_copy if coord.r not in cleared_rows and coord.c not in cleared_cols}
    return next_state

def set_to_place_action(coord_set):
    """
    Turns a set/frozenset into a PlaceAction
    """
    return PlaceAction(*coord_set)