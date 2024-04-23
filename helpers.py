from referee.game.actions import PlaceAction
from referee.game.constants import BOARD_N
from referee.game.coord import Coord, Direction
from referee.game.player import PlayerColor


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

def render_board(
    board: dict[Coord, PlayerColor], 
    target: Coord | None = None,
    ansi: bool = False
) -> str:
    """
    Visualise the Tetress board via a multiline ASCII string, including
    optional ANSI styling for terminals that support this.

    If a target coordinate is provided, the token at that location will be
    capitalised/highlighted.
    """
    output = ""
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if board.get(Coord(r, c), None):
                is_target = target is not None and Coord(r, c) == target
                color = board[Coord(r, c)]
                color = "r" if color == PlayerColor.RED else "b"
                text = f"{color}" if not is_target else f"{color.upper()}"
                if ansi:
                    output += apply_ansi(text, color=color, bold=is_target)
                else:
                    output += text
            else:
                output += "."
            output += " "
        output += "\n"
    return output

def apply_ansi(
    text: str, 
    bold: bool = True, 
    color: str | None = None
):
    """
    Wraps some text with ANSI control codes to apply terminal-based formatting.
    Note: Not all terminals will be compatible!
    """
    bold_code = "\033[1m" if bold else ""
    color_code = ""
    if color == "r":
        color_code = "\033[31m"
    if color == "b":
        color_code = "\033[34m"
    return f"{bold_code}{color_code}{text}\033[0m"