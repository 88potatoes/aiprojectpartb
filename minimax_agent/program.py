# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import math
from .helpers import get_next_state, get_possible_moves, render_board
from referee.game import PlayerColor, Action, PlaceAction, Coord, BOARD_N
from referee.game.coord import Direction
import random

TOTAL_RUNS = 10
MAX_TURNS = 150
C = 2
EXPANSION_FACTOR = 6
MINIMAX_DEPTH = 1

def get_next_player(player: PlayerColor):
    if player == PlayerColor.RED:
        current_player = PlayerColor.BLUE
    else:
        current_player = PlayerColor.RED
    return current_player

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
        self.current_state = Node(PlayerColor.RED, {})
        self.turn_count = 1
        self.first_turn = True
        self.player = color

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updating the state after each turn
        """

        next_player = get_next_player(color)
        self.current_state = Node(next_player, get_next_state(self.current_state.state, action, color))
        self.turn_count += 1

    def action(self, **referee: dict) -> Action:
        """
        (STRATEGY) Do straight monte carlo tree search
        Do a monte carlo tree search given a node

        RED is the maximising player
        BLUE is the minimising player
        """

        # initial move
        if self.first_turn:
            self.first_turn = False
            match self.player:
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

        # Depth limited minimax
        possible_moves = get_possible_moves(self.current_state.state, self.player)
        best_move = None

        if self.player == PlayerColor.RED:
            # maximising player
            best_score = -float('inf')
            for move in possible_moves:
                next_state = get_next_state(self.current_state.state, move, self.player)
                next_state_score = minimax(Node(PlayerColor.BLUE, next_state), MINIMAX_DEPTH, False)
                if next_state_score > best_score:
                    best_score = next_state_score
                    best_move = move
            
            return best_move
        else:
            # minimising player
            best_score = float('inf')

            for move in possible_moves:
                print("h")
                next_state = get_next_state(self.current_state.state, move, self.player)
                next_state_score = minimax(Node(PlayerColor.RED, next_state), MINIMAX_DEPTH, True)
                if next_state_score < best_score:
                    best_score = next_state_score
                    best_move = move
            
            return best_move
                

        
class Node:
    def __init__(self, player: PlayerColor, state: dict[Coord, PlayerColor]):
        self.player = player
        self.state = state

def score1(node: Node):
    """
    Score is the number of moves that the opponent will have
    """

    if node.player == PlayerColor.RED:
        # try to minimise opponent's moves
        moves = get_possible_moves(node.state, node.player)

        # player has lost - no moves
        if len(moves) == 0:
            return -float('inf')
        
        max_opp_moves = 0
        for move in moves:
            next_state = get_next_state(node.state, move, node.player)
            nmoves = get_possible_moves(next_state, PlayerColor.BLUE)
            max_opp_moves = max(max_opp_moves, nmoves)
        
        if max_opp_moves == 0:
            return float('inf')
        return 1 / max_opp_moves
    else:
        moves = get_possible_moves(node.state, node.player)

        if len(moves) == 0:
            return float('inf')
        
        max_opp_moves = 0
        for move in moves:
            next_state = get_next_state(node.state, move, node.player)
            nmoves = len(get_possible_moves(next_state, PlayerColor.RED))
            max_opp_moves = max(max_opp_moves, nmoves)

        if max_opp_moves == 0:
            return -float('inf')

        return 1 / max_opp_moves

score_function = score1
def minimax(node: Node, depth: int, isMaximisingPlayer: bool):

    possible_moves = get_possible_moves(node.state, node.player)
    print(node.player, len(possible_moves))
    # ending state
    if depth == 0 or len(possible_moves) == 0:
        return score_function(node)

    if isMaximisingPlayer:
        assert(node.player == PlayerColor.RED)

        value = -float('inf')
        for move in possible_moves:
            next_state = get_next_state(node.state, move, node.player)
            value = max(value, minimax(Node(PlayerColor.BLUE, next_state), depth - 1, False))
        return value
    else:
        # minimising player
        assert(node.player == PlayerColor.BLUE)

        value = float('inf')
        for move in possible_moves:
            next_state = get_next_state(node.state, move, node.player)
            value = min(value, minimax(Node(PlayerColor.RED, next_state), depth - 1, True))
        
        return value
        