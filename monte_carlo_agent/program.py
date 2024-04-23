# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import math
from helpers import get_next_state, get_possible_moves
from referee.game import PlayerColor, Action, PlaceAction, Coord, BOARD_N
from referee.game.coord import Direction
import random

TOTAL_RUNS = 100
MAX_TURNS = 150
C = 2

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
        self.first_turn = True
        self.monte_carlo_root = None
        self.turn_count = 1

    def action(self, **referee: dict) -> Action:
        """
        Do a monte carlo tree search given a node
        """
        # RED = 0
        # BLUE = 1

        # initial move
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

        # all moves after the first one 
        # (MONTE CARLO)

        runs = 0
        while (runs < TOTAL_RUNS):

            node = self.monte_carlo_root

            # selection
            while node.children:

                # select the highest UBC1 node
                highest_ubc1_node = None
                highest_ubc1 = -1
                for child in node.children:
                    if child.ubc1 > highest_ubc1:
                        highest_ubc1 = child.ubc1
                        highest_ubc1_node = child
                    # TODO can optimise by short circuiting

                # make the highest UBC1 node the next node in the chain
                node = highest_ubc1_node
            
            # now should be at a leaf node
            if node.n == 0:
                # hasn't been rolled out yet
                res = rollout(node.state, node.turn, node.player)
                node.increase_t(res)
                node.calculate_ubc1(runs)
            else:
                # expand one state
                possible_moves = get_possible_moves(node.state, node.player)
                random_next_move = possible_moves[random.randrange(0, len(possible_moves))]
                next_state = get_next_state(node.state, random_next_move, node.player)
                next_player = None
                if node.player == PlayerColor.RED:
                    next_player = PlayerColor.BLUE
                else:
                    next_player = PlayerColor.RED
                new_node = MonteCarloNode(next_state, node, node.turn + 1, next_player)

                # rollout
                res = rollout(new_node.state, new_node.turn, new_node.player)
                new_node.increase_t(res)
                node.calculate_ubc1(runs)
            
            # backpropogate

            runs += 1


        possible_moves = get_possible_moves(self.current_state, self._color)
        size = len(possible_moves)
        assert size > 0
        return possible_moves[random.randrange(0, size)]

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updating the state
        """

        if color == PlayerColor.RED:
            next_player = PlayerColor.BLUE
        else:
            next_player = PlayerColor.RED
        
        self.monte_carlo_root = MonteCarloNode(get_next_state(self.monte_carlo_root.state, action, color), None, self.turn_count, next_player)

        self.turn_count += 1
            


class MonteCarloNode:
    def __init__(self, state: dict, parent, turn: int, player):
        self.state = state
        self.t = 0
        self.n = 0
        self.ubc1 = float('inf')
        self.children = []
        self.parent = parent
        self.turn = turn
        self.player = player

    def increase_t(self, score):
        self.t += score
        self.n += 1
    
    def calculate_ubc1(self, runs):
        assert self.n != 0
        self.ubc1 = (self.t / self.n) + C * math.sqrt(math.log(runs / self.n))
    
    def append_child(self, node):
        self.children.append(node)

def rollout(state: dict, current_turn: int, player: PlayerColor):
    # keep rolling out until one player has no available plays left or the max turns has been reached
    current_player = player
    current_state = state
    while current_turn <= MAX_TURNS:
        possible_moves = get_possible_moves(current_state, current_player)
        if not possible_moves:
            # this player has no possible moves, so the other player wins
            if player == PlayerColor.BLUE:
                # RED WINS
                return 0
            else:
                # BLUE WINS
                return 1
            
        # there are moves, so pick one at random and continue the rollout
        next_action = possible_moves[random.randrange(0, len(possible_moves))]
        current_state = get_next_state(current_state, next_action)

        current_turn += 1
        if current_player == PlayerColor.RED:
            current_player = PlayerColor.BLUE
        else:
            current_player = PlayerColor.RED

    # max turns has been reached
    # return based on whoever has more squares
    reds = 0
    blues = 0
    for _, color in current_state:
        if color == PlayerColor.RED:
            reds += 1
        else:
            blues += 1
    
    if reds > blues:
        # red won
        return 0
    elif reds == blues:
        # tie
        return 0.5
    else:
        # blue won
        return 1
            