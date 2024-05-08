# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import math
from .helpers import get_next_state, get_possible_moves, render_board
from referee.game import PlayerColor, Action, PlaceAction, Coord, BOARD_N
from referee.game.coord import Direction
import copy

NO_MOVES = 1

TOTAL_RUNS = 12
MAX_TURNS = 150
C = 2
EXPANSION_FACTOR = 6

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
        self.monte_carlo_root = MonteCarloNode({}, None, 0, PlayerColor.RED, None)
        self.turn_count = 1

    def action(self, **referee: dict) -> Action:
        """
        (STRATEGY) Do straight monte carlo tree search
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

        for i in range(TOTAL_RUNS):
            root = self.monte_carlo_root

            # selection
            leaf = select(root)
            # print(render_board(leaf.state, None, ansi=True))
            # print('is_terminal', leaf.is_terminal)

            # leaf is terminal state
            if leaf.is_terminal:
                result = rollout(leaf)
                backpropogate(result, leaf)
                continue


            # expansion
            child = expand(leaf)

            if not child:
                # terminal state
                result = rollout(leaf)
                backpropogate(result, leaf)
                continue

            # simulation
            result = rollout(child)

            # backpropogate
            backpropogate(result, child)


        # selecting highest UBC1 from immediate children
        if self._color == PlayerColor.BLUE:
            highest_score = -1
            highest_score_action = None
            for child in self.monte_carlo_root.children:
                assert(child.n > 0)
                score = child.t / child.n
                if score > highest_score:
                    highest_score = score
                    highest_score_action = child.action
            return highest_score_action
        else:
            lowest_score = float('inf')
            lowest_score_action = None

            for child in self.monte_carlo_root.children:
                assert(child.n > 0)
                score = child.t / child.n
                if score < lowest_score:
                    lowest_score = score
                    lowest_score_action = child.action
            return lowest_score_action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updating the state after each turn
        """

        next_player = get_next_player(color)
        self.monte_carlo_root = MonteCarloNode(get_next_state(self.monte_carlo_root.state, action, color), None, self.turn_count, next_player, action)

        self.turn_count += 1
            

class MonteCarloNode:
    def __init__(self, state: dict, parent, turn: int, player: PlayerColor, action: PlaceAction):
        self.state = state
        self.t = 0
        self.n = 0
        self.children = []
        self.parent = parent
        self.turn = turn
        self.player = player
        self.action = action
        self.is_terminal = False
        self.used_actions = set()
        self.n_possible_moves = len(get_possible_moves(self.state, self.player))

        # we consider a node to be expanded if it has 10 children
        # because i think it's easier to code
        self.expanded = False

    def increase_t(self, score):
        self.t += score
        self.n += 1
    
    def append_child(self, node):
        # the function shouldn't be called if the node is already expanded
        assert(len(self.children) < EXPANSION_FACTOR and self.expanded == False)
        self.children.append(node)
        self.used_actions.add(frozenset(node.action.coords))
        
        # determining if a node is expanded
        if len(self.children) == min(self.n_possible_moves, EXPANSION_FACTOR):
            self.expanded = True


def rollout(child: MonteCarloNode):
    """
    MONTE CARLO SIMULATION / ROLLOUT
    """    

    # keep rolling out until one player has no available plays left or the max turns has been reached
    r_node = copy.deepcopy(child)
    
    while r_node.turn <= MAX_TURNS:

        random_move = get_random_move(r_node)

        if not random_move:
            # no possible moves

            if r_node.player == PlayerColor.BLUE:
                # RED WINS
                return 0
            else:
                # BLUE WINS
                return 1

        next_turn = r_node.turn + 1
        next_player = get_next_player(r_node.player)
        next_state = get_next_state(r_node.state, random_move, r_node.player)
        r_node = MonteCarloNode(next_state, None, next_turn, next_player, random_move)

    # max turns has been reached
    # return based on whoever has more squares
    reds = 0
    blues = 0
    for _, color in r_node.state.items():
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

def get_random_move(node: MonteCarloNode) -> PlaceAction:
    possible_moves = get_possible_moves(node.state, node.player)
    if not possible_moves:
        return None
    
    # returns a random move because possible_moves is already in a random order (i'm pretty sure)
    for move in possible_moves:
        if frozenset(move.coords) in node.used_actions:
            continue
        return move


def get_next_player(player: PlayerColor):
    if player == PlayerColor.RED:
        current_player = PlayerColor.BLUE
    else:
        current_player = PlayerColor.RED
    return current_player

def get_ubc1(node: MonteCarloNode):
    assert node.parent != None # should never be calling UBC1 on the root node
    assert(node.parent.n > 0)
    return (node.t / node.n) + C * math.sqrt(math.log(node.parent.n) / node.n)


def select(node: MonteCarloNode) -> MonteCarloNode:
    """
    MONTE CARLO SELECTION
    """    
    current_node = node
    while current_node.expanded:
        # find another node

        # select the highest UBC1 node
        highest_ubc1_node = current_node
        highest_ubc1 = -1
        for child in current_node.children:
            assert child.n > 0
            assert child.parent.n > 0

            ubc1 = get_ubc1(child)
            if ubc1 > highest_ubc1:
                highest_ubc1 = ubc1
                highest_ubc1_node = child

            # TODO can optimise by short circuiting upon hitting inf
            # TODO optimise by having ubc get called once and stored within the node

        # make the highest UBC1 node the next node in the chain
        current_node = highest_ubc1_node
    
    return current_node

def expand(leaf: MonteCarloNode) -> MonteCarloNode:
    """
    MONTE CARLO EXPANSION
    """
    random_move = get_random_move(leaf)

    # terminal state
    if not random_move:
        return None

    next_state = get_next_state(leaf.state, random_move, leaf.player)
    next_player = get_next_player(leaf.player)
    child = MonteCarloNode(next_state, leaf, leaf.turn + 1, next_player, random_move)
    leaf.append_child(child)
    return child

def backpropogate(result, node: MonteCarloNode):
    """
    MONTE CARLO BACKPROPOGATION
    """    
    while node:
        node.increase_t(result)
        node = node.parent