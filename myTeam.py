# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from game import Actions
from util import nearestPoint

OBSERVATION_RANGE = 5

#################### COMMENTS ABOUT TEAM DESIGN #########################################
#This team is designed to play a conservative game, i.e. just get more points than 
#the opposing team. It is generally risk-averse i.e. 'win by a few' or 'prevent capture'
# 
# It does this offensively by implementing a strategy to gather food and 
# and quickly return to its side. It will return either because it is holding more than 8 or
# because it is being chased and making it to the home side is a viable option.
# It will not seek out food when being chased and will try to avoid corners (usually...)
# It will only be aggressive when it has the power pellet and enemies are scared. 
# Otherwise it generally uses minimax with a-b pruning to play chicken with the opponent.
#
# The defensive strategy is somewhat of a "dragons nest" approach. As soon as an invader
# gets stopped and looses its food, the defensive Pacman will bogart right on top of the pile
# effectively preventing the other team from reclaiming any of their food pellets for the 
# remainder of the game. The only way to beat this is if the opponent activates the power pellet. 
# Defensive ghost will use minimax with a-b pruning to avoid them in scared state.  
###########################################################################################




#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent'):

    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent): #methods are utilized here to save time
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        #print(game_state)
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    This agent is designed to be risk averse and bring back small quantities of food 
    while the defensive agent holds off the enemy
    """

    def choose_action(self, game_state):

        actions = game_state.get_legal_actions(self.index)

        # If there are no actions to take, return None
        if len(actions) == 0:
            return None

        # If the agent is not a Pacman (i.e., it's a Ghost), use the normal method
        if not game_state.get_agent_state(self.index).is_pacman:
            return super().choose_action(game_state)


        ###Primary Risk-Averse Strategy#### 
        # If the agent is carrying more than 8 food, call pcmn_full
        if game_state.get_agent_state(self.index).num_carrying > 8:
            return self.pcmn_full(game_state)

        my_pos = game_state.get_agent_state(self.index).get_position() #Get Current position
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)] #List and identify enemy
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        scared_ghosts = [ghost for ghost in ghosts if ghost.scared_timer > 0] #Identify scared ghosts
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]#Current invaders and their positions
        min_ghost_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]) if ghosts else float("inf")#Agents position to ghost

        
        ### Only Aggressive Move for Offensive Pacman###
        # If Pacman has eaten a power pellet and the opposing team is in a scared state, go into offensive attack mode
        if scared_ghosts:
            return self.offensive_attack(game_state, scared_ghosts)

        # If theres a ghost within 3 spaces and Pacman can get to a power pellet before being eaten by a ghost, go for the pellet
        power_pellets = self.get_capsules(game_state)
        if power_pellets and any(self.get_maze_distance(my_pos, ghost.get_position()) <= 3 for ghost in ghosts):
            min_pellet_distance = min([self.get_maze_distance(my_pos, pellet) for pellet in power_pellets])
            if min_pellet_distance < min_ghost_distance:
                return self.offensive_attack(game_state, ghosts)

        ### Safety Manuever that doubles as Risk-Averse Collection###
        # If Pacman is being chased by a ghost and he has the opportunity to go back to his home side to escape the ghost, he should take that option
        if ghosts and min_ghost_distance <= 3:
            best_distance = float("inf") #initialize tracker
            best_action = None #initalize tracker
            #calculates the x-coordinate of the home side. If the agent is on the red team, the home side is on left otherwise right
            home_x = min(my_pos[0], game_state.get_walls().width // 2 - 1) if self.red else max(my_pos[0], game_state.get_walls().width // 2)
            #generates a list of all positions on the home side (non-wall)
            home_positions = [(home_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(home_x, y)]

            #Calculates the maze distance from the agents new position to each position on the home side and takes the minimum of these distances, aka the shortest path
            for action in actions:
                successor = game_state.generate_successor(self.index, action)
                new_pos = successor.get_agent_state(self.index).get_position()
                distance = min(self.get_maze_distance(new_pos, pos) for pos in home_positions)
                if distance < best_distance:
                    best_distance = distance
                    best_action = action

            return best_action


        ###Play Defensive Chicken - Ideally Bring food home in process###
        # If there's a ghost within 3 steps, use MiniMax with Alpha-Beta pruning
        #assumes other teams have ghosts that play "optimally" which may not be true

        if len(ghosts) > 0:
            #initalize minimax variables
            best_value = float("-inf")
            alpha = float("-inf")
            beta = float("inf")
            best_action = None

            #Pick maximizing action (alpha)
            for action in actions:
                value = self.min_value(game_state.generate_successor(self.index, action), 1, alpha, beta, 2)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, value)

            return best_action
        else:
            # If there's no ghost within 3 steps, seek out food pellets & pellets
            food_list = self.get_food(game_state).as_list() + self.get_capsules(game_state)
            if len(food_list) > 0: 
                best_distance = float("inf")
                best_action = None
                for action in actions:
                    successor = game_state.generate_successor(self.index, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    distance = min(self.get_maze_distance(new_pos, food) for food in food_list)
                    if distance < best_distance:
                        best_distance = distance
                        best_action = action
                return best_action

        # If there's no food, return None
        return None


    def min_value(self, game_state, depth, alpha, beta, ghost_index):
        if depth == 0 or game_state.is_over():
            return self.evaluate(game_state, Directions.STOP)

        v = float("inf") #initialize value variable 

    # Check if the ghost is observable
        if game_state.get_agent_state(ghost_index).get_position() is not None:
            for action in game_state.get_legal_actions(ghost_index):
                if ghost_index == game_state.get_num_agents() - 1: #whos turn & generate successor states
                    v = min(v, self.max_value(game_state.generate_successor(ghost_index, action), depth - 1, alpha, beta))
                else:
                    v = min(v, self.min_value(game_state.generate_successor(ghost_index, action), depth, alpha, beta, ghost_index + 1))
                if v < alpha: #Prune this branch
                    return v
                beta = min(beta, v)

        return v


    def max_value(self, game_state, depth, alpha, beta):
        if depth == 0 or game_state.is_over():
            return self.evaluate(game_state, Directions.STOP)

        v = float("-inf") #initialize value variable 
        for action in game_state.get_legal_actions(self.index):
            v = max(v, self.min_value(game_state.generate_successor(self.index, action), depth, alpha, beta, self.index + 1))
            if v > beta: #Prune this branch
                return v
            alpha = max(alpha, v)

        return v


    #Strategy to return home if pacman has more than 8 food pellets (see above)
    def pcmn_full(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        best_distance = float("inf")
        best_action = None

        #Re-use "safe" return home logic from before when being chased
        my_pos = game_state.get_agent_state(self.index).get_position()
        home_x = min(my_pos[0], game_state.get_walls().width // 2 - 1) if self.red else max(my_pos[0], game_state.get_walls().width // 2)
        home_positions = [(home_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(home_x, y)]

        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            distance = min(self.get_maze_distance(new_pos, pos) for pos in home_positions)
            if distance < best_distance:
                best_distance = distance
                best_action = action

        return best_action

    #Added behavior for when Pacman has eaten the power pellet to be aggressive (see choose_action)
    def offensive_attack(self, game_state, ghosts):
        actions = game_state.get_legal_actions(self.index)
        best_distance = float("inf")
        best_action = None

        my_pos = game_state.get_agent_state(self.index).get_position()

        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            distance = min([self.get_maze_distance(new_pos, ghost.get_position()) for ghost in ghosts])
            if distance < best_distance:
                best_distance = distance
                best_action = action

        return best_action
   
    def get_features(self, game_state, action):

        #get game features for successor score, distance to food, distance to ghost
        #chased by ghost, can reach pellet, distance to home, dead-ends, and if Pacman is full (defined in actions)


        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        #same as in choose actions
        my_pos = successor.get_agent_state(self.index).get_position()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        ghost_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
        min_ghost_distance = min(ghost_distances) if ghost_distances else float("inf")

        # Compute distance to the nearest food
        if len(food_list) > 0: 
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Compute distance to the nearest ghost
        if min_ghost_distance <= 5:  # If a ghost is within observation radius
            features['distance_to_ghost'] = min_ghost_distance
        else:  # If no ghost is observable, set it to a high value
            features['distance_to_ghost'] = 20

        # Check if being chased by a ghost
        if game_state.get_agent_state(self.index).num_carrying > 0 and min_ghost_distance <= 5:
            features['chased_by_ghost'] = 1
        else:
            features['chased_by_ghost'] = 0


        # Check if can reach power pellet before ghost - used in offensive_attack above
        power_pellets = self.get_capsules(successor)
        if power_pellets:
            min_pellet_distance = min([self.get_maze_distance(my_pos, pellet) for pellet in power_pellets])
            if min_pellet_distance < min_ghost_distance:
                features['can_reach_pellet'] = 1
            else:
                features['can_reach_pellet'] = 0

        # Compute distance to home - similar logic as in choose_acion 
        if game_state.get_agent_state(self.index).num_carrying > 0:
            home_x = min(my_pos[0], game_state.get_walls().width // 2 - 1) if self.red else max(my_pos[0], game_state.get_walls().width // 2)
            home_positions = [(home_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(home_x, y)]
            home_pos = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
            features['distance_to_home'] = self.get_maze_distance(my_pos, home_pos)

        # Penalize moving into a corner
        my_next_pos = successor.get_agent_state(self.index).get_position()
        possible_moves = [m for m in Actions.get_legal_neighbors(my_next_pos, successor.get_walls()) if m != my_pos]
        if len(possible_moves) == 1:  # If there's only one way out
            features['corner'] = 1

        #To help head home once it has half of food needed to win
        if game_state.get_agent_state(self.index).num_carrying > 8:
            features['full_pacman'] = 1
        else:
            features['full_pacman'] = 0


        # Penalize actions leading to positions with fewer escape routes
        my_next_pos = successor.get_agent_state(self.index).get_position()
        if min_ghost_distance <= 3:  # If a ghost is within 3 steps
            possible_moves = [m for m in Actions.get_legal_neighbors(my_next_pos, successor.get_walls()) if m != my_pos]
            future_moves = [m for move in possible_moves for m in Actions.get_legal_neighbors(move, successor.get_walls())]
            if len(future_moves) <= 2:  # If there are two or less ways out within 3 moves (i.e., it's a potential dead-end)
                features['dead_end'] = 1
            else:
                features['dead_end'] = 0

        return features




    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'distance_to_ghost': 2, 'distance_to_home': -1, 'chased_by_ghost': 1000, 'can_reach_pellet': -1000, 'full_pacman': -1000,'dead_end': -2000}





#---------------------------------------------------------------

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
This agent is designed to be like a dragon guarding its stash of coins
Once the invader gets eaten and drops their food, this agent will bogart 
    """
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        my_state = game_state.get_agent_state(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # If the agent is in a scared state and there are invaders, use MiniMax with Alpha-Beta pruning
        
        #Same logic as in offensive_pacman, comments listed in that section
        #Main difference is that it occurs while defensive agent is scared

        if my_state.scared_timer > 0 and invaders:
            best_value = float("-inf")
            alpha = float("-inf")
            beta = float("inf")
            best_action = None

            for action in actions:
                value = self.min_value(game_state.generate_successor(self.index, action), 1, alpha, beta, self.index + 1)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, value)

            return best_action
        else:
            # If there are no invaders or the agent is not in a scared state, use the default behavior
            return max(actions, key=lambda a: self.evaluate(game_state, a))

    def min_value(self, game_state, depth, alpha, beta, invader_index):
        if depth == 0 or game_state.is_over():
            return self.evaluate(game_state, Directions.STOP)

        v = float("inf")

        # Check if the invader is observable
        if game_state.get_agent_state(invader_index).get_position() is not None:
            for action in game_state.get_legal_actions(invader_index):
                if invader_index == game_state.get_num_agents() - 1:
                    v = min(v, self.max_value(game_state.generate_successor(invader_index, action), depth - 1, alpha, beta))
                else:
                    v = min(v, self.min_value(game_state.generate_successor(invader_index, action), depth, alpha, beta, invader_index + 1))
                if v < alpha:
                    return v
                beta = min(beta, v)

        return v

    def max_value(self, game_state, depth, alpha, beta):
        if depth == 0 or game_state.is_over():
            return self.evaluate(game_state, Directions.STOP)

        v = float("-inf")
        for action in game_state.get_legal_actions(self.index):
            v = max(v, self.min_value(game_state.generate_successor(self.index, action), depth, alpha, beta, self.index + 1))
            if v > beta:
                return v
            alpha = max(alpha, v)

        return v

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            #find distance to each invader
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        #For avoiding 
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Encourage moving away from invaders when in a scared state and keeping invaders within observation radius
        if my_state.scared_timer > 0 and invaders:  # If the agent is in a scared state and there are invaders
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = -min(dists)  # Encourage moving away from invaders

            # Encourage keeping invaders within observation radius
            features['invader_in_radius'] = 1 if min(dists) <= OBSERVATION_RANGE else 0

        #Bogart where there are lots of coins. Make invaders come to us
        #uses the idea of a moore neighborhood
        if not invaders:  # If there are no invaders
            food = self.get_food_you_are_defending(successor)
            food_positions = food.as_list()
            neighborhood = [(my_pos[0] + dx, my_pos[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2)]
            food_concentration = sum(1 for pos in neighborhood if pos in food_positions)
            features['food_concentration'] = food_concentration

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2,'food_concentration': 100, 'invader_in_radius': -500}
