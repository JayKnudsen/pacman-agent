# baselineTeam.py
# ---------------
#v7?
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
# and quickly return to its side. It will return either because it is holding more than 6 or
# because it is being chased and making it to the home side is a viable option.
# It will not seek out food when being chased and will try to avoid corners (usually...)
# It will only be aggressive when it has the power pellet and enemies are scared. 
# Otherwise it generally uses minimax with a-b pruning to play chicken with the opponent.
# Now has a function to help it get out of a stuck cycle by forcing moving in home-side coordinate directions.
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

#----------------------------------------------------------------


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    This agent is designed to be risk averse and bring back small quantities of food 
    while the defensive agent holds off the enemy. Not great at evading so since the preliminary contest, 
    he only eats 6 instead of 8 to be full and has cycle-breaking function
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing) #call parent
        self.chase_timer = 0 #keep track of the number of steps the agent has been chasing an invader on the home side
        self.move_history = [] #Store prev moves
        self.repeat_count = 0 # Num of moves in a sequence
        self.depth_limit = 5  # Set the depth limit to a suitable value

    def choose_action(self, game_state):

        actions = game_state.get_legal_actions(self.index)

        # If there are no actions to take, return None
        if len(actions) == 0:
            return None

        # Error debug- if the agent is not a Pacman (ghost), use the normal method
        if not game_state.get_agent_state(self.index).is_pacman:
            return super().choose_action(game_state)

        ###Primary Risk-Averse Strategy#### 
        # If the agent is carrying more than 5 food, call pcmn_full
        if game_state.get_agent_state(self.index).num_carrying > 5:
            return self.pcmn_full(game_state)

        my_pos = game_state.get_agent_state(self.index).get_position() #Get Current position
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)] #List and identify enemy
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        scared_ghosts = [ghost for ghost in ghosts if ghost.scared_timer > 0] #Identify scared ghosts
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]#Current invaders and their positions
        min_ghost_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]) if ghosts else float("inf")#Agents position to ghost

        # If the agent is on the home side and there is an invader within observation range, chase the invader for up to 20 steps
        if not game_state.get_agent_state(self.index).is_pacman and invaders:
            nearest_invader = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position()))
            if self.get_maze_distance(my_pos, nearest_invader.get_position()) <= OBSERVATION_RANGE:
                if self.chase_timer < 20:
                    self.chase_timer += 1
                    return self.chase_invader(game_state, nearest_invader)
                else:
                    self.chase_timer = 0

        ### Only Aggressive Move for Offensive Pacman###
        # If Pacman has eaten a power pellet and the opposing team is in a scared state, go into offensive attack mode
        if scared_ghosts:
            return self.offensive_attack(game_state, scared_ghosts)

        # If there's a ghost within 3 spaces and Pacman can get to a power pellet before being eaten by a ghost, go for the pellet
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

          # Check if Pacman is stuck in a loop on his home side
        if not game_state.get_agent_state(self.index).is_pacman:
            self.move_history.append(game_state.get_agent_state(self.index).get_position())
            if len(self.move_history) > 8:
                self.move_history.pop(0)
                if self.move_history.count(self.move_history[-1]) >= 5:
                    self.repeat_count += 1
                    if self.repeat_count >= 5:
                        game_state = self.break_cycle(game_state)
            else:
                self.repeat_count = 0

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


    #function to get Pacman to move around a little bit if he gets stuck doing the same moves over and over agin 
    def break_cycle(self, game_state):
        # Get Pacmans current position
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Get direction of Pacmans home side (left or right)
        home_direction = Directions.WEST if self.red else Directions.EAST

        # Arbitrary sequence of moves to break the cycle
        moves = [Directions.NORTH] * 3 + [Directions.SOUTH] * 3 + [home_direction] * 3

        # Define the maximum number of iterations
        max_iterations = 10

        # Execute the sequence of moves
        for i in range(max_iterations):
            if i >= len(moves):
                break

            move = moves[i]

            # Check if the move is legal
            if move in game_state.get_legal_actions(self.index):
                # Update game state with the move
                game_state = game_state.generate_successor(self.index, move)
                my_pos = game_state.get_agent_state(self.index).get_position()
            else:
                # If the move is not legal, try to find an alternative move
                legal_moves = game_state.get_legal_actions(self.index)
                if Directions.STOP in legal_moves:
                    legal_moves.remove(Directions.STOP)
                if len(legal_moves) > 0:
                    # Choose a random legal move
                    alternative_move = random.choice(legal_moves)
                    game_state = game_state.generate_successor(self.index, alternative_move)
                    my_pos = game_state.get_agent_state(self.index).get_position()

        # Reset the repeat count and move history
        self.repeat_count = 0
        self.move_history = []

        # Return the final game state after executing the sequence of moves
        return game_state


        #Function so pacman goes after invaders on his home side for a bit, pretty self explainatory
    def chase_invader(self, game_state, invader):
        my_pos = game_state.get_agent_state(self.index).get_position()
        actions = game_state.get_legal_actions(self.index)
        best_action = None
        min_distance = float("inf")

        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            next_pos = successor.get_agent_state(self.index).get_position()
            distance = self.get_maze_distance(next_pos, invader.get_position())
            if distance < min_distance:
                min_distance = distance
                best_action = action

        return best_action


    #Had issues during the contest, tried to fix indexing.
    def min_value(self, game_state, depth, alpha, beta, ghost_index):
        if depth == 0 or game_state.is_over() or depth > self.depth_limit:
            return self.evaluate(game_state, Directions.STOP)

        v = float("inf") #initalize value var

        # Check if the ghost index is within the valid range
        if 0 <= ghost_index < game_state.get_num_agents():
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
        if depth == 0 or game_state.is_over() or depth > self.depth_limit:
            return self.evaluate(game_state, Directions.STOP)

        v = float("-inf") #initialize value variable 
        for action in game_state.get_legal_actions(self.index):
            v = max(v, self.min_value(game_state.generate_successor(self.index, action), depth, alpha, beta, self.index + 1))
            if v > beta: #Prune this branch
                return v
            alpha = max(alpha, v)

        return v


    #Strategy to return home if pacman has more than 5 food pellets (see above) - REVISED
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
    #should maybe reduce to 30 moves here if time
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
        if game_state.get_agent_state(self.index).num_carrying > 5:
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
    This agent is designed to move around the board between coins until there is a pile of at leat 4 within a moore
    neighborhood and then stay there and bogart it like a dragon on a hoard of coins. Updated since preliminary contest
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.target_position = None #future food he will move to
        self.depth_limit = 5  #To make sure no moves take more than 1 second. Issues from before

    def choose_action(self, game_state):

        #Same idea as offenisve 
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # If agent is in a scared state and there are invaders, use MiniMax with A-B pruning to avoid them
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
            # If there are invaders and agent not scared, attack the closest invader
            if invaders:
                closest_invader = min(invaders, key=lambda a: self.get_maze_distance(my_pos, a.get_position())) #takes an invader a and returns distance between current position and invader's position
                best_action = min(actions, key=lambda a: self.get_maze_distance(game_state.generate_successor(self.index, a).get_agent_state(self.index).get_position(), closest_invader.get_position()))
                return best_action
            else:
                # check if there is a pile of at least 4 food pellets in a Moore neighborhood
                food = self.get_food_you_are_defending(game_state)
                food_positions = food.as_list()

                neighborhood = [(my_pos[0] + dx, my_pos[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2)]
                food_concentration = sum(1 for pos in neighborhood if pos in food_positions)

                if food_concentration >= 4:
                    # stay in the Moore neighborhood with high concentration of food pellets
                    best_action = max(actions, key=lambda a: self.evaluate(game_state, a))
                    return best_action
                else:
                    # Move between food pellets that are at least 4 spaces away - maybe modify this later? 
                    if self.target_position is None or self.get_maze_distance(my_pos, self.target_position) < 4:
                        if food_positions:
                            self.target_position = random.choice(food_positions)
                        else:
                            self.target_position = None

                    if self.target_position:
                        best_action = min(actions, key=lambda a: self.get_maze_distance(game_state.generate_successor(self.index, a).get_agent_state(self.index).get_position(), self.target_position))
                        return best_action
                    else:
                        return random.choice(actions)
    #corrected for error
    def min_value(self, game_state, depth, alpha, beta, ghost_index):
        if depth == 0 or game_state.is_over() or depth > self.depth_limit:
            return self.evaluate(game_state, Directions.STOP)

        v = float("inf") #initalize value var

        # Check if the ghost index is within the valid range
        if 0 <= ghost_index < game_state.get_num_agents():
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
        if depth == 0 or game_state.is_over() or depth > self.depth_limit:
            return self.evaluate(game_state, Directions.STOP)

        v = float("-inf") #initialize value variable 
        for action in game_state.get_legal_actions(self.index):
            v = max(v, self.min_value(game_state.generate_successor(self.index, action), depth, alpha, beta, self.index + 1))
            if v > beta: #Prune this branch
                return v
            alpha = max(alpha, v)

        return v

    def get_features(self, game_state, action):
        #number of invaeders, defense/offensive toggle, distance to invaders, stop & reverse 
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense 
        #originally used to turn on minimizer or maximizer for a/b pruning but had issues. 
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        #Update stop and reverse in dict
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}