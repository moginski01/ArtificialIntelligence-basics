# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(scores)
        # print(bestScore)
        # print(legalMoves)
        # print(chosenIndex)
        print(legalMoves)
        print(chosenIndex)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        #OPIS
        #newScaredTimes jak pacman zje dużą kulke to wtedy może zjadać duchy przez chwile i odmiezana ta chwila jest wlasnie w newScaredTimes
       
        from game import manhattanDistance

        # successorGameState.getGhostPositions()
        # print(successorGameState.getScore())
        ghost_positions = successorGameState.getGhostPositions()
        foodDist_list = []
        ghostDist_list = []

        for food in newFood.asList():
            dist = manhattanDistance(newPos,food)
            foodDist_list.append(dist)
            # print(food)

        for ghost in ghost_positions:
            dist = manhattanDistance(newPos,ghost)
            ghostDist_list.append(dist)
        
        max_foodDist = 1
        max_ghostDist = 1
        if len(foodDist_list) > 0:
            max_foodDist= min(foodDist_list)
        if len(ghostDist_list) > 0:
            max_ghostDist = max(ghostDist_list)
        # max_ghostDist = max(ghostDist_list)

        #pętla żeby pacman nie wchodził na żadnego ducha
        for ghost in ghost_positions:
            if manhattanDistance(newPos,ghost) < 2:
                #bardzo duża liczba ujemna żeby było jasne żeby tak nie robił 
                return -float("inf")

        # print(successorGameState.getScore())
        # print(successorGameState.getScore() + 1.0/max_foodDist-1.0/max_ghostDist)
        # print(max_foodDist)
        # print(max_ghostDist)
        
        return successorGameState.getScore() + 1.0/max_foodDist-1.0/max_ghostDist
    


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # self.evaluationFunction(gameState)
        # if self.depth!=2:
        #     print(self.depth)
        #     print(gameState)

        # print(self.depth)
        def minimax(self,gameState, depth, agentIndex):
            # print(depth)
            # print(self.depth)
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                # print(self.evaluationFunction(gameState))
                return [self.evaluationFunction(gameState),Directions.STOP]
            if agentIndex == (gameState.getNumAgents() -1):
                new_depth = depth+1
                new_agentIndex = self.index#wszystkie duchy wykonały operacje wiec teraz pora na pacmana
            else:
                new_depth = depth
                new_agentIndex = agentIndex + 1

            if agentIndex==0:
                maxEval = -float("inf")
                bestMove = Directions.STOP
                for action in gameState.getLegalActions(agentIndex):
                    # new_depth = depth+1
                    nextGameState = gameState.generateSuccessor(agentIndex, action)
                    eval = minimax(self,nextGameState,(new_depth),(new_agentIndex))[0]
                    if eval>maxEval:
                        maxEval=eval
                        bestMove = action
                        # maxEval = max(maxEval, eval)
                return maxEval,bestMove
            else:
                minEval = float("inf")
                bestMove = Directions.STOP
                for action in gameState.getLegalActions(agentIndex):
                    nextGameState = gameState.generateSuccessor(agentIndex, action)
                    # new_depth = depth+1
                    eval = minimax(self,nextGameState,(new_depth),(new_agentIndex))[0]
                    if eval<minEval:
                        # minEval = min(minEval, eval)
                        minEval = eval
                        bestMove = action
                return minEval, bestMove
        return minimax(self,gameState,0,self.index)[1]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #alphaa to najlepszy ruch dla gracza(max)
        #beta to najlepszy ruch dla ducha(min)
         
        def alpha_beta(gameState,agentIndex,depth,alpha,beta):
            result = []

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState),0
            
            # tutaj w zasadzie tak jak w poprzednim w minimax 
            # depth+1 bo wszystkie duchy wykonaly robote
            if agentIndex == gameState.getNumAgents() - 1:
                depth += 1
                nextAgent = self.index
            else:
                nextAgent = agentIndex + 1
            # Kiedy obecny agent to ostatni duch -> kolejny to pacman
            # if agentIndex == gameState.getNumAgents() - 1:
            #     nextAgent = self.index
            # else:
            #     nextAgent = agentIndex + 1

            actions = gameState.getLegalActions(agentIndex)

            for action in actions:
                #dodanie pierwszego ruchu ponieważ porównujemy potem wartość poprzednią z obecną potrzebna nam najpierw wartość jakaś początkowo
                if len(result)==0:
                    
                    successor = gameState.generateSuccessor(agentIndex,action)
                    eval = alpha_beta(successor,nextAgent,depth,alpha,beta)

                    result.append(eval[0])
                    result.append(action)

                    if agentIndex == self.index:
                        alpha = max(result[0],alpha)
                    else:
                        beta = min(result[0],beta)
                else:
                    # warunek alpha-beta                               
                    if beta < alpha:
                        return result


                    successor = gameState.generateSuccessor(agentIndex,action)

                    prevEval = result[0]
                    eval = alpha_beta(successor,nextAgent,depth,alpha,beta)

                    if agentIndex == self.index:
                        #zamist max() ponieważ potrzebujemy ruch jeszcze dopisać
                        if eval[0] > prevEval:
                            result[0] = eval[0]
                            result[1] = action

                            alpha = max(result[0],alpha)

                    else:
                        #zamiast min() ponieważ potrzebujemy ruch dopisać i trzeba wiedzieć kiedy można 
                        if eval[0] < prevEval:
                            result[0] = eval[0]
                            result[1] = action

                            beta = min(result[0],beta)
            return result


        return alpha_beta(gameState,self.index,0,-float("inf"),float("inf"))[1]




        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #magiczna linijka dla ghostów
        # result.append((1.0 / len(gameState.getLegalActions(agent))) * eval[0] + eval[0])
        def expectimax(gameState,agentIndex,depth):
            result = []

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState),0
            
            # tutaj w zasadzie tak jak w poprzednim w minimax 
            # depth+1 bo wszystkie duchy wykonaly robote
            if agentIndex == gameState.getNumAgents() - 1:
                depth += 1
                nextAgent = self.index
            else:
                nextAgent = agentIndex + 1
            # Kiedy obecny agent to ostatni duch -> kolejny to pacman
            # if agentIndex == gameState.getNumAgents() - 1:
            #     nextAgent = self.index
            # else:
            #     nextAgent = agentIndex + 1

            actions = gameState.getLegalActions(agentIndex)

            for action in actions:
                #dodanie pierwszego ruchu ponieważ porównujemy potem wartość poprzednią z obecną potrzebna nam najpierw wartość jakaś początkowo
                if len(result)==0:
                    
                    successor = gameState.generateSuccessor(agentIndex,action)
                    eval = expectimax(successor,nextAgent,depth)
                    if agentIndex == self.index:
                        result.append(eval[0])
                        result.append(action)
                    else:
                        result.append((1.0/len(actions)*eval[0]))
                        result.append(action)

                else:
                    # warunek alpha-beta                               
    

                    successor = gameState.generateSuccessor(agentIndex,action)

                    prevEval = result[0]
                    eval = expectimax(successor,nextAgent,depth)

                    if agentIndex == self.index:
                        #zamist max() ponieważ potrzebujemy ruch jeszcze dopisać
                        if eval[0] > prevEval:
                            result[0] = eval[0]
                            result[1] = action


                    else:
                        #zamiast min() ponieważ potrzebujemy ruch dopisać i trzeba wiedzieć kiedy można 
                        #TO DO POPRAWY NA RAZIE NIE ZADZIALA
                        if eval[0] < prevEval:
                            result[0] = eval[0]
                            result[1] = action

            return result


        return expectimax(gameState,self.index,0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction