import blackjack
from pylab import *
import random

def run(numEvaluationEpisodes):
    returnSum = 0.0
    for episodeNum in range(numEvaluationEpisodes):
        G = 0
        # get the initial state
        state = blackjack.init()

        # random action selection
        action = random.randint(0,1)

        # get the next state and reward from MDP
        reward, state = blackjack.sample(state, action)

        # adding the reward to the total reward for this episode
        G = G + reward

        while state:
            # random action selection
            action = random.randint(0,1)

            # get the next state and reward from MDP
            reward, state = blackjack.sample(state, action)

            # adding the reward to the total reward for this episode
            G = G + reward

        # print("Episode: ", episodeNum, "Return: ", G)
        returnSum = returnSum + G
    return returnSum / numEvaluationEpisodes

def testRandomPolicy():
    numEvaluationEpisodes = 100000
    avgReturn = run(numEvaluationEpisodes)
    print("Average return for random policy: ", avgReturn)


if __name__ == '__main__':
    testRandomPolicy()