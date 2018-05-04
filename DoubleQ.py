import blackjack
from pylab import *
import random

Q1 = 0.00001 * rand(182, 2)
Q2 = 0.00001 * rand(182, 2)

Q1[181, :] = 0
Q2[181, :] = 0


def learn(alpha, eps, numTrainingEpisodes):
    returnSum = 0.0
    gamma = 1
    for episodeNum in range(numTrainingEpisodes):
        G = 0

        # get the initial state
        state = blackjack.init()

        while state != 181:
            # choosing between greedy action and random with epsilon-greedy policy
            rndProb = random.uniform(0, 1)

            if rndProb < eps:
                action = random.randint(0, 1)

            else:
                # Greedy policy in sum of two action values
                action = getPolicy(state)

            reward, statePrime = blackjack.sample(state, action)

            if statePrime == False:
                statePrime = 181

            # Fill in Q1 and Q2

            # choosing between Q1 and Q2 to update
            updProb = random.randint(0, 1)

            if updProb == 0:
                Q1[state, action] = Q1[state, action] + alpha * (reward + (gamma * Q2[statePrime, argmax(Q1[statePrime, :])]) - Q1[state, action])
            elif updProb == 1:
                Q2[state, action] = Q2[state, action] + alpha * (reward + (gamma * Q1[statePrime, argmax(Q2[statePrime, :])]) - Q2[state, action])


            state = statePrime
            G = G + reward


        # print("Episode: ", episodeNum, "Return: ", G)
        returnSum = returnSum + G
        if episodeNum % 10000 == 0 and episodeNum != 0:
            print("Average return so far: ", returnSum / episodeNum)


def evaluate(numEvaluationEpisodes):
    returnSum = 0.0
    for episodeNum in range(numEvaluationEpisodes):
        G = 0
        # get the initial state
        state = blackjack.init()

        while state != 181:

            # get greedy action from policy
            action = getPolicy(state)

            # get next state and reward
            reward, statePrime = blackjack.sample(state, action)

            if statePrime == False:
                statePrime = 181

            state = statePrime

            # add the reward to total reward for this episode
            G = G + reward

        returnSum = returnSum + G
    return returnSum / numEvaluationEpisodes


def getPolicy(state):
    Q = Q1[state,:]+Q2[state,:]
    return Q.argmax()


def getAvgReturn():
    alpha = 0.001
    epsilon = 0.66
    numTrainingEpisodes = 10000000
    numEvaluationEpisodes = 10000000

    print("running the test...")
    print("\u03B1 =", alpha, " \u03B5 =", epsilon, " num eps =", numTrainingEpisodes)
    print("learning...")
    learn(alpha, epsilon, numTrainingEpisodes)
    print(" ")
    print("evaluation...")
    avgReturn = evaluate(numEvaluationEpisodes)
    print(" ")
    print("return: ", avgReturn)


if __name__ == '__main__':
    getAvgReturn()