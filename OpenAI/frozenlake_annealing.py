import gym
import numpy as np

env = gym.make('FrozenLake-v0')

actions = {
    0: 'U',
    1: 'R',
    2: 'D',
    3: 'L',
}

term = [5, 7, 11, 12, 15]

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .85
y = .99
num_episodes = 9000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0.
    #The Q-Table learning algorithm
    while True:
        j += 1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    jList.append(j)
    rList.append(rAll)
print "Score over time: " +  str(np.average(rList))
print "Average length time: " +  str(np.average(jList))
print "Final Q-Table Values"
print Q

for i in range(len(Q)):
    if i in term:
        print "X",
    else:
        idx = np.argmax(Q[i, :])
        print actions[idx], 
    if i % 4 == 3:
        print

for i in range(len(Q)):
    print np.max(Q[i, :]),
    if i % 4 == 3:
        print