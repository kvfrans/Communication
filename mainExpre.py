import gym
import gym_delivery
import numpy as np
import tensorflow as tf
import random
import time


class Q_Basic():
    def __init__(self,num_actions, num_observations, num_hidden):
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.num_hidden = num_hidden

        self.observations_in = tf.placeholder(tf.float32, [None,num_observations])

        self.w1 = tf.Variable(tf.random_normal([num_observations, num_hidden], stddev=0.1), name="w1")
        self.b1 = tf.Variable(tf.random_normal([num_hidden], stddev=0.), name="b1")
        self.w2 = tf.Variable(tf.random_normal([num_hidden, num_actions], stddev=0.1), name="w2")
        self.b2 = tf.Variable(tf.random_normal([num_actions], stddev=0.), name="b2")

        self.h1 = tf.nn.relu(tf.matmul(self.observations_in, self.w1))
        self.estimated_values = tf.matmul(self.h1, self.w2)

        self.tvars = tf.trainable_variables()

        # one-hot matrix of which action was taken
        self.action_in = tf.placeholder(tf.float32,[None,num_actions])
        # vector of size [timesteps]
        self.return_in = tf.placeholder(tf.float32,[None])
        guessed_action_value = tf.reduce_sum(self.estimated_values * self.action_in, reduction_indices=1)
        loss = tf.nn.l2_loss(guessed_action_value - self.return_in)
        self.debug = loss
        self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def getAction(self, observation):
        values = self.getValues(observation)
        return np.argmax(values[0], axis=0)

    def getValues(self, observation):
        observation_reshaped = np.reshape(observation,(1,self.num_observations))
        return self.sess.run(self.estimated_values, feed_dict={self.observations_in: observation_reshaped})

    def getBatchValues(self,observation):
        return self.sess.run(self.estimated_values, feed_dict={self.observations_in: observation})

    def update(self, observation, action, reward):
        estimated, _ = self.sess.run([self.debug, self.optimizer], feed_dict={self.observations_in: observation, self.action_in: action, self.return_in: reward})
        return estimated




env = gym.make('Delivery-v0')
# env.monitor.start('monitor/', force=True)



num_actions = 4
num_observations = 4

model = Q_Basic(num_actions, num_observations, 30)
transitions = []
for episode in xrange(1000):
    observation = env.reset()
    epsilon = max(0.2, ((100-episode) / 100.0))

    totalreward = 0
    for frame in xrange(100):
        # env.render()

        # epsilon-greedy actions
        action = model.getAction(observation)
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()

        old_observation = observation
        observation, reward, done, info = env.step(action)

        # print model.getValues(observation)

        totalreward += reward
        transitions.append((old_observation,action,reward,False,observation))

        if len(transitions) > 600:
            transitions.pop(0)

        if done:
            break
    print totalreward

    for x in xrange(50):
        observation_history = np.zeros((0,num_observations))
        action_history = np.zeros((0,num_actions))
        TQ_history = np.array(())
        for transition in random.sample(transitions, 64):
            old_observation, action, reward, done, next_observation = transition

            # print "%s w/action %d gives reward %f to %s which was worth %s" % (np.array_str(old_observation), action, reward, np.array_str(next_observation), np.array_str(model.getValues(next_observation)))

            TQ = reward + 0.5 * np.amax(model.getValues(next_observation))
            print TQ

            old_observation_reshaped = np.reshape(old_observation,(1,num_observations))
            observation_history = np.append(observation_history,old_observation_reshaped,axis=0)

            action_onehot = np.zeros((1,num_actions))
            action_onehot[:,action] = 1.0
            action_history = np.append(action_history,action_onehot,axis=0)

            TQ_history = np.append(TQ_history,TQ)

        model.update(observation_history,action_history,TQ_history)
        print "update done"
        # lr *= 0.9995


# env.monitor.close()
