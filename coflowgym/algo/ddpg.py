import tensorflow as tf
## 解决兼容性问题
if eval(tf.__version__.split(".")[0]) == 2:
    import tensorflow.compat.v1 as tf 
    tf.disable_eager_execution()

import numpy as np
import numpy.random as nr
import gym
import time

###############################  DDPG  ####################################

class DDPG(object):
    #####################  hyper parameters  ####################
    LR_A = 0.001    # learning rate for actor， default is 0.001
    LR_C = 0.002    # learning rate for critic, default is 0.002
    GAMMA = 0.9     # reward discount, default to 0.9
    TAU = 0.01      # soft replacement, default to 0.01
    MEMORY_CAPACITY = 10000 # default to 10000
    BATCH_SIZE = 32

    ##################### success config ##############
    # LR_A = 0.001
    # LR_C = 0.0001
    # GAMMA = 0.9 
    # TAU = 0.001 
    # MEMORY_CAPACITY = 10000 
    # BATCH_SIZE = 32

    def __init__(self, a_dim, s_dim, a_bound,GAMMA=0.9):
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.GAMMA = GAMMA
        self.update_every = 1

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + self.GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(td_error, var_list=self.ce_params)
        tf.summary.scalar("td_error", td_error)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=self.ae_params)
        tf.summary.scalar("a_loss", a_loss)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("tf_log/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        actions = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        # print("actions: ", actions)
        return actions[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        for _ in range(self.update_every):
            indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
            bt = self.memory[indices, :]
            bs = bt[:, :self.s_dim]
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br = bt[:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:]

            self.sess.run(self.atrain, {self.S: bs})
            self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
            summary = self.sess.run(self.merged, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
            self.writer.add_summary(summary, self.pointer)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 600
            n_l2 = 600
            # net1 = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            w1 = tf.get_variable("w1", [self.s_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable("b1", [1, n_l1], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1) + b1)
            ## BN
            # net1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(s, w1) + b1, training=True, name="BN_1"))
            w2 = tf.get_variable('w2', [n_l1, n_l2], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l2], trainable=trainable)
            net = tf.nn.relu(tf.matmul(net1, w2) + b2)
            ## BN
            # net = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(net1, w2) + b2, training=True, name="BN_2"))
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)

            tf.summary.histogram(scope+"/w1", w1)
            tf.summary.histogram(scope+"/b1", b1)
            tf.summary.histogram(scope+"/w2", w2)
            tf.summary.histogram(scope+"/b2", b2)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 600
            n_l2 = 600
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            data = tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1
            ## BN
            # data = tf.layers.batch_normalization(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1, training=True, name="BN_1")
            net1 = tf.nn.relu(data)
            w2 = tf.get_variable('w2', [n_l1, n_l2], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l2], trainable=trainable)
            net = tf.nn.relu(tf.matmul(net1, w2) + b2)
            ## BN
            # net = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(net1, w2) + b2, training=True, name="BN_2"))

            tf.summary.histogram(scope+"/w1_s", w1_s)
            tf.summary.histogram(scope+"/w1_a", w1_a)
            tf.summary.histogram(scope+"/b1", b1)
            tf.summary.histogram(scope+"/w2", w2)
            tf.summary.histogram(scope+"/b2", b2)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
    
    def save(self, filename="./model.ckpt"):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.save(self.sess, filename)
    
    def load(self, filename="./model.ckpt"):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("In load(): before")
            saver.restore(self.sess, filename)
            print("In load():after")
    
    def __str__(self):
        return "class = %s, LR_A = %s, LR_C = %s, GAMMA = %s, TAU = %s, MEMORY_CAPACITY = %s, BATCH_SIZE = %s, a_dim = %s, s_dim = %s, a_bound = %s"%(self.__class__.__name__, self.LR_A, self.LR_C, self.GAMMA, self.TAU, self.MEMORY_CAPACITY, self.BATCH_SIZE, self.a_dim, self.s_dim, self.a_bound)

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

###############################  training  ####################################
###############################  A Demo  ####################################
MAX_EPISODES = 2000
MAX_EP_STEPS = 200 # default is 200
RENDER = False
ENV_NAME = 'Pendulum-v0'
A_BOUND = 2

# RENDER = True
MAX_EP_STEPS = 1000
ENV_NAME = "MountainCarContinuous-v0"
A_BOUND = 1

EXPLORE = 70
pre_trained = False

def train():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = np.array([np.float64(A_BOUND)])
    print(a_bound, env.action_space.high, env.action_space.low)

    oun = OUNoise(a_dim, mu=0.4)

    ddpg = DDPG(a_dim, s_dim, a_bound)

    if pre_trained:
        ddpg.load('./log/model.ckpt')

    var = 3  # control exploration
    t1 = time.time()
    ave_rs = []
    his_step = []

    def test():
        TEST_EPISODE = 10
        ep_steps = 0
        test_reward = 0
        for _ in range(TEST_EPISODE):
            s = env.reset()
            for t in range(MAX_EP_STEPS):
                # env.render()
                a = ddpg.choose_action(s)
                s, r, done, _ = env.step(a)
                test_reward += r
                if t == MAX_EP_STEPS-1 or done:
                    # print("in test: consume %s steps!"%(t))
                    ep_steps += t
                    break
        return ep_steps//TEST_EPISODE, test_reward//TEST_EPISODE

    epsilon = 1
    for episode in range(1, 1+MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        oun.reset()
        epsilon -= (epsilon/EXPLORE)

        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            action_original = ddpg.choose_action(s)
            # a = 2*a-1 ## for sigmoid
            # a = np.clip(np.random.normal(action_original, var), -1*a_bound[0], a_bound[0])    # add randomness to action selection for exploration
            a = action_original+max(0.01, epsilon)*oun.noise()
            s_, r, done, _ = env.step(a)
            # print("step:", j, s, r, a)
            # if r < -5:
            #     r = r*10

            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > ddpg.BATCH_SIZE:
                if ddpg.pointer % 1 == 0:
                    var *= .9995    # decay the action randomness
                ddpg.learn()
                if ddpg.pointer == (ddpg.BATCH_SIZE+1):
                    print("Begin learning...")

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1 or done:
                # print('Episode:', episode, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                # if ep_reward > -300:RENDER = True
                his_step.append(j)
                print("episode", episode, "consume", j, "steps, epsilon =", epsilon,"var = ", var, "ep_reward:", ep_reward)
                break
        if episode % 20 == 0:
            t_steps, t_rewards = test()
            print("in test: average consume %s steps and ep_rewards is %s!"%(t_steps, t_rewards))
            if ENV_NAME == "MountainCarContinuous-v0" and t_steps < 180:
                ddpg.save("./log/model.ckpt")
                break
            if ENV_NAME == 'Pendulum-v0' and t_rewards > -150:
                ddpg.save("./log/model.ckpt")
                break
        # if (len(his_step) >= 5 and sum(his_step[-10:])/10 < 180):
        #     ddpg.save("./log/model.ckpt")
        #     return
        # sys.stdout.flush()
    print('Running time: ', time.time() - t1)


def validate():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = np.array([np.float64(A_BOUND)])
    print(a_bound, env.action_space.high, env.action_space.low)

    oun = OUNoise(a_dim, mu=0.4)

    ddpg = DDPG(a_dim, s_dim, a_bound)

    ddpg.load('./log/model.ckpt')

    var = 3  # control exploration
    t1 = time.time()
    
    for i in range(10):
        state = env.reset()
        ep_r = 0
        for j in range(MAX_EP_STEPS):
            env.render()
            action = ddpg.choose_action(state) # direct action for test
            state, reward, done, _ = env.step(action)
            ep_r += reward
            if j==MAX_EP_STEPS-1 or done:
                print("in validate: consume %s steps and ep_reward is %s!"%(j, ep_r))
                break
    print('Running time: ', time.time() - t1)

if __name__ == "__main__":
    # sys.stdout = open("log/%s-log.txt"%(ENV_NAME), "w")

    # TRAINABLE = True
    # # TRAINABLE = False
    # if TRAINABLE:
    #     train()
    # else:
    #     validate()
    pass
    version = tf.__version__
    print(version, type(version), )