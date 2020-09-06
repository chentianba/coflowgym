import tensorflow as tf
import numpy as np
import numpy.random as nr
import gym
import time, copy
from algo.ddpg import OUNoise, DDPG

###############################  DDPG with LSTM  ####################################

class DDPG_LSTM(object):
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

    def __init__(self, a_dim, s_dim, a_bound, time_sequence=10, lstm_dim=128,GAMMA=0.9):
        # (s, s', a, r)
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * time_sequence * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.time_sequence = time_sequence
        self.lstm_dim = lstm_dim
        self.GAMMA = GAMMA
        self.update_every = 1

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, self.time_sequence, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.time_sequence, s_dim], 's_')
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
            bs = bt[:, :self.s_dim*self.time_sequence]
            ba = bt[:, self.s_dim*self.time_sequence: self.s_dim*self.time_sequence + self.a_dim]
            br = bt[:, -self.s_dim*self.time_sequence - 1: -self.s_dim*self.time_sequence]
            bs_ = bt[:, -self.s_dim*self.time_sequence:]
            bs = bs.reshape(-1, self.time_sequence, self.s_dim)
            bs_ = bs_.reshape(-1, self.time_sequence, self.s_dim)

            self.sess.run(self.atrain, {self.S: bs})
            self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
            summary = self.sess.run(self.merged, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
            self.writer.add_summary(summary, self.pointer)

    def store_transition(self, s, a, r, s_):
        # print(s.reshape(1, -1).shape, a.shape, r, s_.reshape(1,-1).shape)
        transition = np.hstack((s.reshape(-1), a, [r], s_.reshape(-1)))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            ## add LSTM
            hidden_units1 = 100
            # lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_units1)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_dim)
            multi_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell])
            # init_state = multi_lstm.zero_state(batch_size=self.BATCH_SIZE, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(cell=multi_lstm, inputs=s, dtype=tf.float32)
            lstm_h = outputs[:, -1, :]

            n_l1 = 600
            n_l2 = 600
            # net1 = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            w1 = tf.get_variable("w1", [self.lstm_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable("b1", [1, n_l1], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(lstm_h, w1) + b1)
            ## BN
            # net1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(s, w1) + b1, training=True, name="BN_1"))
            w2 = tf.get_variable('w2', [n_l1, n_l2], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l2], trainable=trainable)
            net = tf.nn.relu(tf.matmul(net1, w2) + b2)
            ## BN
            # net = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(net1, w2) + b2, training=True, name="BN_2"))
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)

            # tf.summary.histogram(scope+"/w1", w1)
            # tf.summary.histogram(scope+"/b1", b1)
            # tf.summary.histogram(scope+"/w2", w2)
            # tf.summary.histogram(scope+"/b2", b2)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            ## add LSTM
            hidden_units1 = 128
            # lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_units1)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_dim)
            multi_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell])
            # init_state = multi_lstm.zero_state(batch_size=self.BATCH_SIZE, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(cell=multi_lstm, inputs=s, dtype=tf.float32)
            lstm_h = outputs[:, -1, :]

            n_l1 = 600
            n_l2 = 600
            w1_s = tf.get_variable('w1_s', [self.lstm_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            data = tf.matmul(lstm_h, w1_s) + tf.matmul(a, w1_a) + b1
            ## BN
            # data = tf.layers.batch_normalization(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1, training=True, name="BN_1")
            net1 = tf.nn.relu(data)
            w2 = tf.get_variable('w2', [n_l1, n_l2], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l2], trainable=trainable)
            net = tf.nn.relu(tf.matmul(net1, w2) + b2)
            ## BN
            # net = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(net1, w2) + b2, training=True, name="BN_2"))

            # tf.summary.histogram(scope+"/w1_s", w1_s)
            # tf.summary.histogram(scope+"/w1_a", w1_a)
            # tf.summary.histogram(scope+"/b1", b1)
            # tf.summary.histogram(scope+"/w2", w2)
            # tf.summary.histogram(scope+"/b2", b2)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
    
    def save(self, filename="./model.ckpt"):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.save(self.sess, filename)
    
    def load(self, filename="./model.ckpt"):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(self.sess, filename)

###############################  training  ####################################
###############################  A Demo  ####################################
MAX_EPISODES = 2000
MAX_EP_STEPS = 200 # default is 200
RENDER = False
ENV_NAME = 'Pendulum-v0'
A_BOUND = 2

# RENDER = True
MAX_EP_STEPS = 300
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
    time_sequence = 20

    oun = OUNoise(a_dim, mu=0.4)

    ddpg = DDPG_LSTM(a_dim, s_dim, a_bound, time_sequence)

    if pre_trained:
        ddpg.load('./log/model.ckpt')

    var = 3  # control exploration
    t1 = time.time()
    ave_rs = []
    his_step = []

    epsilon = 1
    for episode in range(1, 1+MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        oun.reset()
        epsilon -= (epsilon/EXPLORE)
        ## record state in one episode
        last_s = [[0]*s_dim]*(time_sequence-1)
        last_s.append(s)

        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # # get state sequence [st-3, st-2, st-1, st]
            # ep_s.append(s)
            # if len(ep_s) >= time_sequence:
            #     ts = ep_s[-time_sequence:]
            # else:
            #     ts = [[0]*s_dim]*(time_sequence-len(ep_s))
            #     ts.extend(ep_s)

            # Add exploration noise
            action_original = ddpg.choose_action(np.array(last_s))
            # a = 2*a-1 ## for sigmoid
            # a = np.clip(np.random.normal(action_original, var), -1*a_bound[0], a_bound[0])    # add randomness to action selection for exploration
            a = action_original+max(0.01, epsilon)*oun.noise()
            s_, r, done, _ = env.step(a)
            # print("step:", j, s, r, a)

            last_s_ = last_s.copy()
            del last_s_[0]
            last_s_.append(s_)
            ddpg.store_transition(np.array(last_s), a, r, np.array(last_s_))
            # print("last_s:", last_s)
            # print("last_s_:", last_s_)
            last_s = last_s_

            if ddpg.pointer > ddpg.BATCH_SIZE:
                if ddpg.pointer % 1 == 0:
                    var *= .9995    # decay the action randomness
                ddpg.learn()
                if ddpg.pointer == (ddpg.BATCH_SIZE+1):
                    print("Begin learning...")

            # s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1 or done:
                his_step.append(j)
                print("episode", episode, "consume", j, "steps, epsilon =", epsilon,"var = ", var, "ep_reward:", ep_reward)
                break
        if episode % 20 == 0:
            t_steps, t_rewards = test(env, ddpg)
            print("in test: average consume %s steps and ep_rewards is %s!"%(t_steps, t_rewards))
            if ENV_NAME == "MountainCarContinuous-v0" and t_steps < 180:
                ddpg.save("./log/model.ckpt")
                break
            if ENV_NAME == 'Pendulum-v0' and t_rewards > -150:
                ddpg.save("./log/model.ckpt")
                break
        # sys.stdout.flush()
    print('Running time: ', time.time() - t1)

def test(env, agent):
    TEST_EPISODE = 10
    ep_steps = 0
    test_reward = 0
    time_sequence = agent.time_sequence
    for _ in range(TEST_EPISODE):
        s = env.reset()
        last_s = [[0]*agent.s_dim]*(agent.time_sequence-1)
        last_s.append(s)
        for t in range(MAX_EP_STEPS):
            # env.render()
            a = agent.choose_action(np.array(last_s))
            s, r, done, _ = env.step(a)
            last_s.append(s)
            del last_s[0]
            test_reward += r
            if t == MAX_EP_STEPS-1 or done:
                # print("in test: consume %s steps!"%(t))
                ep_steps += t
                break
    return ep_steps//TEST_EPISODE, test_reward//TEST_EPISODE

if __name__ == "__main__":
    
    train()
    pass