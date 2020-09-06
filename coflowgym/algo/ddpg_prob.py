import tensorflow as tf
import numpy as np
import numpy.random as nr
import gym
import time

###############################  DDPG  ####################################

class DDPGProb(object):
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
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.softmax, name='a', trainable=trainable)

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
            saver.restore(self.sess, filename)
