import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf

# number of worker agents
no_of_workers = multiprocessing.cpu_count()

# maximum number of steps per episode
no_of_ep_steps = 2000

# total number of episodes
no_of_episodes = 2000

global_net_scope = 'Global_Net'

# sets how often the global network should be updated
update_global = 50

# discount factor
gamma = 0.9

# entropy factor
entropy_beta = 0.01

# learning rate for actor
lr_a = 0.0001

# learning rate for critic
lr_c = 0.0001

# boolean for rendering the environment
render=True

# directory for storing logs
log_dir = 'logs'

env = gym.make('MountainCarContinuous-v0')
env.reset()

# we get the number of states, actions and also the action bound
no_of_states = env.observation_space.shape[0]
no_of_actions = env.action_space.shape[0]
action_bound = [env.action_space.low, env.action_space.high]

print('num_states:',no_of_states)
print('num_actions:',no_of_actions)
print('action_bound:',action_bound)


class ActorCritic(object):
    def __init__(self, scope, sess, globalAC=None):

        # first we initialize the session and RMS prop optimizer for both
        # our actor and critic networks

        self.sess = sess

        self.actor_optimizer = tf.train.RMSPropOptimizer(lr_a, name='RMSPropA')
        self.critic_optimizer = tf.train.RMSPropOptimizer(lr_c, name='RMSPropC')

        # now, if our network is global then,
        if scope == global_net_scope:
            with tf.variable_scope(scope):

                # initialize states and build actor and critic network
                self.s = tf.placeholder(tf.float32, [None, no_of_states], 'S')

                # get the parameters of actor and critic networks
                self.a_params, self.c_params = self._build_net(scope)[-2:]

        # if our network is local then,
        else:
            with tf.variable_scope(scope):

                # initialize state, action and also target value as v_target
                self.s = tf.placeholder(tf.float32, [None, no_of_states], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, no_of_actions], 'A') # a_history
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                # since we are in continuous actions space, we will calculate
                # mean and variance for choosing action
                mean, var, self.v, self.a_params, self.c_params = self._build_net(scope)

                # then we calculate td error as the difference between v_target - v
                td = tf.subtract(self.v_target, self.v, name='TD_error')

                # minimize the TD error
                with tf.name_scope('critic_loss'):
                    self.critic_loss = tf.reduce_mean(tf.square(td))

                # update the mean and var value by multiplying mean with the action bound and adding var with 1e-4
                # 因为 tanh 输出的 mean 是在 [-1,1] 区间，需要转化到 action_bound 内，同时避免 var 等于零
                with tf.name_scope('wrap_action'):
                    mean, var = mean * action_bound[1], var + 1e-4

                # we can generate distribution using this updated mean and var
                normal_dist = tf.contrib.distributions.Normal(mean, var)

                # now we shall calculate the actor loss. Recall the loss function.
                with tf.name_scope('actor_loss'):
                    # calculate first term of loss which is log(pi(s))
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td) # td 的优化交给 critic_loss

                    # calculate entropy from our action distribution for ensuring exploration
                    # When the entropy value is high, every action's probability will be
                    # the same, so the agent will be unsure as to which action to perform, and when
                    # the entropy value is lowered, one action will have a higher probability than the
                    # others and the agent can pick up the action that has this high probability
                    entropy = normal_dist.entropy()

                    # we can define our final loss as,
                    self.exp_v = exp_v + entropy_beta * entropy

                    # then, we try to minimize the loss
                    self.actor_loss = tf.reduce_mean(-self.exp_v)

                # now, we choose action by drawing from the distribution and clipping it between action bounds,
                with tf.name_scope('choose_action'):
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), action_bound[0],
                                              action_bound[1])

                # calculate gradients for both of our actor and critic networks,
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.actor_loss, self.a_params)
                    self.c_grads = tf.gradients(self.critic_loss, self.c_params)

            # now, we update our global network weights,
            with tf.name_scope('sync'):

                # pull the global network weights to the local networks
                with tf.name_scope('pull'):
                    # 把每一个 g_p 赋值给 l_p
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]

                # push the local gradients to the global network
                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))

    # next, we define a function called _build_net for building our actor and critic network

    def _build_net(self, scope):
        # initialize weights
        w_init = tf.random_normal_initializer(0., .1)

        with tf.variable_scope('actor'):
            # 三层全连接：输入==> 隐藏层 l_a ==> 两个独立的输出层（mean，var）
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            # tanh 的输出在 [-1,1] 区间内
            mean = tf.layers.dense(l_a, no_of_actions, tf.nn.tanh, kernel_initializer=w_init, name='mean')
            # softplus 是平滑的 relu，表示输出大于零
            var = tf.layers.dense(l_a, no_of_actions, tf.nn.softplus, kernel_initializer=w_init, name='var')

        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 10, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return mean, var, v, a_params, c_params

    # update the local gradients to the global network
    def update_global(self, feed_dict):
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

    # get the global parameters to the local networks
    def pull_global(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    # select action
    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})[0]


class Worker(object):
    def __init__(self, name, globalAC, sess):
        # intialize environment for each worker
        self.env = gym.make('MountainCarContinuous-v0').unwrapped
        self.name = name

        # create ActorCritic agent for each worker
        self.AC = ActorCritic(name, sess, globalAC)
        self.sess = sess

    def work(self):
        global global_rewards, global_episodes
        total_step = 1
        # store state, action, reward
        buffer_s, buffer_a, buffer_r = [], [], []
        # loop if the coordinator is active and global episode is less than the maximum episode
        # 在函数中没有修改 coord，故可以不加 global 修饰符
        while not coord.should_stop() and global_episodes < no_of_episodes:

            # initialize the environment by resetting
            s = self.env.reset()

            # store the episodic reward
            ep_r = 0
            for ep_t in range(no_of_ep_steps):

                # Render the environment for only worker 1
                if self.name == 'W_0' and render:
                    self.env.render()

                # choose the action based on the policy
                a = self.AC.choose_action(s)

                # perform the action (a), recieve reward (r) and move to the next state (s_)
                s_, r, done1, info = self.env.step(a)

                # set done as true if we reached maximum step per episode
                # python 语言中的三元运算，statement1 if condition else statement2
                done2 = True if ep_t == no_of_ep_steps - 1 else False
                done = done1 or done2

                ep_r += r
                # if self.name == 'W_0':
                #     print('done:{} r:{}'.format(done,r))

                # store the state, action and rewards in the buffer
                buffer_s.append(s)
                buffer_a.append(a)

                # normalize the reward
                buffer_r.append((r-50)/50)


                # we Update the global network after particular time step
                if total_step % update_global == 0 or done:
                    if done1:
                        v_s_ = 0 # done 结束状态没有 reward，不需要考虑，注意 done 和 done1 的区别
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                        # np.newaxis 的用法，它实际等价于 None
                        # >>> x
                        # array([0, 1, 2])
                        #
                        # >>> x.shape
                        # (3,)
                        #
                        # >>> x[:, np.newaxis]
                        # array([[0],
                        #        [1],
                        #        [2]])

                    # buffer for target v
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    # np.vstack 把行向量转化成列向量
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }

                    # update global network
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # get global parameters to local ActorCritic
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(global_rewards) < 5:
                        global_rewards.append(ep_r)
                    else:
                        global_rewards.append(ep_r)
                        global_rewards[-1] = (np.mean(global_rewards[-5:]))

                    global_episodes += 1
                    break
            if self.name == 'W_0':
                print('global_episode: {} reward:{}' .format(global_episodes,ep_r))

# create a list for string global rewards and episodes
global_rewards = []
global_episodes = 0

# start tensorflow session
sess = tf.Session()

with tf.device("/cpu:0"):
    # create an instance to our ActorCritic Class
    global_ac = ActorCritic(global_net_scope, sess)

    workers = []

    # loop for each workers
    for i in range(no_of_workers):
        i_name = 'W_%i' % i
        workers.append(Worker(i_name, global_ac, sess))

coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())

# log everything so that we can visualize the graph in tensorboard
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
tf.summary.FileWriter(log_dir, sess.graph)

worker_threads = []
# start workers
for worker in workers:
    job = lambda: worker.work()
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
coord.join(worker_threads)