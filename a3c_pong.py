from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import multiprocessing
import threading
from time import sleep

import gym

# For: Pong-v0, Pong-v4, PongDeterministic-v0:
# Size of action_space: 6 [int in [0, 255]]
# Size of observation_space: (210, 160, 3) [np.array]
#
# Pong-ram-v4: observation_space has shape (128,)
# TODO: Correct arguments.

# Hyperparameters. TODO: Correct values.
max_episode_length = 5  # Maximum length of one episode. Paper value: 5
max_global_steps = 4e6  # Maximum number of global steps. Paper value: 4 * 10**6
gamma = 0.99  # Discount rate.
decay = 0.99  # RMSProp decay factor.
epsilon = 0.1  # Epsilon hyperparameter for RMSProp.


def entropy(pmf, base=2., name=None):
    """Returns the Shannon entropy of a discrete probability distribution.

    Args:
      pmf: `Tensor` of shape [None], all values of a discrete
           probability mass function.
      base: `float`, base of the logarithm used when computing entropy.
      name: `string`, name of the operation.

    Returns:
      `Tensor` of shape [], entropy of the input pmf.
    """
    with tf.name_scope(name, "entropy", [pmf]):
        pmf = tf.convert_to_tensor(pmf)
        pmf = tf.clip_by_value(pmf, 1e-20, 1.0)  # Avoid log(0).
        return - tf.reduce_sum(pmf * tf.log(pmf)) / tf.log(base)


class Net():
    """CNN to be used globally and by workers."""
    T = 0  # Global step counter.

    # Create global RMSPropOptimizer.
    #
    # The initial learning rate will be sampled from LogUniform(10^-4, 10^-2),
    # i.e. as in the A3C paper, and then linearly annealed to zero; see:
    # https://github.com/muupan/async-rl/wiki
    #
    # TODO: Make sure this implements a /shared/ statistics g as
    # described in the A3C paper.
    learning_rate = 10 ** np.random.uniform(-4, -2)
    learning_rate = tf.train.polynomial_decay(learning_rate, T,
                                              max_global_steps,
                                              end_learning_rate=0)
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay,
                                          epsilon=epsilon)

    def __init__(self, scope, beta, n_actions):
        self.beta = beta

        # Create network layers.
        with tf.variable_scope(scope):
            # Create placeholders for inputs to be fed from experience.
            self.observations = tf.placeholder(tf.float32,
                                               shape=[None, 84, 84, 1])
            self.actions = tf.placeholder(tf.uint8, shape=[None])
            self.returns = tf.placeholder(tf.float32, shape=[None])

            # Create layers shared by actor and critic.
            with tf.variable_scope("shared"):
                self.conv1 = tf.layers.conv2d(self.observations, 16,
                                              8, strides=4, padding="same",
                                              activation=tf.nn.relu,
                                              name="conv1")
                self.conv2 = tf.layers.conv2d(self.conv1, 32, 4,
                                              strides=2, padding="same",
                                              activation=tf.nn.relu,
                                              name="conv2")
                self.fc = tf.layers.dense(tf.layers.flatten(self.conv2),
                                          256, activation=tf.nn.relu,
                                          name="fc")
            # Create actor output layer.
            with tf.variable_scope("actor"):
                self.policy = tf.layers.dense(self.fc, n_actions,
                                              activation=tf.nn.softmax,
                                              name="policy")
                self.p_max = tf.summary.scalar("p_max",
                                               tf.reduce_max(self.policy))
            # Create critic output layer.
            with tf.variable_scope("critic"):
                self.value = tf.layers.dense(self.fc, 1, name="value")
                self.value_log = tf.summary.scalar("value",
                                                   tf.reduce_mean(self.value))

        # Create collections of trainable parameters for actor and critic.
        for var in tf.trainable_variables(scope + "/shared"):
            tf.add_to_collections(["actor_params", "critic_params"], var)
        for var in tf.trainable_variables(scope + "/actor"):
            tf.add_to_collection("actor_params", var)
        for var in tf.trainable_variables(scope + "/critic"):
            tf.add_to_collection("critic_params", var)
        self.actor_params = tf.get_collection("actor_params", scope=scope)
        self.critic_params = tf.get_collection("critic_params", scope=scope)

        # Create operations for backpropagation.
        with tf.variable_scope(scope):
            self.policy_clipped = tf.clip_by_value(self.policy, 1e-20, 1.0)
            self.entropy = - tf.reduce_sum(
                self.policy_clipped * tf.log(self.policy_clipped)
            ) / tf.log(2.)
            self.relevant_probs = tf.reduce_sum(
                self.policy * tf.one_hot(self.actions, n_actions,
                                         dtype=tf.float32),
                axis=1
            )
            self.advantage = self.returns - self.value
            # Enforcing a strictly positive lower bound for relevant_probs
            # was necessary to avoid log(0) computations and warnings like:
            # "RuntimeWarning: invalid value encountered in less
            #  a = np.random.choice(self.env.action_space.n, p=policy[0])".
            # Fix found at:
            # https://github.com/awjuliani/DeepRL-Agents/issues/27
            self.loss_actor = - tf.reduce_sum(
                tf.log(tf.clip_by_value(self.relevant_probs, 1e-20, 1))
                * self.advantage
            ) - self.beta * self.entropy
            self.loss_actor_log = tf.summary.scalar("loss_actor",
                                                    self.loss_actor)
            # The 0.5 factor in loss_value was used in the A3C paper, see:
            # https://github.com/muupan/async-rl/wiki
            self.loss_critic = 0.5 * tf.reduce_sum(tf.square(self.advantage))
            self.loss_critic_log = tf.summary.scalar("loss_critic",
                                                     self.loss_critic)
            self.grads_actor = tf.gradients(self.loss_actor, self.actor_params)
            self.grads_critic = tf.gradients(self.loss_critic,
                                             self.critic_params)
            # Monitor gradient norms to detect vanishing or exploding
            # gradients.
            self.grad_norm_actor = tf.summary.scalar(
                "grad_norm_actor",
                tf.global_norm(self.grads_actor)
            )
            self.grad_norm_critic = tf.summary.scalar(
                "grad_norm_critic",
                tf.global_norm(self.grads_critic)
            )
            self.update_actor = Net.optimizer.apply_gradients(
                zip(self.grads_actor, tf.get_collection("actor_params",
                                                        scope="global"))
            )
            self.update_critic = Net.optimizer.apply_gradients(
                zip(self.grads_critic, tf.get_collection("critic_params",
                                                         scope="global"))
                )

    # TODO: Make handling of single input frame vs batch style guide compliant.
    def fwd_prop(self, processed_frame, sess):
        feed_dict = {self.observations: processed_frame}
        policy, value = sess.run([self.policy, self.value],
                                 feed_dict=feed_dict)
        return policy, value


class Worker(Net):
    """A3C workers will be instances of this class."""
    def __init__(self, scope, beta=0.01, env=gym.make("Pong-v4")):
        # Create feedforward ops
        Net.__init__(self, scope, beta, env.action_space.n)
        self.scope = scope
        self.env = env
        self.t = 0  # Thread step counter
        self.score = 0  # Sum of rewards over training epoch.

        # Set up summaries for TensorBoard.
        self.writer = tf.summary.FileWriter("/tmp/a3c/" + scope)
        self.merged = tf.summary.merge([self.p_max, self.value_log,
                                        self.grad_norm_actor,
                                        self.grad_norm_critic,
                                        self.loss_actor_log,
                                        self.loss_critic_log])

        """Process Atari frame as in Mnih et al. (2015).

        Args:
            frame: `Tensor`, usually of shape (batch_size, 210, 160, 3),
                   RGB frames of Atari emulator.
            sess: `tf.Session`

        Returns:
            `Tensor` of shape (batch_size, 84, 84, 1),
            resized greyscale version of input `frame`.
        """
        self.frame_in = tf.placeholder(tf.float32, shape=[210, 160, 3])
        self.frame = tf.convert_to_tensor(self.frame_in)
        self.frame = tf.image.resize_images(self.frame, (110, 84))
        self.frame = tf.image.rgb_to_grayscale(self.frame)
        self.frame = tf.image.crop_to_bounding_box(self.frame,
                                                   offset_height=13,
                                                   offset_width=0,
                                                   target_height=84,
                                                   target_width=84)
        self.preprocess = tf.reshape(self.frame, [-1, 84, 84, 1])

        # Synchronize thread-specific parameters.
        local_vars = tf.trainable_variables(self.scope)
        global_vars = tf.trainable_variables("global")
        self.synchronize = []
        for local_var, global_var in zip(local_vars, global_vars):
            self.synchronize.append(tf.assign(local_var, global_var))

    def interact(self, sess, steps=max_episode_length, initial_obs=None):
        """Interact with environment at most `steps` times."""
        # Get initial observation.
        if initial_obs is None:
            s_raw = self.env.reset()
            initial_obs = sess.run(self.preprocess,
                                   feed_dict={self.frame_in: s_raw})

        observations = np.zeros([steps, 84, 84, 1])
        actions = np.zeros(steps, dtype=int)
        rewards = np.zeros(steps)
        done = False
        s = initial_obs
        t_local = 0
        while t_local < steps and not done:
            observations[t_local] = s
            policy, value = self.fwd_prop(s, sess)
            # The Gym environment uses frame/action skipping by default, see
            # https://github.com/openai/gym/issues/275
            a = np.random.choice(self.env.action_space.n, p=policy[0])
            s_raw, r, done, info = self.env.step(a)
            self.score += r
            s = sess.run(self.preprocess, feed_dict={self.frame_in: s_raw})
            actions[t_local] = a
            rewards[t_local] = r
            t_local += 1

        observations = observations[:t_local]
        actions = actions[:t_local]
        rewards = rewards[:t_local]
        self.t += t_local
        Net.T += t_local

        # Compute estimates of action-value function Q by bootstrapping from
        # the critic's value estimate for the last state.
        if done:
            bootstrap = 0
        else:
            _, bootstrap = self.fwd_prop(s, sess)
        Q = np.zeros(t_local)
        Q = np.append(Q, bootstrap)
        for i, reward in enumerate(reversed(rewards)):
            Q[-i-2] = rewards[-i-1] + gamma*Q[-i-1]
        Q = Q[:-1]

        exp_buffer = {}
        keys = ["observations", "actions", "Q_estimates"]
        values = [observations, actions, Q]
        for key, value in zip(keys, values):
            exp_buffer[key] = value

        return exp_buffer, s, done

    def work(self, sess, coord, global_steps=max_global_steps):
        self.score = 0
        done = True
        print("Starting worker " + self.scope)
        with sess.as_default(), sess.graph.as_default():
            while Net.T <= global_steps and not coord.should_stop():
                sess.run(self.synchronize)

                if done:
                    s_raw = self.env.reset()  # Get initial state.
                    s = sess.run(self.preprocess,
                                 feed_dict={self.frame_in: s_raw})

                exp_buffer, s, done = self.interact(sess, initial_obs=s)

                _, _, summary = sess.run(
                    [self.update_actor, self.update_critic, self.merged],
                    feed_dict={self.observations:
                               exp_buffer["observations"], self.actions:
                               exp_buffer["actions"], self.returns:
                               exp_buffer["Q_estimates"]}
                )

                self.writer.add_summary(summary, global_step=Net.T)
        print("worker " + self.scope + " total score this epoch: " +
              str(self.score))


if __name__ == "__main__":
    # Create global network and workers.
    master = Worker("global")
    n_workers = 1  # multiprocessing.cpu_count()
    team = []
    for i in range(n_workers):
        team.append(Worker("1t_full_epoch_2018-07-13" + str(i)))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        threads = []
        for worker in team:
            t = threading.Thread(target=worker.work,
                                 args=(sess, coord),
                                 kwargs={"global_steps": max_global_steps})
            t.start()
            sleep(1)
            threads.append(t)
        coord.join(threads)
