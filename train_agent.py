from cnn_model import cnn_model
import gym
import highway_env
import sys
sys.path.insert(0, '/content/highway-env/scripts/')
import numpy as np
import ast
import keras
from keras.layers import Input
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('huber_log.csv', append=True, separator=';')


class Agent():
    def __init__(self, discount_factor=0.1, max_steps=500, sync_models=10,
                 model_type='dqn', betches_size=50, action_space=4,
                 per=False, priority_scale=0.7, max_epsilon=0.99, env_number=1, loss='mse',
                 decay_factor=0.15) -> None:
        self.env_number = env_number
        self.action_space = action_space
        self.input_shape = [self.action_space, 128, 128]
        model_input = Input(self.input_shape, name='input')
        self.model_type = model_type
        self.batch_size = betches_size
        self.q_model = cnn_model(model_input, loss=loss)
        self.target_model = keras.models.clone_model(self.q_model)

        self.discount_factor = discount_factor
        self.max_steps = max_steps
        self.init_env()
        self.sync_models = sync_models
        self.actions_number = len(highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL)

        self.priority_scale = priority_scale
        self.per = per
        self.max_epsilon = max_epsilon
        self.decay_factor = decay_factor
        self.steps_taken = []

        self.models_counter = 0

        self.begining_states_history_last = np.zeros([0, 4, 128, 128])
        self.ending_states_history_last = np.zeros([0, 4, 128, 128])
        self.rewards_history_last = np.zeros([0, 1])
        self.Qs_history_last = np.zeros([0, self.actions_number])
        self.actions_last = np.zeros([0, 1])
        self.priorities_last = np.zeros([0, 1])
        self.errors_last = np.zeros([0, 1])

        self.reset_history()

    def init_env(self):
        file = open(f'configs/config_ex{self.env_number}.txt', 'r')
        contents = file.read()
        config1 = ast.literal_eval(contents)
        file.close()
        self.env = gym.make("highway-fast-v0")
        self.env.config['duration'] = 500
        self.env.configure(config1)

    def reset_history(self):
        self.begining_states_history = np.zeros([0, 4, 128, 128])
        self.ending_states_history = np.zeros([0, 4, 128, 128])
        self.rewards_history = np.zeros([0, 1])
        self.Qs_history = np.zeros([0, self.actions_number])
        self.actions = np.zeros([0, 1])
        self.prev_reward = 1

        if (len(self.begining_states_history_last) > 0):
            indicies = np.unique(np.random.choice(np.arange(len(self.begining_states_history_last), dtype=int),
                                                  size=(int)(self.batch_size / 2)))
            self.begining_states_history = self.begining_states_history_last.copy()[indicies]
            self.ending_states_history = self.ending_states_history_last.copy()[indicies]
            self.rewards_history = self.rewards_history_last.copy()[indicies]
            self.Qs_history = self.Qs_history_last.copy()[indicies]
            self.actions = self.actions_last.copy()[indicies]

    def get_probabilities(self):
        priorites = (self.errors ** self.priority_scale) / sum(self.errors ** self.priority_scale)
        return priorites

    def fit_model(self, done):
        # debug
        if (len(self.rewards_history) != len(self.begining_states_history) != len(self.ending_states_history) != len(
                self.Qs_history) != len(self.actions)):
            print("Problem")
        rewards = self.rewards_history
        observations_end = self.ending_states_history
        rows = np.arange(len(rewards))
        # use ddqn model
        qs = self.q_model.predict(self.begining_states_history)
        if self.model_type == 'ddqn':
            # qs = self.q_model.predict(observations_end)
            next_actions = np.argmax(self.q_model.predict(observations_end), axis=1)
            ys = rewards + np.array(
                [(self.discount_factor * self.target_model.predict(observations_end)[rows, next_actions.flatten()])]).T
            qs_targets = self.q_model.predict(self.begining_states_history)
        # use dqn
        else:
            ys = rewards + np.array([self.discount_factor * np.max(self.target_model.predict(observations_end),
                                                                   axis=1)]).T
            qs_targets = qs.copy()
        if self.per:
            self.errors = (abs(ys.flatten() - qs[rows, np.array(self.actions.flatten(), dtype=int)])).flatten()
        else:
            self.errors = np.ones([len(ys)])
        if done:
            ys[-1, 0] = rewards[-1, 0]

        qs_targets[rows, np.array(self.actions, dtype=int).flatten()] = ys.flatten()
        if self.per:
            prorities = self.get_probabilities()

            indicies = np.unique(np.random.choice(np.arange(len(qs_targets)),
                                                  size=self.batch_size,
                                                  p=prorities))
        else:
            indicies = np.unique(np.random.choice(np.arange(len(qs_targets)),
                                                  size=self.batch_size))
        to_x_train = self.begining_states_history[indicies]
        to_y_train = qs_targets[indicies]
        self.q_model.fit(x=to_x_train, y=to_y_train, verbose=1, batch_size=self.batch_size, callbacks=[csv_logger])

    def train_agent(self, episodes, epsilon, discount_factor):
        max_reward = 0
        self.reset_history()
        self.epsilon = epsilon
        for episode in range(episodes):
            current_reward = 0
            print(f'STARTING TO FIT, episode: {len(self.steps_taken)}')
            observation = self.env.reset()
            expended_observation = np.expand_dims(observation, axis=0)
            for step in range(self.max_steps):  # need to config it in init so that max steps <= duration
                self.begining_states_history = np.concatenate([self.begining_states_history, expended_observation])
                step_q = self.q_model.predict(expended_observation)
                self.Qs_history = np.concatenate([self.Qs_history, step_q])

                q_flatten = np.matrix.flatten(step_q)
                # soft epsilon
                # TODO add function for take action
                self.env.get_available_actions()
                prob = np.ones(self.actions_number) * ((1 - self.epsilon) / len(self.env.get_available_actions()))
                prob[np.argmax(step_q)] += self.epsilon
                # allow only possible actions
                mask = np.zeros([self.actions_number])
                mask[self.env.get_available_actions()] = 1
                prob *= mask
                # higher priority to speed up and lower priority to slow down
                prob[3] *= 2
                prob[4] *= 0.5

                prob *= 1 / prob.sum()
                action = np.random.choice(np.arange(self.actions_number), p=prob)

                # simulate enviroment stockticness
                prob = np.ones(self.actions_number) * ((0.15) / (self.actions_number - 1))
                prob[action] = 0.85

                action = np.random.choice(np.arange(self.actions_number), p=prob)
                observation_end, step_reward, step_done, _ = self.env.step(action)

                tmp_reward = step_reward
                # Crashed
                if (step_reward < 0.4):
                    step_reward = -20
                # High speed
                if (step_reward > 0.9 and self.prev_reward < 0.9):
                    step_reward *= 5
                self.prev_reward = tmp_reward

                print('action:', action)
                print('reward:', step_reward)
                self.actions = np.concatenate([self.actions, np.expand_dims([action], axis=0)])
                self.rewards_history = np.concatenate([self.rewards_history, np.expand_dims([step_reward], axis=0)])
                current_reward += step_reward
                self.ending_states_history = np.concatenate(
                    [self.ending_states_history, np.expand_dims(observation_end, axis=0)])
                self.fit_model(step_done)

                self.models_counter += 1
                if self.models_counter % self.sync_models == 0:
                    self.target_model = keras.models.clone_model(self.q_model)

                if self.models_counter % 10 == 0:
                    if (self.epsilon * (1 + self.decay_factor)) < self.max_epsilon:
                        self.epsilon *= (1 + self.decay_factor)
                        print(f'in iteration number {self.models_counter} epsilon is {self.epsilon}')
                    else:
                        self.epsilon = self.max_epsilon

                if self.models_counter % (self.sync_models * 3) == 0:
                    self.reset_history()

                if step_done or step == (self.max_steps - 1):
                    if max_reward < current_reward:
                        max_reward = current_reward
                    self.begining_states_history_last = np.concatenate(
                        [self.begining_states_history_last, expended_observation])
                    self.ending_states_history_last = np.concatenate(
                        [self.ending_states_history_last, np.expand_dims(observation_end, axis=0)])
                    self.rewards_history_last = np.concatenate(
                        [self.rewards_history_last, np.expand_dims([step_reward], axis=0)])
                    self.Qs_history_last = np.concatenate([self.Qs_history_last, step_q])
                    self.actions_last = np.concatenate([self.actions_last, np.expand_dims([action], axis=0)])

                    print(f"episode ended in {step} steps")
                    self.steps_taken.append(step)
                    break

                expended_observation = np.expand_dims(observation_end, axis=0)

        return max_reward, self.models_counter, self.Qs_history



# ddqn.train_agent(episodes=5, epsilon=0.01, discount_factor=0.9)
# log_data = pd.read_csv('training.log', sep=',', engine='python')
# episodes = 5
# print(log_data['loss'])
# while(log_data['loss'].to_numpy()[0] > 1):
#   ddqn.train_agent(episodes=5, epsilon=ddqn.epsilon, discount_factor=ddqn.discount_factor)
#   log_data = pd.read_csv('training.log', sep=',', engine='python')
#   episodes += 5
#
# print(f'Total episodes: {episodes}')