import os
import sys

# možni paths za sumo (linux in windows)
sumo_paths = [
    '/usr/share/sumo/tools',
    # program files (x86) je default install location i think
    'C:\\Program Files (x86)\\Eclipse\\Sumo\\tools',
    'C:\\Program Files\\Sumo\\tools',
]

# če je sumo že v sys env variables
s_h = os.environ.get('SUMO_HOME')
if s_h:
    sumo_paths.insert(0, os.path.join(s_h, 'tools'))

for p in sumo_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

from  Model import Model
import traci # type: ignore
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
from tqdm import tqdm

import utils

tf.keras.utils.disable_interactive_logging()

class TLAgent:
    def __init__(self, env, traffic_gen, max_steps, num_experients, total_episodes, qmodel_filename, stats_filename, stats , init_epoch, learn = True):     
        self.env = env
        self.traffic_gen = traffic_gen
        self.total_episodes = total_episodes
        self.discount = 0.95
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 100
        self.num_states = 88
        self.num_actions = 4
        self.num_experiments = num_experients
        # phases are in same order as specified in the .net.xml file
        self.PHASE_NS_GREEN = 0  # action 0 code 00
        self.PHASE_NS_YELLOW = 1
        self.PHASE_NSL_GREEN = 2  # action 1 code 01
        self.PHASE_NSL_YELLOW = 3
        self.PHASE_EW_GREEN = 4  # action 2 code 10
        self.PHASE_EW_YELLOW = 5
        self.PHASE_EWL_GREEN = 6  # action 3 code 11
        self.PHASE_EWL_YELLOW = 7
        
        self.green_duration = 10
        self.yellow_duration = 4
        self.stats = stats
        self.QModel = None
        self.tau = 20
        self.TargetQModel = None
        self.qmodel_filename = qmodel_filename
        self.stats_filename = stats_filename
        self.init_epoch = init_epoch
        self._load_models(learn)
        self.max_steps = max_steps
                        
    def _load_models(self, learn=True):
        self.QModel = Model(self.num_states, self.num_actions)
        self.TargetQModel = Model(self.num_states, self.num_actions)

        if self.init_epoch != 0 or not learn:
            print('model read from file')
            if os.path.exists(self.qmodel_filename):
                loaded_keras_model = load_model(self.qmodel_filename)
                self.QModel.model.set_weights(loaded_keras_model.get_weights())
                self.TargetQModel.model.set_weights(loaded_keras_model.get_weights())
            else:
                print(f"Warning: File {self.qmodel_filename} not found. Starting with fresh weights.")

        return self.QModel, self.TargetQModel

    def _preprocess_input(self, state):
        state = np.reshape(state, [1, self.num_states])
        return state
    
    def _add_to_replay_buffer(self, curr_state, action, reward, next_state, done):
        self.replay_buffer.append((curr_state, action, reward, next_state, done))
        
    def _sync_target_model(self):
        self.TargetQModel.set_weights( self.QModel.get_weights()) 
        
    def _replay(self):
        mini_batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))

        # stack vse state-e v en array in naredi SAMO 2 predict-a
        # namesto 200 (po 1 na sample za QModel + 1 za TargetQModel)
        curr_states = np.array([m[0][0] for m in mini_batch])
        next_states = np.array([m[3][0] for m in mini_batch])

        # batched predictions - to je glavni speedup
        q_curr = self.QModel.predict(curr_states)
        q_next = self.TargetQModel.predict(next_states)

        # update Q values for taken actions
        for i, (_, action, reward, _, done) in enumerate(mini_batch):
            if done:
                q_curr[i][action] = reward
            else:
                q_curr[i][action] = reward + self.discount * np.max(q_next[i])

        # train_on_batch ima manj overhead-a kot fit() za en sam batch
        self.QModel.model.train_on_batch(curr_states, q_curr)

    def _agent_policy(self, episode, state, learn = True):
        if learn:
            epsilon = max(0.05, 1 - episode / self.total_episodes)
            choice  = np.random.random()
            if choice <= epsilon:
                action = np.random.choice(range(self.num_actions))
            else:
                action =  np.argmax(self.QModel.predict(state))
        else:
            action =  np.argmax(self.QModel.predict(state))
                
        return action
        
     # SET IN SUMO THE CORRECT YELLOW PHASE
    def _set_yellow_phase(self, old_action):
        yellow_phase = old_action * 2 + 1 # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase("TL", yellow_phase)

    # SET IN SUMO A GREEN PHASE
    def _set_green_phase(self, action):
        if action == 0:
            traci.trafficlight.setPhase("TL", self.PHASE_NS_GREEN)
        elif action == 1:
            traci.trafficlight.setPhase("TL", self.PHASE_NSL_GREEN)
        elif action == 2:
            traci.trafficlight.setPhase("TL", self.PHASE_EW_GREEN)
        elif action == 3:
            traci.trafficlight.setPhase("TL", self.PHASE_EWL_GREEN)

    
    def evaluate_model(self, experiment, seeds):
        self.traffic_gen.generate_routefile(seeds[self.init_epoch])
        curr_state = self.env.start()
        
        for e in range( self.init_epoch, self.total_episodes):
            done = False
            sum_intersection_queue = 0
            sum_neg_rewards = 0
            old_action = None
            while not done:
                curr_state = self._preprocess_input( curr_state )
                action = self._agent_policy( e, curr_state, learn = False)
                yellow_reward = 0
                    
                if old_action!= None and old_action != action:
                    self._set_yellow_phase(old_action)
                    yellow_reward, _ , _ = self.env.step(self.yellow_duration)
                   
                self._set_green_phase(action)
                reward, next_state, done = self.env.step(self.green_duration)
                reward += yellow_reward
                next_state = self._preprocess_input( next_state )
                curr_state = next_state
                old_action = action
                sum_intersection_queue += self.env.last_queue_sum
                if reward < 0:
                    sum_neg_rewards += reward

            steps_used = max(1, self.env.steps)
            self._save_stats(experiment, e, sum_intersection_queue / steps_used, sum_neg_rewards / steps_used)
            print('sum_neg_rewards={}'.format(sum_neg_rewards))
            print('sum_intersection_queue={}'.format(sum_intersection_queue))
            print('Epoch {} complete'.format(e))
            if e != 0:
                utils.remove_stats(experiment, e-1)
            elif experiment != 0:
                utils.remove_stats(experiment-1, self.total_episodes-1)
            if e + 1 < self.total_episodes:
                self.traffic_gen.generate_routefile(seeds[e+1])
            curr_state = self.env.reset()

    def execute_classical(self, experiment, seeds):
        self.traffic_gen.generate_routefile(seeds[self.init_epoch])
        self.env.start()
         
        for e in range(self.init_epoch, self.total_episodes):
            done = False
            sum_intersection_queue = 0
            sum_neg_rewards = 0
            while not done:
                for action in range(self.num_actions):
                    self._set_green_phase(action)
                    reward, _, done = self.env.step(self.green_duration)
                    self._set_yellow_phase(action)
                    yellow_reward, _, _ = self.env.step(self.yellow_duration)
                    reward += yellow_reward
                    if reward < 0:
                        sum_neg_rewards += reward
                    sum_intersection_queue += self.env.last_queue_sum
                    if done:
                        break

            steps_used = max(1, self.env.steps)
            self._save_stats(experiment, e, sum_intersection_queue / steps_used, sum_neg_rewards / steps_used)
            print('sum_neg_rewards={}'.format(sum_neg_rewards))
            print('sum_intersection_queue={}'.format(sum_intersection_queue))
            print('Epoch {} complete'.format(e))
            if e != 0:
                utils.remove_stats(experiment, e-1)
            elif experiment != 0:
                utils.remove_stats(experiment-1, self.total_episodes-1)
            if e + 1 < self.total_episodes:
                self.traffic_gen.generate_routefile(seeds[e+1])
            self.env.reset()
            
        
    def train(self, experiment):
        self.traffic_gen.generate_routefile(experiment * self.total_episodes + self.init_epoch)
        curr_state = self.env.start()

        ep_bar = tqdm(range(self.init_epoch, self.total_episodes),
                      desc=f"Exp {experiment}", unit="ep", position=0, leave=True)

        for e in ep_bar:
            curr_state = self._preprocess_input(curr_state)
            old_action =  None
            done = False
            sum_intersection_queue = 0
            sum_neg_rewards = 0
            sum_delay = 0
            sum_stops = 0
            sum_co2 = 0

            if e > 0 and e % self.tau == 0:
                self._sync_target_model()

            step_bar = tqdm(total=self.max_steps, desc=f"  Ep {e:3d} steps",
                            unit="step", position=1, leave=False)

            while not done:
                action = self._agent_policy(e, curr_state)
                yellow_reward = 0

                if old_action != None and old_action != action:
                    self._set_yellow_phase(old_action)
                    yellow_reward, _ , _ = self.env.step(self.yellow_duration)
                    step_bar.update(self.yellow_duration)

                self._set_green_phase(action)
                reward, next_state, done = self.env.step(self.green_duration)
                step_bar.update(self.green_duration)
                reward += yellow_reward
                next_state = self._preprocess_input( next_state )
                self._add_to_replay_buffer( curr_state, action, reward, next_state, done )

                self._replay()

                curr_state = next_state
                old_action = action
                sum_intersection_queue += self.env.last_queue_sum
                sum_delay += self.env.last_delay_sum
                sum_stops += self.env.last_stops_sum
                sum_co2 += self.env.last_co2_sum
                if reward < 0:
                    sum_neg_rewards += reward

            step_bar.close()
            steps_used = max(1, self.env.steps)

            avg_intersection_queue = sum_intersection_queue / steps_used
            avg_neg_rewards = sum_neg_rewards / steps_used
            avg_delay = sum_delay / steps_used
            avg_stops = sum_stops / steps_used
            avg_co2 = sum_co2 / steps_used

            ep_bar.set_postfix({
                'reward': f'{avg_neg_rewards:.4f}',
                'queue': f'{avg_intersection_queue:.2f}',
                'delay': f'{avg_delay:.2f}',
            })

            self._save_stats(experiment, e, avg_intersection_queue, avg_neg_rewards, avg_delay, avg_stops, avg_co2)

            utils.save_qmodel(self.QModel, experiment, e)
            if e != 0:
                utils.remove_qmodel(experiment, e-1)
                utils.remove_stats(experiment, e-1)
            if e + 1 < self.total_episodes:
                seed = experiment * self.total_episodes + e + 1
                self.traffic_gen.generate_routefile(seed)
            curr_state = self.env.reset()
            
    def _save_stats(self, experiment, episode, avg_queue, avg_rewards, avg_delay=0, avg_stops=0, avg_co2=0):
        self.stats['rewards'][experiment, episode] = avg_rewards
        self.stats['intersection_queue'][experiment, episode] = avg_queue
        self.stats['delay'][experiment, episode] = avg_delay
        self.stats['stops'][experiment, episode] = avg_stops
        self.stats['co2'][experiment, episode] = avg_co2
        utils.save_stats(self.stats, experiment, episode)