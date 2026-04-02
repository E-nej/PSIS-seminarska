# -*- coding: utf-8 -*-

from  Model import Model
from SumoEnv import SumoEnv
import numpy as np
from TrafficGenerator import TrafficGenerator
import utils 
import copy

from TLAgent import TLAgent

if __name__ == "__main__":
    # --- TRAINING OPTIONS ---
    training_enabled = True
    gui = False
   
    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = 'sumo'
    else:
        sumoBinary = 'sumo-gui'

    # initializations
    # max_steps = 5400  # seconds = 1 h 30 min each episode
    # total_episodes = 100
    max_steps = 150
    total_episodes = 3
    num_experiments = 1
    learn = training_enabled
    traffic_gen = TrafficGenerator(max_steps)
    qmodel_filename, stats_filename = utils.get_file_names()
    init_experiment, init_epoch = utils.get_init_epoch( stats_filename, total_episodes)
    if not learn:
        init_experiment, init_epoch = 0, 0
    print('init_experiment={} init_epoch={}'.format(init_experiment,init_epoch ))
    stats = utils.get_stats(stats_filename, num_experiments, total_episodes)
    
    for experiment in range(init_experiment, num_experiments):
        env = SumoEnv(sumoBinary,max_steps )
        tl = TLAgent( env, traffic_gen, max_steps, num_experiments, total_episodes, qmodel_filename, stats_filename, stats,init_epoch, learn )
        init_epoch = 0 # reset init_epoch after first experiment
        if learn:
            tl.train(experiment)
        else:
            seeds = np.load('seed.npy')
            tl.evaluate_model( experiment, seeds)
            
        stats = copy.deepcopy(tl.stats)
        print(stats['rewards'][0:experiment+1, :])
        print(stats['intersection_queue'][0:experiment+1, :])
        utils.plot_rewards(stats['rewards'][0:experiment+1, :])
        utils.plot_intersection_queue_size( stats['intersection_queue'][0:experiment+1, :])
        del env
        del tl
        print('Experiment {} complete.........'.format(experiment))