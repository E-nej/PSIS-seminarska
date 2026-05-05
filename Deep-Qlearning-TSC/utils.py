# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:55:20 2020

@author: XZ01M2
"""

import os

import matplotlib.pyplot as plt # type: ignore
import numpy as np 
import glob
import seaborn as sns # type: ignore

FOLDER = 'results/'

def get_file_names():
    qmodel_file_name = glob.glob(f'{FOLDER}qmodel*')
    stats_file_name = glob.glob(f'{FOLDER}stats*')
    
    if not qmodel_file_name:
        qmodel_file_name = ''
    else:
        qmodel_file_name = qmodel_file_name[0]

    if not stats_file_name:
        stats_file_name = ''
    else:
        stats_file_name = stats_file_name[0]

    return qmodel_file_name, stats_file_name

def get_init_epoch( filename,total_episodes ):
    if filename:
        index = filename.find('_')
        exp_start = index + 1 
        exp_end  = int(filename.find('_', exp_start))
        exp = int(filename[exp_start:exp_end])
        epoch_start= exp_end + 1
        epoch_end = int(filename.find('.', epoch_start))
        epoch = int(filename[epoch_start:epoch_end])
        if epoch < total_episodes -1:
            epoch +=1
        else:
            epoch = 0
            exp +=1
    else:
        exp=0
        epoch = 0
    return exp , epoch

def get_stats(stats_filename, num_experiments, total_episodes, learn = True):
    if stats_filename and learn:
        stats = np.load(stats_filename, allow_pickle = True)[()]
        expected_shape = (num_experiments, total_episodes)
        if ('rewards' not in stats or 'intersection_queue' not in stats or
                stats['rewards'].shape != expected_shape or
                stats['intersection_queue'].shape != expected_shape):
            print(f"Stats file shape mismatch: rewards={stats.get('rewards').shape if 'rewards' in stats else None}, "
                  f"intersection_queue={stats.get('intersection_queue').shape if 'intersection_queue' in stats else None}. "
                  f"Reinitializing stats to {expected_shape}.")
            reward_store = np.zeros(expected_shape)
            intersection_queue_store = np.zeros(expected_shape)
            if 'rewards' in stats and stats['rewards'].ndim == 2:
                n_exp = min(num_experiments, stats['rewards'].shape[0])
                n_ep = min(total_episodes, stats['rewards'].shape[1])
                reward_store[:n_exp, :n_ep] = stats['rewards'][:n_exp, :n_ep]
            if 'intersection_queue' in stats and stats['intersection_queue'].ndim == 2:
                n_exp = min(num_experiments, stats['intersection_queue'].shape[0])
                n_ep = min(total_episodes, stats['intersection_queue'].shape[1])
                intersection_queue_store[:n_exp, :n_ep] = stats['intersection_queue'][:n_exp, :n_ep]
            stats = {'rewards': reward_store, 'intersection_queue': intersection_queue_store}
    else:
        reward_store = np.zeros((num_experiments,total_episodes))
        intersection_queue_store = np.zeros((num_experiments,total_episodes))
        stats = {'rewards': reward_store, 'intersection_queue': intersection_queue_store }

    return stats 
    
def plot_sample(sample, title, xlabel, legend_label, show= True):
   #plt.hist(sample, bins = 5, histtype = 'bar')
    #plt.xlabel(xlabel)
    ax= sns.distplot(sample, kde=True, label =  legend_label)
    ax.set(xlabel=xlabel, title= title)
    ax.legend()
    if show:
        plt.show()   
    
def plot_rewards( reward_store):
    x = np.mean(reward_store, axis = 0 )
    plt.plot( x , label = "Cummulative negative wait times") 
    plt.xlabel('Episodes') 
    plt.ylabel('Cummulative negative wait times') 
    plt.title('Cummulative negative wait times across episodes') 
    plt.legend() 
    plt.show() 
    
def plot_intersection_queue_size( intersection_queue_store):
    x = np.mean(intersection_queue_store, axis = 0 )
    plt.plot(x, label = "Cummulative intersection queue size ", color='m') 
    plt.xlabel('Episodes') 
    plt.ylabel('Cummulative intersection queue size') 
    plt.title('Cummulative intersection queue size across episodes') 
    plt.legend() 
    plt.show() 
    
def remove_qmodel(experiment, e):
    os.remove('{}qmodel_{}_{}.keras'.format(FOLDER, experiment, e))

def remove_stats(experiment, e):
    os.remove('{}stats_{}_{}.npy'.format(FOLDER, experiment, e))

def save_qmodel(qmodel, experiment, e):
    qmodel.save('{}qmodel_{}_{}.keras'.format(FOLDER, experiment, e))
    
def save_stats(stats, experiment, e):
    np.save('{}stats_{}_{}.npy'.format(FOLDER, experiment, e), stats)