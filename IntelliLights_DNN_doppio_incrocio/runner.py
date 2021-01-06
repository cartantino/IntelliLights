from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
from utils import import_train_configuration, set_sumo, set_train_path
from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
import datetime
from shutil import copyfile


# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options



if __name__ == "__main__":

    options = get_options()
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    [path_1, path_2] = set_train_path(config['models_path_name'])


    Model_1 = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    
    Model_2 = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    Memory_1 = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )
    
    Memory_2 = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )


    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path_1,
        path_2, 
        dpi=96
    )
        
    Simulation = Simulation(
        [Model_1, Model_2],
        [Memory_1, Memory_2],
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()


    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:" + path_1 + " and " +  path_2)

    Model_1.save_model_1(path_1)
    Model_2.save_model_2(path_2)

    copyfile(src='training_settings.ini', dst=os.path.join(path_1, 'training_settings.ini'))
    copyfile(src='training_settings.ini', dst=os.path.join(path_2, 'training_settings.ini'))

    Visualization.save_data_and_plot_1(data=Simulation.reward_store_1, filename='reward_1', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot_2(data=Simulation.reward_store_2, filename='reward_2', xlabel='Episode', ylabel='Cumulative negative reward')

    Visualization.save_data_and_plot_1(data=Simulation.cumulative_wait_store_1, filename='delay_1', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot_2(data=Simulation.cumulative_wait_store_2, filename='delay_2', xlabel='Episode', ylabel='Cumulative delay (s)')

    Visualization.save_data_and_plot_1(data=Simulation.avg_queue_length_store_1, filename='queue_1', xlabel='Episode', ylabel='Average queue length (vehicles)')
    Visualization.save_data_and_plot_2(data=Simulation.avg_queue_length_store_2, filename='queue_2', xlabel='Episode', ylabel='Average queue length (vehicles)')

    Visualization.save_data_and_plot_1(data=Simulation.avg_speed_store_1, filename='avg_speed_1', xlabel='Episode', ylabel='Average speed (m/s)')
    Visualization.save_data_and_plot_2(data=Simulation.avg_speed_store_2, filename='avg_speed_2', xlabel='Episode', ylabel='Average speed (m/s)')
