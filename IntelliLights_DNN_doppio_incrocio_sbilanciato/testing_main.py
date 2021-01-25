from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    path = config['models_path_name']
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path_1, model_path_2, plot_path_1, plot_path_2 = set_test_path(config['models_path_name'], config['model_to_test'])

    Model_1 = TestModel(
        input_dim=config['num_states'],
        model_path=model_path_1,
        tl=1
    )

    Model_2 = TestModel(
        input_dim=config['num_states'],
        model_path=model_path_2,
        tl=2
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path_1,
        plot_path_2, 
        dpi=96
    )
        
    Simulation = Simulation(
        [Model_1, Model_2],
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at: " + plot_path_1 + "and " + plot_path_2)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path_1, 'testing_settings.ini'))
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path_2, 'testing_settings.ini'))

    Visualization.save_data_and_plot_1(data=Simulation.reward_episode_1, filename='reward_1', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot_2(data=Simulation.reward_episode_2, filename='reward_2', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot_1(data=Simulation.queue_length_episode_1, filename='queue_1', xlabel='Step', ylabel='Queue lenght (vehicles)')
    Visualization.save_data_and_plot_2(data=Simulation.queue_length_episode_2, filename='queue_2', xlabel='Step', ylabel='Queue lenght (vehicles)')
