# Modified training_main.py
from __future__ import absolute_import, print_function

import os
import datetime
from shutil import copyfile
import numpy as np  # Import numpy
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.metrics import mean_squared_error # Import mean_squared_error

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path
from data_loader import load_and_preprocess_data
from lstm_model import create_lstm_model
from lstm_trainer import train_lstm_model

from tensorflow.python.client import device_lib

if __name__ == "__main__":
    print(device_lib.list_local_devices())

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    # Create a directory for state data
    state_data_path = os.path.join(path, 'state_data')
    os.makedirs(state_data_path, exist_ok=True)

    Model = TrainModel(
        config['num_layers'],
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'],
        input_dim=config['num_states'],
        output_dim=config['num_actions']
    )

    Memory = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path,
        dpi=96
    )

    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        state_data_path=state_data_path
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    # Create LSTM model (outside the episode loop)
    lstm_sequence_length = 9
    # Dummy data for creating the LSTM model initially
    dummy_X = np.random.rand(1, lstm_sequence_length, config['num_states'])
    # Create and inject LSTM model with sigmoid activation into Simulation class:
    input_shape = (9, config['num_states'])  # Sequence length is now fixed at 9 for predictions.
    lstm_model_with_sigmoid = create_lstm_model(input_shape=input_shape)
    Simulation.set_lstm_model(lstm_model_with_sigmoid)

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])
        simulation_time, training_time = Simulation.run(episode, epsilon)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
              round(simulation_time + training_time, 1), 's')

        print("\n----- LSTM Training and Validation -----")

        X, y, scaler = load_and_preprocess_data(state_data_path, lstm_sequence_length)
        if X is None or y is None:
            print("Skipping LSTM training due to insufficient or missing data.")
        else:
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust test_size as needed

            input_shape = (X_train.shape[1], X_train.shape[2])
            # lstm_model = create_lstm_model(input_shape) # no need to create the lstm model every episode, it already has
            train_lstm_model(lstm_model, X_train, y_train, save_path=path)
            Simulation.set_lstm_model(lstm_model) # Inject the trained lstm model into the simulation class

            # Make predictions on the validation set
            y_pred = lstm_model.predict(X_val)

            # Calculate Mean Squared Error
            mse = mean_squared_error(y_val, y_pred)
            print(f"Validation Mean Squared Error: {mse:.4f}")

        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode',
                                     ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode',
                                     ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode',
                                     ylabel='Average queue length (vehicles)')

