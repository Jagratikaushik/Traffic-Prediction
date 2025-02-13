import configparser
from sumolib import checkBinary
import os
import sys

def import_train_configuration(config_file):
    """
    Reads the training configuration file and imports its content.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the training configuration.
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    # Simulation settings
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')

    # Model settings
    config['num_layers'] = content['model'].getint('num_layers')
    config['width_layers'] = content['model'].getint('width_layers')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['training_epochs'] = content['model'].getint('training_epochs')

    # Memory settings
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')

    # Agent settings
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['gamma'] = content['agent'].getfloat('gamma')

    # Directory paths
    config['models_path_name'] = content['dir']['models_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']

    return config

def import_test_configuration(config_file):
    """
    Reads the testing configuration file and imports its content.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the testing configuration.
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    # Simulation settings
    config['gui'] = content['simulation'].getboolean('gui')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['episode_seed'] = content['simulation'].getint('episode_seed')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')

    # Agent settings
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')

    # Directory paths
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['models_path_name'] = content['dir']['models_path_name']
    
    # Model to test
    config['model_to_test'] = content['dir'].getint('model_to_test')

    return config

def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configures SUMO (Simulation of Urban MObility) parameters.

    Args:
        gui (bool): Whether to use SUMO with GUI or in command-line mode.
        sumocfg_file_name (str): Name of the SUMO configuration file.
        max_steps (int): Maximum number of simulation steps.

    Returns:
        list: SUMO command to run the simulation.
    
    Raises:
        SystemExit: If the SUMO_HOME environment variable is not set.
    """
    
    # Ensure SUMO_HOME is set in the environment
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'.")

    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    
    if tools_path not in sys.path:
        sys.path.append(tools_path)

    # Choose between GUI or command-line mode
    sumoBinary = checkBinary("sumo-gui" if gui else "sumo")

    # Construct the SUMO command
    sumo_cmd = [
        sumoBinary,
        "-c", os.path.join("intersection", sumocfg_file_name),
        "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps)
    ]

    return sumo_cmd

def set_train_path(models_path_name):
    """
    Creates a new model directory for training with an incremental version number.

    Args:
        models_path_name (str): Name of the base directory for storing models.

    Returns:
        str: Path to the newly created model directory.
    """
    # Ensure models_path_name is provided
    if not models_path_name or not isinstance(models_path_name, str):
        raise ValueError("Invalid models_path_name. Please provide a valid string.")

    # Construct the full path for models directory
    models_path = os.path.join(os.getcwd(), models_path_name)

    # Ensure the base models directory exists
    os.makedirs(models_path, exist_ok=True)

    # List all existing directories in models_path
    dir_content = os.listdir(models_path)

    # Filter directories that match naming pattern model_<number>
    previous_versions = [
        int(name.split("_")[1])
        for name in dir_content
        if name.startswith("model_") and name.split("_")[1].isdigit()
    ]

    # Determine new model version number 
    new_version = max(previous_versions) + 1 if previous_versions else 1

    # Create new directory for this training session 
    new_dir_name = f"model_{new_version}"
    data_path = os.path.join(models_path, new_dir_name)
    os.makedirs(data_path, exist_ok=True)

    return data_path
