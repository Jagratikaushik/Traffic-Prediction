# data_loader.py
import os
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob

def load_and_preprocess_data(csv_folder, sequence_length=9):
    """
    Loads state data from CSV files, preprocesses it, and prepares it for LSTM training.

    Args:
        csv_folder (str): Path to the folder containing state CSV files.
        sequence_length (int): Length of the sequence used for prediction (predicting the next state).

    Returns:
        tuple: (X, y, scaler) where X is the input data, y is the target data, and scaler
               is the MinMaxScaler used for scaling.
    """
    all_states = []

    #Iterate through state files in a sorted order
    for filename in sorted(glob.glob(os.path.join(csv_folder, "state_data_episode_*.csv"))):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_folder, filename)
            with open(filepath, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                header = next(csv_reader)  # Skip header row
                for row in csv_reader:
                    try:
                        state = [float(x) for x in row[1:]]  # Convert state values to floats
                        all_states.append(state)
                    except ValueError as e:
                        print(f"Error converting row to float: {row}")
                        raise e  # Reraise exception to stop execution

    # Check if there are enough states to create at least one sequence
    if len(all_states) <= sequence_length:
        print(f"Not enough state data to create sequences (need > {sequence_length} states). Skipping LSTM training for this episode.")
        return None, None, None

    # Scale the data
    all_states = np.array(all_states)
    noise = np.random.normal(0, 0.0001, all_states.shape)  # Small noise
    all_states = all_states + noise
    all_states = all_states.tolist()  # Convert back to list for scaling
    scaler = MinMaxScaler()
    scaled_states = scaler.fit_transform(all_states)

    X, y = [], []
    for i in range(len(scaled_states) - sequence_length):
        X.append(scaled_states[i:i + sequence_length])
        y.append(scaled_states[i + sequence_length]) #Predict the 10th state

    X, y = np.array(X), np.array(y)

    # Check again in case X or y are empty
    if len(X) == 0 or len(y) == 0:
      print("Not enough data to create sequences X and y")
      return None, None, None

    return X, y, scaler
