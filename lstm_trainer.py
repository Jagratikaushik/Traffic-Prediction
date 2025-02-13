# lstm_trainer.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os

def train_lstm_model(model, X, y, validation_split=0.2, epochs=10, batch_size=32, save_path=None):
    """
    Trains the LSTM model and saves it if a save path is provided.

    Args:
        model (tf.keras.models.Sequential): The LSTM model to train.
        X (np.array): Input data.
        y (np.array): Target data.
        validation_split (float): Fraction of data to use for validation.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        save_path (str, optional): Path to save the trained model. Defaults to None.

    Returns:
        tuple: (history, mse) where history is the training history and mse is the
               mean squared error on the validation set.
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation Mean Squared Error: {mse}")

    if save_path:
        model.save(os.path.join(save_path, 'lstm_traffic_model.h5'))
        print(f"LSTM model saved to {save_path}")

    return history, mse

if __name__ == '__main__':
    # Example Usage:
    from lstm_model import create_lstm_model

    # Generate some dummy data
    sequence_length = 10
    num_states = 80
    num_samples = 100
    X = np.random.rand(num_samples, sequence_length, num_states)
    y = np.random.rand(num_samples, num_states)
    input_shape = (sequence_length, num_states)

    # Create and train the model
    lstm_model = create_lstm_model(input_shape)
    save_path = './models' # Replace with your desired save path
    history, mse = train_lstm_model(lstm_model, X, y, save_path=save_path)
