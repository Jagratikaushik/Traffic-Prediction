import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape, lstm_units=50, learning_rate=0.001):
    """
    Creates an LSTM model for traffic state prediction.

    Args:
        input_shape (tuple): Shape of the input data (sequence_length, num_states).
        lstm_units (int): Number of LSTM units in each layer.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tf.keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential()
    # First LSTM layer
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    # Second LSTM layer
    model.add(LSTM(units=lstm_units))
    # Output layer with sigmoid activation for binary predictions
    model.add(Dense(units=input_shape[1], activation='sigmoid'))  # Sigmoid activation added here
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy',  # Binary crossentropy for binary outputs
                  metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    # Example usage:
    input_shape = (10, 80)  # Example: sequence length = 10, num_states = 80
    lstm_model = create_lstm_model(input_shape)
