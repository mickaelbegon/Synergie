import keras
from keras import layers

def lstm():
    """
    Build and compile an LSTM-based neural network model.

    This model takes two inputs:
    1. temporal_input: A time-series input of shape (180, 10), e.g. 180 time steps with 10 features each.
    2. scalar_input: A scalar input of shape (2,), e.g. two additional features such as mass and height.

    The model processes the temporal_input through LSTMs and dense layers,
    and the scalar_input through a small dense network. The outputs are concatenated
    and passed through final dense layers to produce a 2-class softmax prediction.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """
    # Temporal input branch
    temporal_input = keras.Input(shape=(180, 10), name='temporal_input')
    x = layers.BatchNormalization()(temporal_input)
    x = keras.layers.LSTM(128, return_sequences=True)(x)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(16, activation='relu')(x)

    # Scalar input branch (e.g., mass and height)
    scalar_input = keras.Input(shape=(2,), name='scalar_input')
    y = layers.Dense(16, activation='relu')(scalar_input)

    # Combine both branches
    combined = layers.concatenate([x, y])

    # Final dense layers
    z = layers.Dense(16, activation="relu")(combined)
    z = keras.layers.Dropout(0.2)(combined)  # Note: This dropout seems to be incorrectly using 'combined' instead of 'z'
    outputs = keras.layers.Dense(2, activation='softmax')(z)

    # Optimizer with a small learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)

    model = keras.Model([temporal_input, scalar_input], outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Build a single Transformer encoder block.

    This block uses Multi-Head Attention, Layer Normalization, and two Conv1D layers
    to form a feed-forward network. It includes residual connections and optional dropout.

    Args:
        inputs (tf.Tensor): The input tensor of shape (batch_size, timesteps, features).
        head_size (int): Dimensionality of each attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network hidden layer.
        dropout (float): Dropout rate.

    Returns:
        tf.Tensor: The output tensor after applying a Transformer encoder block.
    """
    # Normalization and Multi-Head Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs  # Residual connection

    # Feed-forward network
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res  # Residual connection

def transformer(
        input_shape=(240, 10),
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=128,
        dropout=0,
        mlp_dropout=0,
):
    """
    Build and compile a Transformer-based model.

    The model uses a series of transformer encoder blocks, followed by global average pooling,
    and merges with a scalar input (e.g., mass and height) to produce a final classification.

    Args:
        input_shape (tuple): Shape of the temporal input data (timesteps, features).
        head_size (int): Dimensionality of attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network in the encoder.
        num_transformer_blocks (int): Number of stacked transformer encoder blocks.
        mlp_units (int): Number of units in the first dense layer after pooling.
        dropout (float): Dropout rate for the transformer encoder.
        mlp_dropout (float): Dropout rate for the MLP after the encoder.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """
    n_classes = 6

    # Temporal input branch
    temporal_input = keras.Input(shape=input_shape, name="temporal_input")
    x = layers.BatchNormalization()(temporal_input)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(mlp_units, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
    x = layers.Dense(16, activation="relu")(x)

    # Scalar input branch (e.g., mass and height)
    scalar_input = keras.Input(shape=(2,), name='scalar_input')
    y = layers.Dense(16, activation='relu')(scalar_input)

    # Combine both branches
    combined = layers.concatenate([x, y])

    # Final classification layers
    z = layers.Dense(16, activation="relu")(combined)
    z = layers.Dropout(0.2)(x)  # Note: This again uses 'x' instead of 'z'. Possibly a bug that needs fixing.
    outputs = layers.Dense(n_classes, activation="softmax")(z)

    # Use a slightly higher learning rate compared to the LSTM model
    optimizer = keras.optimizers.Adam(learning_rate=0.00005)

    model = keras.Model([temporal_input, scalar_input], outputs)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def transformerTraining(hp):
    """
    Build and compile a Transformer model for hyperparameter tuning.

    This function allows hyperparameters (such as head_size, num_heads, ff_dim, and
    num_transformer_blocks) to be tuned using the Keras Tuner API.

    Args:
        hp: A hyperparameter object provided by Keras Tuner.

    Returns:
        keras.Model: A compiled Keras model with hyperparameter-defined architecture.
    """
    input_shape = (240, 10)
    head_size = hp.Int('head_size', min_value=32, max_value=512, step=32)
    num_heads = hp.Int('num_heads', min_value=2, max_value=16, step=2)
    ff_dim = hp.Int('ff_dim', min_value=128, max_value=2048, step=128)
    num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=12, step=1)
    mlp_units = 128
    dropout = 0.3
    mlp_dropout = 0.1
    n_classes = 6
    learning_rate = 0.00005

    inputs = keras.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(mlp_units, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def save_model(model, path="saved_models/model.keras"):
    """
    Save the entire model to the specified path in the Keras format.

    Args:
        model (keras.Model): The model to be saved.
        path (str): The path where the model should be saved.
    """
    # Saving the entire model architecture, weights, and optimizer state.
    keras.saving.save_model(model, path, overwrite=True)

def load_model(path="saved_models/model.keras"):
    """
    Load a model from the specified path in the Keras format.

    Args:
        path (str): The path where the model is saved.

    Returns:
        keras.Model: The loaded Keras model.
    """
    return keras.saving.load_model(path)
