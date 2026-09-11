import keras
import numpy as np
import tensorflow as tf
from keras import layers

from synergie.config import SUCCESS_WINDOW_FRAMES, TYPE_WINDOW_FRAMES


def _compile_model(model: keras.Model, learning_rate: float = 0.00005) -> keras.Model:
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def _residual_block(x, filters: int, dilation_rate: int, dropout: float):
    residual = x
    x = layers.Conv1D(filters, kernel_size=3, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel_size=3, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters, kernel_size=1, padding="same")(residual)

    x = layers.Add()([x, residual])
    return layers.Activation("relu")(x)


def _inception_module(x, filters: int, bottleneck_filters: int):
    if int(x.shape[-1]) > 1:
        x_reduced = layers.Conv1D(bottleneck_filters, kernel_size=1, padding="same", activation="relu")(x)
    else:
        x_reduced = x

    branch_9 = layers.Conv1D(filters, kernel_size=9, padding="same", activation="relu")(x_reduced)
    branch_19 = layers.Conv1D(filters, kernel_size=19, padding="same", activation="relu")(x_reduced)
    branch_39 = layers.Conv1D(filters, kernel_size=39, padding="same", activation="relu")(x_reduced)
    pooled = layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
    pooled = layers.Conv1D(filters, kernel_size=1, padding="same", activation="relu")(pooled)

    x = layers.Concatenate()([branch_9, branch_19, branch_39, pooled])
    return layers.BatchNormalization()(x)


def lstm(
    input_shape=(SUCCESS_WINDOW_FRAMES, 10),
    first_units: int = 128,
    second_units: int = 64,
    dropout: float = 0.4,
    learning_rate: float = 0.00001,
):
    temporal_input = keras.Input(shape=input_shape, name="temporal_input")
    x = layers.BatchNormalization()(temporal_input)
    x = layers.LSTM(first_units, return_sequences=True)(x)
    x = layers.LSTM(second_units)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation="relu")(x)

    scalar_input = keras.Input(shape=(2,), name="scalar_input")
    y = layers.Dense(16, activation="relu")(scalar_input)

    combined = layers.Concatenate()([x, y])
    z = layers.Dense(16, activation="relu")(combined)
    z = layers.Dropout(0.2)(z)
    outputs = layers.Dense(2, activation="softmax")(z)

    model = keras.Model([temporal_input, scalar_input], outputs)
    return _compile_model(model, learning_rate=learning_rate)


def tcn_success(
    input_shape=(SUCCESS_WINDOW_FRAMES, 10),
    filters: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.00003,
):
    temporal_input = keras.Input(shape=input_shape, name="temporal_input")
    x = temporal_input
    for dilation_rate in (1, 2, 4, 8):
        x = _residual_block(x, filters=filters, dilation_rate=dilation_rate, dropout=dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)

    scalar_input = keras.Input(shape=(2,), name="scalar_input")
    y = layers.Dense(16, activation="relu")(scalar_input)

    combined = layers.Concatenate()([x, y])
    z = layers.Dense(16, activation="relu")(combined)
    z = layers.Dropout(dropout)(z)
    outputs = layers.Dense(2, activation="softmax")(z)

    model = keras.Model([temporal_input, scalar_input], outputs)
    return _compile_model(model, learning_rate=learning_rate)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer(
    input_shape=(TYPE_WINDOW_FRAMES, 10),
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=128,
    dropout=0,
    mlp_dropout=0,
    learning_rate: float = 0.00005,
):
    n_classes = 6

    temporal_input = keras.Input(shape=input_shape, name="temporal_input")
    x = layers.BatchNormalization()(temporal_input)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(mlp_units, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
    x = layers.Dense(16, activation="relu")(x)

    scalar_input = keras.Input(shape=(2,), name="scalar_input")
    y = layers.Dense(16, activation="relu")(scalar_input)

    combined = layers.Concatenate()([x, y])
    z = layers.Dense(16, activation="relu")(combined)
    z = layers.Dropout(0.2)(z)
    outputs = layers.Dense(n_classes, activation="softmax")(z)

    model = keras.Model([temporal_input, scalar_input], outputs)
    return _compile_model(model, learning_rate=learning_rate)


def inception_time(
    input_shape=(TYPE_WINDOW_FRAMES, 10),
    filters: int = 32,
    bottleneck_filters: int = 32,
    modules: int = 3,
    dropout: float = 0.2,
    learning_rate: float = 0.00003,
):
    n_classes = 6

    temporal_input = keras.Input(shape=input_shape, name="temporal_input")
    x = temporal_input
    residual = x

    for module_index in range(modules):
        x = _inception_module(x, filters=filters, bottleneck_filters=bottleneck_filters)
        if module_index % 2 == 1:
            if residual.shape[-1] != x.shape[-1]:
                residual = layers.Conv1D(int(x.shape[-1]), kernel_size=1, padding="same")(residual)
            x = layers.Add()([x, residual])
            x = layers.Activation("relu")(x)
            residual = x

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    scalar_input = keras.Input(shape=(2,), name="scalar_input")
    y = layers.Dense(16, activation="relu")(scalar_input)

    combined = layers.Concatenate()([x, y])
    z = layers.Dense(32, activation="relu")(combined)
    z = layers.Dropout(dropout)(z)
    outputs = layers.Dense(n_classes, activation="softmax")(z)

    model = keras.Model([temporal_input, scalar_input], outputs)
    return _compile_model(model, learning_rate=learning_rate)


def build_model(task: str, architecture: str, **overrides):
    if task == "type":
        if architecture == "transformer":
            params = {"dropout": 0.3, "mlp_dropout": 0.1}
            params.update(overrides)
            return transformer(**params)
        if architecture == "inceptiontime":
            return inception_time(**overrides)
    elif task == "success":
        if architecture == "lstm":
            return lstm(**overrides)
        if architecture == "tcn":
            return tcn_success(**overrides)

    raise ValueError(f"Unsupported model combination: task={task}, architecture={architecture}")


def default_architecture(task: str) -> str:
    if task == "type":
        return "inceptiontime"
    if task == "success":
        return "tcn"
    raise ValueError(f"Unsupported task: {task}")


def transformerTraining(hp):
    input_shape = (TYPE_WINDOW_FRAMES, 10)
    head_size = hp.Int("head_size", min_value=32, max_value=512, step=32)
    num_heads = hp.Int("num_heads", min_value=2, max_value=16, step=2)
    ff_dim = hp.Int("ff_dim", min_value=128, max_value=2048, step=128)
    num_transformer_blocks = hp.Int("num_transformer_blocks", min_value=1, max_value=12, step=1)
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
    return _compile_model(model, learning_rate=learning_rate)


def save_model(model, path="saved_models/model.keras"):
    keras.saving.save_model(model, path, overwrite=True)


class LegacySavedModelPredictor:
    def __init__(self, path: str):
        self.path = path
        self._loaded = tf.saved_model.load(path)
        self._signature = self._loaded.signatures["serving_default"]
        self._output_key = next(iter(self._signature.structured_outputs))

    def predict(self, data, verbose: int = 0):
        if isinstance(data, dict):
            inputs = {
                key: tf.convert_to_tensor(value, dtype=tf.float32)
                for key, value in data.items()
            }
        else:
            inputs = {"temporal_input": tf.convert_to_tensor(data, dtype=tf.float32)}
        outputs = self._signature(**inputs)
        return outputs[self._output_key].numpy()


def _legacy_saved_model_dir(path: str) -> str | None:
    candidate = path
    if tf.io.gfile.isdir(candidate):
        return candidate
    if path.endswith(".keras") or path.endswith(".h5"):
        fallback = path.rsplit(".", 1)[0]
        if tf.io.gfile.isdir(fallback):
            return fallback
    return None


def load_model(path="saved_models/model.keras", for_training: bool = False):
    legacy_dir = _legacy_saved_model_dir(path)
    should_use_legacy = legacy_dir is not None and (
        tf.io.gfile.isdir(path) or not tf.io.gfile.exists(path)
    )
    if should_use_legacy:
        if for_training:
            raise ValueError(
                f"Pretrained model '{path}' uses legacy SavedModel format and cannot be reused for training with Keras 3. "
                "Retrain once to create a new .keras checkpoint."
            )
        return LegacySavedModelPredictor(legacy_dir)
    return keras.saving.load_model(path)
