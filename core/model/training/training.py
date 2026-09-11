import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

import core.model.model
import core.model.training.loader as loader


class Trainer:
    EARLY_STOPPING_PATIENCE = 40
    def __init__(self, dataset : loader.Dataset, model: keras.models.Model, model_filepath: str):
        self.dataset = dataset
        self.model = model

        self.loss_fn = keras.losses.CategoricalCrossentropy()  # from_logits?

        self.accuracy = tf.keras.metrics.CategoricalAccuracy()

        self.model_filepath = model_filepath

    def model_save_best(self, path):
        return keras.callbacks.ModelCheckpoint(
            path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

    def model_load_best(self, path):
        return keras.models.load_model(path)

    def plot(self, path):
        self.plot_confusion_matrix(path)

    def plot_confusion_matrix(self, path):
        model = self.model_load_best(path)

        y_pred = model.predict({"temporal_input" : self.dataset.temporal_features_test, "scalar_input" : self.dataset.scalar_features_test})
        y_pred2 = []
        for x in y_pred:
            y_pred2.append(np.argmax(x))
        y_true = []
        for x in self.dataset.labels_test:
            y_true.append(np.argmax(x))
        results = confusion_matrix(y_true, y_pred2)
        print(results)

    def evaluate_best_model(self, path):
        model = self.model_load_best(path)
        y_pred = model.predict(
            {"temporal_input": self.dataset.temporal_features_test, "scalar_input": self.dataset.scalar_features_test},
            verbose=0,
        )
        predicted_labels = [int(np.argmax(x)) for x in y_pred]
        true_labels = [int(np.argmax(x)) for x in self.dataset.labels_test]
        return {
            "test_accuracy": float(accuracy_score(true_labels, predicted_labels)),
            "confusion_matrix": confusion_matrix(true_labels, predicted_labels).tolist(),
            "test_samples": int(len(true_labels)),
        }

    def early_stopping(self):
        """Stop long runs once validation accuracy no longer improves."""
        return keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=self.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )

    def _history_summary(self, history):
        history_data = history.history if history is not None else {}
        val_accuracy_history = history_data.get("val_accuracy", [])
        train_accuracy_history = history_data.get("accuracy", [])
        val_loss_history = history_data.get("val_loss", [])
        train_loss_history = history_data.get("loss", [])
        return {
            "epochs_ran": int(len(train_accuracy_history) or len(val_accuracy_history)),
            "final_train_accuracy": float(train_accuracy_history[-1]) if train_accuracy_history else None,
            "best_val_accuracy": float(max(val_accuracy_history)) if val_accuracy_history else None,
            "final_val_accuracy": float(val_accuracy_history[-1]) if val_accuracy_history else None,
            "final_train_loss": float(train_loss_history[-1]) if train_loss_history else None,
            "final_val_loss": float(val_loss_history[-1]) if val_loss_history else None,
            "history": {
                "accuracy": [float(value) for value in train_accuracy_history],
                "val_accuracy": [float(value) for value in val_accuracy_history],
                "loss": [float(value) for value in train_loss_history],
                "val_loss": [float(value) for value in val_loss_history],
            },
        }

    def _dataset_training_metadata(self):
        return {
            "class_weight": dict(self.dataset.class_weight or {}),
            "stratified_split_used": bool(self.dataset.stratified_split_used),
            "class_counts": dict(self.dataset.class_counts or {}),
        }

    def train(self, epochs: int = 100, plot: bool = True, batch_size: int | None = None):
        """
        Do the training, and plot the confusion matrix and losses through epochs
        :param epochs:
        :param plot:
        :return: none
        """

        self.model.summary()
        history = None
        try:
            history = self.model.fit(
                {"temporal_input" : self.dataset.temporal_features_train, "scalar_input" : self.dataset.scalar_features_train},
                self.dataset.labels_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=self.dataset.val_dataset,
                callbacks=[self.model_save_best(self.model_filepath), self.early_stopping()],
            )
        except KeyboardInterrupt:
            self.plot(self.model_filepath)

        if plot:
            self.model = core.model.model.load_model(self.model_filepath)
            self.plot(self.model_filepath)
        summary = self._history_summary(history)
        summary.update(self.evaluate_best_model(self.model_filepath))
        summary.update(self._dataset_training_metadata())
        return summary

    def train_success(self, epochs: int = 100, plot: bool = True, batch_size: int | None = None):
        """
        Do the training, and plot the confusion matrix and losses through epochs
        :param epochs:
        :param plot:
        :return: none
        """

        self.model.summary()
        history = None
        try:
            history = self.model.fit(
                {"temporal_input" : self.dataset.temporal_features_train, "scalar_input" : self.dataset.scalar_features_train},
                self.dataset.labels_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=self.dataset.val_dataset,
                callbacks=[self.model_save_best(self.model_filepath), self.early_stopping()],
                class_weight=self.dataset.class_weight or None,
            )
        except KeyboardInterrupt:
            self.plot(self.model_filepath)

        if plot:
            self.model = core.model.model.load_model(self.model_filepath)
            self.plot(self.model_filepath)
        summary = self._history_summary(history)
        summary.update(self.evaluate_best_model(self.model_filepath))
        summary.update(self._dataset_training_metadata())
        return summary
