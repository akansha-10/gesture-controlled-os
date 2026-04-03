import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

from .utils import train_test_split
from .hand import Hand


class ClassificationModel:
    """
    MLP-based hand pose classification model.
    """

    n_features = 42  # 21 landmarks * (x, y)

    def __init__(self):
        self.num_classes = len(Hand.Pose)
        self.model = Sequential()

    def read_sample(self, path):
        with open(path, "r") as f:
            label = int(f.readline())
            features = list(map(np.float32, f.readline().split()))
        return features, label

    def read_dataset(self, dataset_path):
        files = sorted(
            [
                os.path.join(dataset_path, f)
                for f in os.listdir(dataset_path)
                if f.endswith(".dat")
            ]
        )

        X = np.empty((len(files), self.n_features), dtype=np.float32)
        y = np.empty(len(files), dtype=np.int32)

        for i, file in enumerate(files):
            X[i], y[i] = self.read_sample(file)

        self.data = X
        self.labels = y

    def preprocess(self):
        for i in range(len(self.data)):
            self.data[i, 0::2] -= self.data[i, 0::2].mean()
            self.data[i, 1::2] -= self.data[i, 1::2].mean()

    def train(
        self,
        hidden_layers=(50, 25, 10),
        learning_rate=0.01,
        epochs=15,
        test_size=0.3,
    ):
        for i, units in enumerate(hidden_layers):
            if i == 0:
                self.model.add(
                    Dense(units, activation="relu", input_shape=(self.n_features,))
                )
            elif i < len(hidden_layers) - 1:
                self.model.add(Dense(units, activation="relu"))
            else:
                self.model.add(Dense(self.num_classes, activation="softmax"))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=test_size
        )

        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            verbose=2,
        )

    def save(self, path):
        self.model.save(path)
